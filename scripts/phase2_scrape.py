"""Phase 2 — ขั้น 1: crawl reg.chula.ac.th/th/ → corpus + chunks

เก็บเฉพาะหน้า HTML ภาษาไทยใน host www.reg.chula.ac.th (และดึง PDF ระเบียบ/ประกาศที่ลิงก์ถึง)
ทำความสะอาดข้อความ ตัด nav/script/style แล้วแบ่งเป็น chunk

Output:
  data/chunks.jsonl     — {"source": url, "title": ..., "text": chunk}
  data/corpus_raw.txt   — ข้อความดิบทั้งหมด (สำหรับ vocab analysis Phase 3)
  data/scrape_log.json  — สรุป URL ที่ดึง/ข้าม

รัน:  .venv\Scripts\python.exe scripts\phase2_scrape.py --max-pages 150 --max-pdf 40
"""
import argparse
import io
import json
import os
import re
import time
from collections import deque
from urllib.parse import urljoin, urldefrag, urlparse

from _bootstrap import bootstrap
bootstrap()

import requests
from bs4 import BeautifulSoup

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
HOST = "www.reg.chula.ac.th"
SEED = "https://www.reg.chula.ac.th/th/"
HEADERS = {"User-Agent": "Mozilla/5.0 (pholama-research-crawler; contact nawatpim@gmail.com)"}

SKIP_EXT = re.compile(r"\.(jpg|jpeg|png|gif|svg|webp|css|js|ico|zip|rar|docx?|xlsx?|pptx?|mp4|mp3)(\?|$)", re.I)
PDF_EXT = re.compile(r"\.pdf(\?|$)", re.I)

CHUNK_CHARS = 1500
CHUNK_OVERLAP = 200


def clean_html(html):
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    for tag in soup(["script", "style", "nav", "footer", "header", "form", "noscript", "svg", "button"]):
        tag.extract()
    text = soup.get_text("\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text, soup


def chunk_text(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    text = text.strip()
    out, i = [], 0
    while i < len(text):
        ck = text[i:i + size].strip()
        if len(ck) >= 200:          # ตัด chunk สั้นเกินไป (เมนู/ขยะ)
            out.append(ck)
        i += size - overlap
    return out


def is_thai_doc(text):
    # หน้าที่มีตัวไทยพอสมควรเท่านั้น (กันหน้า en/ว่าง)
    thai = len(re.findall(r"[฀-๿]", text))
    return thai >= 150


def in_scope(url):
    p = urlparse(url)
    if p.netloc != HOST:
        return False
    if SKIP_EXT.search(p.path):
        return False
    return True


def extract_pdf(content):
    import pdfplumber
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        return "\n".join((pg.extract_text() or "") for pg in pdf.pages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=150)
    ap.add_argument("--max-pdf", type=int, default=40)
    ap.add_argument("--delay", type=float, default=0.4)
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    sess = requests.Session()
    sess.headers.update(HEADERS)

    seen = set()
    q = deque([SEED])
    pdf_urls = []
    chunks = []
    visited_pages, skipped = [], []

    def fetch(url, timeout=30):
        return sess.get(url, timeout=timeout)

    # ---- BFS หน้า HTML ----
    while q and len(visited_pages) < args.max_pages:
        url = urldefrag(q.popleft())[0]
        if url in seen or not in_scope(url):
            continue
        seen.add(url)
        if PDF_EXT.search(url):
            pdf_urls.append(url)
            continue
        try:
            r = fetch(url)
        except Exception as e:
            skipped.append([url, str(e)[:80]]); continue
        ct = r.headers.get("content-type", "")
        if r.status_code != 200 or "html" not in ct:
            skipped.append([url, f"status={r.status_code} ct={ct[:30]}"]); continue

        title, text, soup = clean_html(r.text)
        if is_thai_doc(text):
            n0 = len(chunks)
            for ck in chunk_text(text):
                chunks.append({"source": url, "title": title, "text": ck})
            visited_pages.append({"url": url, "title": title, "chars": len(text), "chunks": len(chunks) - n0})
        # หา link ต่อ
        for a in soup.find_all("a", href=True):
            nxt = urldefrag(urljoin(url, a["href"]))[0]
            if nxt not in seen and in_scope(nxt):
                if PDF_EXT.search(nxt):
                    if nxt not in pdf_urls:
                        pdf_urls.append(nxt)
                else:
                    q.append(nxt)
        print(f"[html {len(visited_pages):>3}] {len(chunks):>4} chunks | {title[:50]}")
        time.sleep(args.delay)

    # ---- ดึง PDF (ระเบียบ/ประกาศ มักเป็นทองของโดเมนนี้) ----
    for url in pdf_urls[:args.max_pdf]:
        try:
            r = fetch(url, timeout=60)
            if r.status_code != 200:
                skipped.append([url, f"pdf status={r.status_code}"]); continue
            text = re.sub(r"\n{3,}", "\n\n", extract_pdf(r.content)).strip()
        except Exception as e:
            skipped.append([url, "pdf:" + str(e)[:80]]); continue
        if is_thai_doc(text):
            title = os.path.basename(urlparse(url).path)
            for ck in chunk_text(text):
                chunks.append({"source": url, "title": title, "text": ck})
            print(f"[pdf ] {len(chunks):>4} chunks | {title[:50]}")
        time.sleep(args.delay)

    # ---- เขียน output ----
    with open(os.path.join(DATA_DIR, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(os.path.join(DATA_DIR, "corpus_raw.txt"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c["text"] + "\n")
    with open(os.path.join(DATA_DIR, "scrape_log.json"), "w", encoding="utf-8") as f:
        json.dump({"pages": visited_pages, "pdf_seen": len(pdf_urls),
                   "pdf_fetched": min(len(pdf_urls), args.max_pdf),
                   "chunks": len(chunks), "skipped": skipped[:50]},
                  f, ensure_ascii=False, indent=2)

    print(f"\n✅ pages={len(visited_pages)} pdf_links={len(pdf_urls)} chunks={len(chunks)} -> data/chunks.jsonl")


if __name__ == "__main__":
    main()
