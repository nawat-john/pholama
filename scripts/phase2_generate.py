"""Phase 2 — ขั้น 2: gen Q&A จาก chunks ด้วย Gemini

- grounded: คำตอบต้องอิงเฉพาะข้อความใน chunk ห้ามแต่ง
- หลากหลาย: ทางการ/ไม่ทางการ, ไทย/อังกฤษ/ปนกัน
- คำถามต้อง standalone (ห้ามอ้าง "จากเอกสารนี้/ตามข้อความข้างต้น")
- JSON mode (response_mime_type) ตัดปัญหา markdown fence
- resumable: ข้าม chunk ที่ทำไปแล้ว (เช็คจาก data/qa_raw.jsonl)

รัน:  .venv\Scripts\python.exe scripts\phase2_generate.py --pairs 6
      เพิ่ม --limit 10 เพื่อทดสอบ
"""
import argparse
import hashlib
import json
import os
import sys
import time

from _bootstrap import bootstrap
bootstrap()

import google.generativeai as genai

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
CHUNKS = os.path.join(DATA_DIR, "chunks.jsonl")
OUT = os.path.join(DATA_DIR, "qa_raw.jsonl")

GEN_MODEL = "gemini-2.5-flash-lite"   # free RPD สูง (~1000) + เร็ว; พอสำหรับ Q&A แบบ grounded

PROMPT = """คุณคือผู้ช่วยสร้างชุดข้อมูลถาม-ตอบ (Q&A) สำหรับ "สำนักงานการทะเบียน จุฬาลงกรณ์มหาวิทยาลัย" (reg.chula.ac.th)

จากข้อความอ้างอิงด้านล่าง จงสร้างคู่ถาม-ตอบจำนวน {n} คู่ ตามกติกา:
- คำตอบต้องอิงข้อเท็จจริงจากข้อความอ้างอิง "เท่านั้น" ห้ามแต่งข้อมูลที่ไม่มี ถ้าข้อความมีข้อมูลไม่พอให้สร้างน้อยกว่า {n} คู่ได้
- คำถามต้องเป็นแบบ standalone เข้าใจได้เอง ห้ามอ้าง "จากเอกสารนี้/ตามข้อความข้างต้น" และให้ระบุบริบทในคำถาม (เช่นชื่อบริการ/ปีการศึกษา) ให้ชัด
- ใช้คำว่า "นิสิต" (ไม่ใช่ "นักศึกษา") ตามที่จุฬาฯ ใช้
- คละสไตล์และภาษา: ทางการ/ไม่ทางการ และ ไทยล้วน/อังกฤษล้วน/ไทยปนอังกฤษ ให้หลากหลายภายในชุดนี้
- คำตอบกระชับ ตรงคำถาม ถ้ามีวันเวลา/ค่าธรรมเนียม/ขั้นตอน ให้ระบุให้ครบตามข้อความ

ตอบเป็น JSON array เท่านั้น แต่ละสมาชิก:
{{"question": "...", "answer": "...", "lang": "th|en|mix", "style": "formal|casual"}}

ข้อความอ้างอิง (หัวข้อ: {title}):
\"\"\"
{context}
\"\"\""""

SAFETY = [
    {"category": c, "threshold": "BLOCK_NONE"}
    for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
              "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
]


def chunk_id(c):
    return hashlib.md5((c["source"] + "|" + c["text"][:80]).encode("utf-8")).hexdigest()[:12]


def load_done():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                done.add(json.loads(line)["_cid"])
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=GEN_MODEL, help="ชื่อ Gemini model")
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="จำกัดจำนวน chunk (ทดสอบ)")
    ap.add_argument("--delay", type=float, default=4.5, help="หน่วงระหว่าง call (กัน RPM limit)")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--quota-stop", type=int, default=4, help="หยุดเมื่อ 429 ติดกันกี่ chunk")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("ไม่พบ GEMINI_API_KEY ใน .env")
    genai.configure(api_key=api_key, transport="rest")

    # multi-model fallback: free-tier โควต้าแยกราย model-name → หมดตัวนึงสลับตัวถัดไป
    model_names = [m.strip() for m in args.model.split(",") if m.strip()]

    def make_model(name):
        return genai.GenerativeModel(
            name,
            generation_config={"response_mime_type": "application/json", "temperature": 0.9},
            safety_settings=SAFETY,
        )

    midx = 0
    model = make_model(model_names[midx])
    print(f"models (fallback chain): {model_names}\nactive: {model_names[midx]}")

    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    if args.limit:
        chunks = chunks[:args.limit]
    done = load_done()
    print(f"chunks={len(chunks)} | ทำไปแล้ว={len(done)} | เหลือ={len(chunks) - sum(1 for c in chunks if chunk_id(c) in done)}")

    out_f = open(OUT, "a", encoding="utf-8")
    total_new = 0
    consec_quota = 0          # นับ chunk ที่ติด 429 ติดกัน (circuit breaker)
    t0 = time.time()
    for i, c in enumerate(chunks):
        cid = chunk_id(c)
        if cid in done:
            continue
        prompt = PROMPT.format(n=args.pairs, title=c.get("title", "")[:120], context=c["text"])
        rows, quota_hit = None, False
        for attempt in range(args.max_retries):
            try:
                resp = model.generate_content(prompt, request_options={"timeout": 90})
                rows = json.loads(resp.text)
                if isinstance(rows, dict):
                    rows = rows.get("data") or rows.get("qa") or [rows]
                break
            except Exception as e:
                msg = str(e)[:120]
                is_quota = "429" in msg or "quota" in msg.lower() or "exceeded" in msg.lower()
                if is_quota:
                    quota_hit = True   # observed: 429 ที่นี่คงค้าง (daily) ไม่ใช่ RPM → ไม่รอ ไม่เปลือง retry
                    break
                # error อื่น (parse/transient) — retry สั้นๆ
                if attempt == args.max_retries - 1:
                    print(f"  [{i}] ล้มเหลว: {msg}")
                else:
                    time.sleep(3 * (attempt + 1))
        # circuit breaker: 429 ติดกัน = model นี้โควต้าหมด → สลับ model ถัดไปใน chain
        if rows is None and quota_hit:
            consec_quota += 1
            if consec_quota >= args.quota_stop:
                if midx + 1 < len(model_names):
                    midx += 1
                    model = make_model(model_names[midx])
                    consec_quota = 0
                    print(f"\n🔁 โควต้าหมด — สลับไป model: {model_names[midx]} (chunk {i})")
                    continue
                print(f"\n⛔ ทุก model ในchain โควต้าหมด หยุดไว้ก่อน (resume ได้ภายหลัง)")
                break
            continue
        consec_quota = 0
        if not rows:
            continue
        n_ok = 0
        for r in rows:
            if not isinstance(r, dict):
                continue
            q, a = (r.get("question") or "").strip(), (r.get("answer") or "").strip()
            if not q or not a:
                continue
            rec = {"question": q, "answer": a,
                   "lang": r.get("lang", "th"), "style": r.get("style", "formal"),
                   "source": c["source"], "_cid": cid}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
        out_f.flush()
        total_new += n_ok
        if (i + 1) % 10 == 0 or i == len(chunks) - 1:
            rate = total_new / max(time.time() - t0, 1) * 60
            print(f"  [{i+1}/{len(chunks)}] +{n_ok} (รวมรอบนี้ {total_new}, ~{rate:.0f}/min)")
        time.sleep(args.delay)

    out_f.close()
    # นับรวมทั้งไฟล์
    total_all = sum(1 for _ in open(OUT, encoding="utf-8"))
    print(f"\n✅ เพิ่มรอบนี้ {total_new} คู่ | รวมใน qa_raw.jsonl = {total_all} คู่")


if __name__ == "__main__":
    main()
