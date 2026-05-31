"""Phase 7 — RAG: embed chunks + retrieve (เลี่ยง sentence_transformers ที่ segfault)

ใช้ intfloat/multilingual-e5-small ผ่าน transformers ตรงๆ (mean-pool) + faiss
- build index จาก data/chunks.jsonl (cache → artifacts/rag/)
- retrieve(query, k) คืน chunk ที่เกี่ยวสุด

รัน:  .venv\\Scripts\\python.exe scripts\\phase7_rag.py --build
      .venv\\Scripts\\python.exe scripts\\phase7_rag.py --query "นิสิตลงทะเบียนเรียนที่ไหน"
"""
import argparse, json, os
from _bootstrap import bootstrap
bootstrap()
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
RAG = os.path.join(ROOT, "artifacts", "rag")
EMB_MODEL = "intfloat/multilingual-e5-small"   # 384-dim, ไทย-อังกฤษดี, เล็ก

_tok = _mdl = None
def _load():
    global _tok, _mdl
    if _mdl is None:
        _tok = AutoTokenizer.from_pretrained(EMB_MODEL)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _mdl = AutoModel.from_pretrained(EMB_MODEL).to(dev).eval()
    return _tok, _mdl


@torch.no_grad()
def embed(texts, prefix):
    """e5 ต้องใส่ prefix 'query: ' หรือ 'passage: '"""
    tok, mdl = _load()
    out = []
    for i in range(0, len(texts), 32):
        batch = [prefix + t for t in texts[i:i + 32]]
        enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(mdl.device)
        h = mdl(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)   # mean pool
        out.append(F.normalize(emb, p=2, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0).astype("float32")


def build():
    os.makedirs(RAG, exist_ok=True)
    chunks = [json.loads(l) for l in open(os.path.join(DATA, "chunks.jsonl"), encoding="utf-8")]
    texts = [c["text"] for c in chunks]
    print(f"embedding {len(texts)} chunks …")
    emb = embed(texts, "passage: ")
    np.save(os.path.join(RAG, "emb.npy"), emb)
    json.dump([{"text": c["text"], "source": c.get("source", ""), "title": c.get("title", "")}
               for c in chunks], open(os.path.join(RAG, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"✅ index → {RAG} (emb {emb.shape})")


def retrieve(query, k=4):
    import faiss
    emb = np.load(os.path.join(RAG, "emb.npy"))
    meta = json.load(open(os.path.join(RAG, "meta.json"), encoding="utf-8"))
    index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
    q = embed([query], "query: ")
    D, I = index.search(q, k)
    return [{"score": float(D[0][j]), **meta[I[0][j]]} for j in range(k)]


SERVER = "http://127.0.0.1:8080"   # llama-server (IPv4 — เลี่ยง localhost→IPv6)

SYS_RAG = ("คุณคือผู้ช่วยตอบคำถามของสำนักงานการทะเบียน จุฬาลงกรณ์มหาวิทยาลัย "
           "ตอบโดยอิงข้อมูลใน 'บริบท' ที่ให้เท่านั้น ถ้าไม่มีข้อมูลให้บอกว่าไม่ทราบ ตอบกระชับเป็นภาษาไทย")


def ask(query, k=4, use_rag=True, max_tokens=200, server=SERVER):
    """RAG: retrieve context → ส่ง llama-server (OpenAI-compatible)"""
    import requests
    if use_rag:
        ctx = retrieve(query, k)
        context = "\n\n".join(f"[{i+1}] {c['text'][:600]}" for i, c in enumerate(ctx))
        messages = [{"role": "system", "content": SYS_RAG},
                    {"role": "user", "content": f"บริบท:\n{context}\n\nคำถาม: {query}"}]
    else:
        messages = [{"role": "user", "content": query}]
        ctx = []
    r = requests.post(f"{server}/v1/chat/completions",
                      json={"messages": messages, "temperature": 0.0, "max_tokens": max_tokens},
                      timeout=120)
    ans = r.json()["choices"][0]["message"]["content"].strip()
    return ans, ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--query", default=None, help="แสดง retrieval อย่างเดียว")
    ap.add_argument("--ask", default=None, help="RAG generation เต็ม (ต้อง start llama-server ก่อน)")
    ap.add_argument("--no-rag", action="store_true", help="ถามตรงๆ ไม่ใช้ RAG (เทียบ)")
    ap.add_argument("-k", type=int, default=4)
    args = ap.parse_args()
    if args.build:
        build()
    if args.query:
        for r in retrieve(args.query, args.k):
            print(f"\n[{r['score']:.3f}] {r['title'][:50]} | {r['source'][:60]}")
            print("  " + r["text"][:200].replace("\n", " "))
    if args.ask:
        ans, ctx = ask(args.ask, args.k, use_rag=not args.no_rag)
        if ctx:
            print("บริบทที่ดึงมา:", ", ".join(f"{c['score']:.2f}" for c in ctx))
        print(f"\nQ: {args.ask}\nA: {ans}")


if __name__ == "__main__":
    main()
