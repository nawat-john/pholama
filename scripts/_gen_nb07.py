# -*- coding: utf-8 -*-
"""สร้าง notebooks/07_eval_deploy.ipynb — Deploy (RAG) + eval 3 แกน × 3 baselines
รัน:  .venv\\Scripts\\python.exe scripts\\_gen_nb07.py
"""
import json, os
cells = []
def md(s):  cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s):cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(keepends=True)})

md(r"""# Phase 7 — Deploy (RAG) + Eval + Report

ขั้นสุดท้าย: serve โมเดลที่ผ่าน pipeline ด้วย **RAG** แล้วประเมิน **3 แกน × 3 baselines** + เขียนรายงาน

**3 baselines (บังคับ):**
- **B1** = Typhoon2 3B เดิม (ไม่ทำอะไร)
- **B2** = เดิม + QLoRA + quantize (**ไม่ prune**) ← ต้องทำเพิ่มเพื่อเทียบ
- **B3** = pipeline เต็ม (vocab trim + prune + recovery + quantize) ← ที่ทำมา

**3 แกน eval:**
1. **in-domain** — accuracy บน test 403 ข้อ (RAG vs no-RAG), judge ด้วย LLM
2. **out-of-domain regression** — gsm8k/humaneval (ควร**ตก** = specialization สำเร็จ)
3. **efficiency** — size, RAM, tokens/sec

**คำถามรายงาน:** B3 ได้ขนาด/ความเร็วดีกว่า B2 จริงไหม แลกด้วยคุณภาพเท่าไหร่?

---
## 🧭 ทำทีละขั้น
1. attach dataset: GGUF (Phase 6) ของ B2 + B3, `test.jsonl`, `chunks.jsonl`
2. ติดตั้ง llama.cpp (serve GGUF) + embedding model (RAG)
3. รัน RAG eval (in-domain) ทั้ง B2/B3 → ตาราง
4. รัน lm-eval (gsm8k/humaneval) ทั้ง B1/B2/B3 → ตาราง regression
5. รวม efficiency → เขียน REPORT""")

md("## ขั้น 0 — setup")
code(r"""
%pip install -q -U transformers faiss-cpu requests
import os, json, subprocess, time, requests, numpy as np, torch
import torch.nn.functional as F
INPUT="/kaggle/input"; OUT="/kaggle/working" if os.path.isdir("/kaggle/working") else "."
DS="pholama-phase6"   # <-- TODO มี GGUF + test.jsonl + chunks.jsonl
GGUF_B3=f"{INPUT}/{DS}/recovered-Q4_K_M.gguf"
GGUF_B2=f"{INPUT}/{DS}/b2-Q4_K_M.gguf"      # ถ้ามี (ไม่ prune)
TEST=f"{INPUT}/{DS}/test.jsonl"; CHUNKS=f"{INPUT}/{DS}/chunks.jsonl"
""")

md(r"""## ขั้น 1 — RAG: embed chunks + retrieve
ใช้ `intfloat/multilingual-e5-small` ผ่าน transformers (เลี่ยง sentence_transformers)""")
code(r"""
from transformers import AutoTokenizer, AutoModel
et=AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
em=AutoModel.from_pretrained("intfloat/multilingual-e5-small").to("cuda").eval()
@torch.no_grad()
def embed(texts, prefix):
    out=[]
    for i in range(0,len(texts),32):
        enc=et([prefix+t for t in texts[i:i+32]],padding=True,truncation=True,max_length=512,return_tensors="pt").to("cuda")
        h=em(**enc).last_hidden_state; m=enc["attention_mask"].unsqueeze(-1).float()
        e=(h*m).sum(1)/m.sum(1).clamp(min=1e-9); out.append(F.normalize(e,2,1).cpu().numpy())
    return np.concatenate(out).astype("float32")
chunks=[json.loads(l) for l in open(CHUNKS,encoding="utf-8")]
import faiss
cemb=embed([c["text"] for c in chunks],"passage: ")
index=faiss.IndexFlatIP(cemb.shape[1]); index.add(cemb)
def retrieve(q,k=4):
    D,I=index.search(embed([q],"query: "),k); return [chunks[i]["text"] for i in I[0]]
print("indexed",len(chunks),"chunks")
""")

md("## ขั้น 2 — serve GGUF (llama-server) + ฟังก์ชัน ask (RAG)")
code(r"""
# build/serve llama.cpp — ดู notebook 06 สำหรับ build; ที่นี่สมมติมี ./llama.cpp/build/bin/llama-server
SYS="คุณคือผู้ช่วยสำนักงานการทะเบียน จุฬาฯ ตอบโดยอิงบริบทที่ให้เท่านั้น กระชับ เป็นไทย"
def start_server(gguf, port=8080):
    p=subprocess.Popen(["./llama.cpp/build/bin/llama-server","-m",gguf,"--port",str(port),
                        "-c","4096","-ngl","999","--jinja"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            if requests.get(f"http://127.0.0.1:{port}/health",timeout=2).json().get("status")=="ok": return p
        except: time.sleep(2)
    return p
def ask(q,port=8080,use_rag=True,max_tokens=200):
    if use_rag:
        ctx="\n\n".join(f"[{i+1}] {c[:600]}" for i,c in enumerate(retrieve(q)))
        msgs=[{"role":"system","content":SYS},{"role":"user","content":f"บริบท:\n{ctx}\n\nคำถาม: {q}"}]
    else: msgs=[{"role":"user","content":q}]
    r=requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"messages":msgs,"temperature":0,"max_tokens":max_tokens},timeout=120)
    return r.json()["choices"][0]["message"]["content"].strip()
""")

md(r"""## ขั้น 3 — in-domain eval (B2 & B3) ด้วย LLM-judge
รัน test ผ่านแต่ละ baseline (RAG) → judge correct/partial/wrong (ใช้ Gemini หรือโมเดลใหญ่)""")
code(r"""
# โครง: วน test, ask(q) ต่อ baseline, judge → tally. ดู scripts/phase7_eval.py สำหรับ judge เต็ม
# srv=start_server(GGUF_B3); ... ; for ex in test[:100]: ans=ask(ex['question']); verdict=judge(...)
print("ดู scripts/phase7_eval.py (รัน local แล้ว) — คัดลอก judge logic มาที่นี่สำหรับ B2 ด้วย")
""")

md(r"""## ขั้น 4 — out-of-domain regression (lm-eval gsm8k/humaneval × B1/B2/B3)
**ควรเห็น B3 ตก** เทียบ B1 = พิสูจน์ specialization สำเร็จ""")
code(r"""
%pip install -q lm-eval
# B1/B2 = HF model, B3 = pruned+recovered (bf16 ก่อน quantize) หรือ gguf ผ่าน gguf loader
# !lm_eval --model hf --model_args pretrained=<B3_bf16> --tasks gsm8k,hellaswag --limit 100 --output_path /kaggle/working/lmeval_b3
# ทำซ้ำ B1 (scb10x/llama3.2-typhoon2-3b-instruct) และ B2 → เก็บ acc เทียบ
print("รัน lm-eval ทีละ baseline (gsm8k = คณิต, humaneval = โค้ด) เก็บ accuracy")
""")

md("## ขั้น 5 — efficiency + REPORT")
code(r"""
# รวมตาราง 3 แกน × 3 baseline → เขียน REPORT.md
eff = {
 "B1 original":  {"params":"3.21B","weights":"6.43GB bf16 / 1.77GB 4bit","tok_s":"10.6 (GPU 4bit)"},
 "B2 +qlora+quant (no prune)": {"params":"3.21B","weights":"~1.9GB Q4","tok_s":"TODO"},
 "B3 full pipeline": {"params":"2.55B","weights":"1.53GB Q4_K_M","tok_s":"23.5 (CPU)"},
}
print(json.dumps(eff,ensure_ascii=False,indent=2))
# REPORT.md: ตอบคำถาม — B3 เล็ก/เร็วกว่า B2 ไหม แลกคุณภาพ in-domain เท่าไหร่ + OOD ตกจริงไหม
""")

md(r"""---
## ✅ Deliverable Phase 7
- demo RAG ที่รันได้ (ถาม-ตอบโดเมนจริง)
- ตาราง 3 แกน × 3 baseline (in-domain / OOD regression / efficiency)
- **REPORT.md** ตอบคำถามหลัก: pipeline เต็มคุ้มไหม
- README + push artifact

> สรุป thesis: B3 (2.55B, Q4 1.53GB, CPU 23.5 t/s) เล็ก/เร็วกว่าเดิม • in-domain กู้ด้วย RAG+recovery • OOD (code/math) ตก = specialization สำเร็จตามตั้งใจ""")

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(__file__),"..","notebooks","07_eval_deploy.ipynb")
json.dump(nb, open(out,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("เขียน", out, "|", len(cells), "cells")
