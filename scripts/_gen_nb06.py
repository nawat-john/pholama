# -*- coding: utf-8 -*-
"""สร้าง notebooks/06_quantize.ipynb (GGUF via llama.cpp, รัน Kaggle/Colab/CPU-Linux)
รัน:  .venv\\Scripts\\python.exe scripts\\_gen_nb06.py
"""
import json, os

cells = []
def md(s):  cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s):cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(keepends=True)})

md(r"""# Phase 6 — Quantization (GGUF via llama.cpp)

บีบขนาด/RAM สุดท้าย • สร้างหลายระดับเทียบกัน (Q4_K_M / Q5_K_M / Q8_0 + f16 อ้างอิง)

**Input (attach Kaggle Dataset):** โมเดล merge แล้วจาก Phase 5 `recovered/` (หรือ pruned ตรงๆ ถ้าข้าม recovery)
**Output → Kaggle Dataset:** ไฟล์ `.gguf` หลายระดับ + `quant_report.json`

| ระดับ | bits/w โดยประมาณ | ใช้เมื่อ |
|---|---|---|
| Q8_0  | 8.5 | อ้างอิงคุณภาพ (เกือบเท่า f16) |
| Q5_K_M | 5.5 | คุณภาพสูง |
| **Q4_K_M** | 4.5 | **สมดุล size/quality (แนะนำ deploy)** |

---
## 🧭 ทำทีละขั้น
1. Kaggle: Accelerator = **None/CPU** ก็พอ (quantize ใช้ CPU). Add Input → attach dataset ที่มี `recovered/`
2. แก้ `HF_DIR` ให้ตรง
3. รันทุก cell: build llama.cpp → **patch convert (trimmed tokenizer)** → HF→GGUF f16 → quantize → ตารางขนาด → ทดสอบ generate แต่ละระดับ
4. Save Version → Dataset ไฟล์ `.gguf` ให้ Phase 7

> ⚠️ **gotcha โมเดลนี้:** vocab ถูก trim (Phase 3) → `convert_hf_to_gguf.py` จะหา pre-tokenizer hash ไม่เจอ (raise error) — cell patch ด้านล่างบังคับเป็น `llama-bpe` (regex เหมือน Llama3 เดิม ไม่ได้แก้ pre_tokenizer)""")

md("## ขั้น 0 — paths + ระดับ quantization")
code(r"""
import os, glob, json, subprocess
INPUT = "/kaggle/input"; OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
DS = "pholama-phase5"                       # <-- TODO ชื่อ dataset
HF_DIR = f"{INPUT}/{DS}/recovered"          # โมเดล merge จาก Phase 5
LEVELS = ["Q8_0", "Q5_K_M", "Q4_K_M"]
LCPP = os.path.join(OUT, "llama.cpp")
assert os.path.isdir(HF_DIR), f"ไม่พบ {HF_DIR} — attach dataset + แก้ DS/HF_DIR"
print("HF_DIR:", HF_DIR)
""")

md("## ขั้น 1 — build llama.cpp (clone + cmake สำหรับ llama-quantize/llama-cli)")
code(r"""
if not os.path.isdir(LCPP):
    !git clone --depth 1 https://github.com/ggml-org/llama.cpp {LCPP}
!pip install -q -r {LCPP}/requirements/requirements-convert_hf_to_gguf.txt
# build เฉพาะ tool ที่ใช้ (เร็วกว่า build ทั้งหมด)
!cmake -S {LCPP} -B {LCPP}/build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF > /tmp/cmake.log 2>&1 && \
 cmake --build {LCPP}/build -j --config Release -t llama-quantize llama-cli > /tmp/build.log 2>&1 && echo "build OK" || tail -30 /tmp/build.log
QUANT = f"{LCPP}/build/bin/llama-quantize"; CLI = f"{LCPP}/build/bin/llama-cli"
print("quantize:", os.path.exists(QUANT), "| cli:", os.path.exists(CLI))
""")

md(r"""## ขั้น 2 — patch convert สำหรับ trimmed tokenizer (สำคัญ)
vocab ถูก trim → hash ไม่ตรง model ที่ llama.cpp รู้จัก → `get_vocab_base_pre()` จะ raise
patch ให้ fallback เป็น `llama-bpe` (ปลอดภัยเพราะ pre_tokenizer regex = Llama3 เดิม)""")
code(r"""
import glob
# needle อาจอยู่ใน convert_hf_to_gguf.py หรือ conversion/base.py (แล้วแต่เวอร์ชัน) → ค้นทั้ง repo
needle = 'raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")'
repl   = 'logger.warning("llama-bpe fallback (trimmed Llama3 tokenizer)"); res = "llama-bpe"'
n = 0
for f in glob.glob(f"{LCPP}/**/*.py", recursive=True):
    s = open(f, encoding="utf-8").read()
    if needle in s:
        open(f, "w", encoding="utf-8").write(s.replace(needle, repl)); n += 1; print("patched:", f)
print(f"✅ patched {n} file(s) → fallback llama-bpe" if n else
      "⚠️ ไม่พบ needle — ถ้า convert error เรื่อง pre-tokenizer ให้ค้น 'BPE pre-tokenizer was not recognized' แล้วแก้เป็น res='llama-bpe'")
""")

md("## ขั้น 3 — HF → GGUF (f16)")
code(r"""
f16 = os.path.join(OUT, "model-f16.gguf")
!python {LCPP}/convert_hf_to_gguf.py {HF_DIR} --outfile {f16} --outtype f16
print("f16:", round(os.path.getsize(f16)/1e9, 2), "GB")
""")

md("## ขั้น 4 — quantize หลายระดับ")
code(r"""
made = {"f16": f16}
for lv in LEVELS:
    out = os.path.join(OUT, f"model-{lv}.gguf")
    !{QUANT} {f16} {out} {lv}
    made[lv] = out
    print(lv, "→", round(os.path.getsize(out)/1e9, 2), "GB")
""")

md("## ขั้น 5 — ตารางขนาด + ประมาณ RAM")
code(r"""
sizes = {k: round(os.path.getsize(v)/1e9, 3) for k, v in made.items()}
print("ขนาดไฟล์ (GB):"); [print(f"  {k:8s}: {sizes[k]}") for k in sizes]
# RAM ตอนรัน ≈ ขนาด weights + KV cache + overhead (~+0.3–0.5GB)
print("\nประมาณ RAM ตอนรัน (Q4_K_M + ctx 2048) ≈", round(sizes.get('Q4_K_M',0)+0.4,2), "GB")
""")

md(r"""## ขั้น 6 — ทดสอบ generate แต่ละระดับ (เทียบคุณภาพ)
รันคำถามโดเมนผ่านแต่ละ gguf เทียบสายตา (คาด Q8≈Q5>Q4 แต่ Q4 ควรยังตอบรู้เรื่อง)""")
code(r"""
import shlex
PROMPT = "นิสิตจุฬาฯ ลงทะเบียนเรียนได้ที่ไหน"
def gen(gguf):
    cmd = f'{CLI} -m {gguf} -p {shlex.quote(PROMPT)} -n 60 -ngl 0 --temp 0 -no-cnv 2>/dev/null'
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()[-400:]
for lv in ["Q8_0","Q4_K_M"]:
    print(f"\n=== {lv} ===\n{gen(made[lv])}")
""")

md("## ขั้น 7 — เซฟ report + push")
code(r"""
report = {"source": os.path.basename(HF_DIR), "sizes_GB": sizes, "levels": LEVELS,
          "recommend": "Q4_K_M", "ram_estimate_Q4_GB": round(sizes.get('Q4_K_M',0)+0.4,2)}
json.dump(report, open(os.path.join(OUT,"quant_report.json"),"w"), ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
print("\nไฟล์ .gguf:", [os.path.basename(p) for p in glob.glob(os.path.join(OUT,'*.gguf'))])
""")

md(r"""---
## ✅ เสร็จ Phase 6 — ขั้นต่อไป
1. Save Version → Dataset ไฟล์ `.gguf` (อย่างน้อย Q4_K_M + Q8_0 อ้างอิง)
2. ⚠️ **3 baselines:** ต้อง quantize **B2** (original+QLoRA, ไม่ prune) ด้วย เพื่อเทียบใน Phase 7 — รัน notebook นี้ซ้ำกับโมเดล B2
3. **Phase 7 (`07_eval_deploy.ipynb`)**: Ollama + RAG (FAISS/Chroma) + eval 3 แกน (in-domain / out-of-domain regression / efficiency) × 3 baselines
4. อัปเดต `PROGRESS.md` + `experiments.csv` (ขนาด/RAM แต่ละระดับ)

**หมายเหตุรัน local Windows:** `convert_hf_to_gguf.py` รันได้ด้วย `pip install gguf` (ไม่ต้อง build) → ได้ f16 gguf • ส่วน `llama-quantize` ใช้ binary สำเร็จรูปจาก [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) (Windows) ได้เลย""")

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(__file__),"..","notebooks","06_quantize.ipynb")
json.dump(nb, open(out,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("เขียน", out, "|", len(cells), "cells")
