# -*- coding: utf-8 -*-
"""สร้าง notebooks/04_pruning.ipynb (พร้อมรันบน Kaggle GPU) แบบ valid JSON
รัน:  .venv\\Scripts\\python.exe scripts\\_gen_nb04.py
"""
import json, os

cells = []
def md(s):  cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s):cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(keepends=True)})

md(r"""# Phase 4 — Structured Pruning (แกนของโปรเจกต์)

ตัด **layer** ที่สำคัญน้อย (ShortGPT) และ (ทางเลือก) ลด **FFN width** โดยใช้ `train.jsonl` โดเมนเป็น calibration → โมเดลเอนเข้าโดเมน

| สถานการณ์ | วิธี | param |
|---|---|---|
| **Moderate** | layer 28 → 24 | ~2.55B |
| **Aggressive** | layer 28 → 24 + FFN 8192 → 6144 | ~2.1B |

> ⚠️ **finding จากการรันจริง (local RTX 3050):** ShortGPT one-shot ตัด 8 layer (28→20) **พังโมเดล** — ppl 5.8→237, generate ขยะ เพราะตัด layer ติดกันยาว (residual กระโดด). sweep หาจุด knee: ตัด **4 layer (28→24)** ppl 13.2 ยัง coherent = จุดพอดี. ตั้ง`TARGET_LAYERS=24`. ดูตาราง sweep ใน `scripts/phase4_sweep.py`

**Input (attach เป็น Kaggle Dataset):**
- โมเดล Phase 3 `vocab_trimmed/` (push จาก `artifacts/vocab_trimmed/`)
- `train.jsonl` (calibration) + `val.jsonl` (วัด perplexity) จาก Phase 2

**Output → Kaggle Dataset:** `model_pruned_moderate/` (และ/หรือ `_aggressive/`) + ตาราง perplexity ก่อน recovery

---
## 🧭 ทำทีละขั้น (อ่านก่อนรัน)
1. **เตรียม Kaggle:** Settings → Accelerator = **GPU T4 ×1** (หรือ P100). Add Input → attach dataset ที่มี `vocab_trimmed/`, `train.jsonl`, `val.jsonl`
2. แก้ `DS` ใน cell ถัดไปให้ตรงชื่อ dataset ที่ attach (ดู path ใน `/kaggle/input/`)
3. รันทีละ cell ตามหัวข้อ: โหลด → วัด layer importance → วัด ppl ฐาน → ตัด layer → วัด ppl ใหม่ → (Aggressive) FFN prune → เซฟ
4. เลือกสถานการณ์ด้วย `RUN_WIDTH_PRUNING` (False = Moderate, True = Aggressive). แนะนำรัน **Moderate ก่อน** ดู ppl แล้วค่อยลอง Aggressive
5. Save Version → สร้าง/อัปเดต Kaggle Dataset จาก output ให้ Phase 5 attach ต่อ

> หมายเหตุ: perplexity จะ **เพิ่มขึ้น (แย่ลง)** หลังตัด — ปกติ Phase 5 (QLoRA recovery) จะกู้คืนในโดเมน""")

md("## ขั้น 0 — ติดตั้ง + ตั้งค่า path/knobs")
code(r"""
%pip install -q -U transformers accelerate datasets
import os, json, gc, math
import torch, torch.nn as nn, torch.nn.functional as F

INPUT = "/kaggle/input"
OUT   = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

# ⚙️ แก้ตรงนี้: ชื่อ Kaggle Dataset ที่ attach (โฟลเดอร์ใต้ /kaggle/input/<DS>/)
DS = "pholama-phase3"           # <-- TODO เปลี่ยนเป็นชื่อจริง
MODEL_DIR = f"{INPUT}/{DS}/vocab_trimmed"
CALIB_FP  = f"{INPUT}/{DS}/train.jsonl"     # calibration = train โดเมน
VAL_FP    = f"{INPUT}/{DS}/val.jsonl"       # วัด perplexity

# ── knobs ──
TARGET_LAYERS      = 24        # 28 → 24 (ตัด 4) — จุด knee จาก sweep; 28→20 พังโมเดล
PROTECT_FIRST_LAST = True      # กันตัด layer 0 และ layer สุดท้าย (มักสำคัญ)
N_CALIB            = 256       # จำนวนตัวอย่าง calibration
MAX_LEN            = 1024
RUN_WIDTH_PRUNING  = False     # False=Moderate, True=Aggressive (FFN prune)
NEW_INTERMEDIATE   = 6144      # 8192 → 6144 (−25%) เมื่อ Aggressive
FORCE_CPU          = False     # True = บังคับ CPU (ช้ากว่า GPU ~9 เท่า — เลี่ยงถ้าทำได้)

# โมเดล Phase 3 = 2.96B → bf16 ~5.9GB. ถ้า VRAM < 5.9GB ไดรเวอร์ NVIDIA รุ่นใหม่จะ spill
# ส่วนเกินลง "shared system RAM" (CUDA sysmem fallback) → ยังรันได้ แค่ช้าลงเล็กน้อย
# ทดสอบบน RTX 3050 Laptop 4GB: โหลดได้ (alloc 5.9GB), forward ~2.7s, Moderate รวม ~25–30 นาที
device = "cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda"
print("device:", device, "| torch:", torch.__version__)
if device == "cuda":
    free = torch.cuda.mem_get_info()[0]/1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM free: {free:.1f}GB"
          + (" (จะ spill ลง shared RAM — ช้าลงนิดหน่อยแต่รันได้)" if free < 6 else ""))
assert os.path.isdir(MODEL_DIR), f"ไม่พบ {MODEL_DIR} — attach dataset แล้วแก้ DS ให้ตรง"
""")

md("""## ขั้น 1 — โหลดโมเดล vocab-trimmed + calibration

โหลดโมเดล Phase 3 และเตรียมข้อความ calibration จาก train (จัดรูปแบบ chat ให้ตรงกับตอนใช้งานจริง)""")
code(r"""
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to(device).eval()
model.config.use_cache = False
N_LAYERS = model.config.num_hidden_layers
print(f"layers={N_LAYERS} | hidden={model.config.hidden_size} | inter={model.config.intermediate_size}")
print(f"param เริ่มต้น: {sum(p.numel() for p in model.parameters())/1e9:.4f}B")

def chat_text(ex):
    msgs = [{"role":"user","content":ex["question"]},
            {"role":"assistant","content":ex["answer"]}]
    return tok.apply_chat_template(msgs, tokenize=False)

calib = [json.loads(l) for l in open(CALIB_FP, encoding="utf-8")][:N_CALIB]
calib_texts = [chat_text(e) for e in calib]
val   = [json.loads(l) for l in open(VAL_FP, encoding="utf-8")]
print("calib:", len(calib_texts), "| val:", len(val))
""")

md(r"""## ขั้น 2 — วัดความสำคัญแต่ละ layer (ShortGPT Block Influence)

**แนวคิด ShortGPT:** ถ้า hidden state ก่อน/หลัง layer แทบไม่เปลี่ยน → layer นั้น "ซ้ำซ้อน" ตัดได้
วัดด้วย **angular distance** ระหว่าง `hidden_states[i]` กับ `hidden_states[i+1]` (เฉลี่ยทุก token)
ค่ามาก = เปลี่ยนมาก = สำคัญ → เก็บ • ค่าน้อย = สำคัญน้อย → **ตัด**""")
code(r"""
@torch.no_grad()
def block_influence(texts):
    n = model.config.num_hidden_layers
    score = torch.zeros(n, dtype=torch.float64)
    ntok  = 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.to(device)
        hs = model(ids, output_hidden_states=True).hidden_states  # tuple len n+1
        for i in range(n):
            a, b = hs[i][0].float(), hs[i+1][0].float()           # [T, H]
            cos = F.cosine_similarity(a, b, dim=-1).clamp(-1, 1)   # [T]
            ang = torch.arccos(cos) / math.pi                      # angular distance 0..1
            score[i] += ang.sum().item()
        ntok += ids.shape[1]
    return (score / max(ntok, 1)).tolist()

bi = block_influence(calib_texts)
order = sorted(range(N_LAYERS), key=lambda i: bi[i])   # น้อย→มาก
print("Block Influence ต่อ layer (เรียงจากสำคัญน้อยสุด):")
for i in order:
    print(f"  layer {i:2d}: {bi[i]:.4f}")
""")
code(r"""
# เลือก layer ที่จะตัด = BI ต่ำสุด (กัน 0/สุดท้ายถ้า PROTECT_FIRST_LAST)
protected = {0, N_LAYERS-1} if PROTECT_FIRST_LAST else set()
cand = [i for i in order if i not in protected]
n_drop = N_LAYERS - TARGET_LAYERS
drop = sorted(cand[:n_drop])
keep = [i for i in range(N_LAYERS) if i not in drop]
print(f"ตัด {n_drop} layers: {drop}")
print(f"เก็บ {len(keep)} layers: {keep}")
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(11,3))
    plt.bar(range(N_LAYERS), bi, color=["#d33" if i in drop else "#39c" for i in range(N_LAYERS)])
    plt.title("Block Influence ต่อ layer (แดง=ตัด)"); plt.xlabel("layer"); plt.ylabel("angular dist")
    plt.tight_layout(); plt.show()
except Exception as e:
    print("(ข้าม plot)", e)
""")

md("## ขั้น 3 — วัด perplexity ฐาน (ก่อนตัด) บน val")
code(r"""
@torch.no_grad()
def perplexity(samples, max_n=150):
    nll, ntok = 0.0, 0
    for ex in samples[:max_n]:
        ids = tok(chat_text(ex), return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.to(device)
        if ids.shape[1] < 2: continue
        out = model(ids, labels=ids)
        t = ids.shape[1] - 1
        nll += out.loss.item() * t; ntok += t
    return math.exp(nll / max(ntok, 1))

ppl_base = perplexity(val)
print(f"perplexity ฐาน (vocab-trimmed, ก่อนตัด layer): {ppl_base:.3f}")
""")

md(r"""## ขั้น 4 — ตัด layer (28 → 20)

> ⚠️ **สำคัญ:** หลังลบ layer ต้อง **reassign `layer_idx`** ของแต่ละ layer ที่เหลือให้ต่อเนื่อง
> ไม่งั้น KV cache จะ index ผิด → output พัง""")
code(r"""
model.model.layers = nn.ModuleList([model.model.layers[i] for i in keep])
model.config.num_hidden_layers = len(keep)
# reassign layer_idx (กัน KV cache พัง)
for new_i, layer in enumerate(model.model.layers):
    if hasattr(layer, "self_attn"):
        layer.self_attn.layer_idx = new_i
    if hasattr(layer, "layer_idx"):
        layer.layer_idx = new_i
gc.collect(); torch.cuda.empty_cache() if device=="cuda" else None
p_after_layer = sum(p.numel() for p in model.parameters())/1e9
print(f"เหลือ {model.config.num_hidden_layers} layers | param: {p_after_layer:.4f}B (Moderate 28→24 ~2.55B)")
""")
code(r"""
ppl_layer = perplexity(val)
print(f"perplexity หลังตัด layer: {ppl_layer:.3f}  (ฐาน {ppl_base:.3f}, +{ppl_layer-ppl_base:.2f})")
print("→ ขึ้นเป็นเรื่องปกติ; Phase 5 (QLoRA) จะกู้คืนในโดเมน")
""")

md(r"""## ขั้น 5 — (Aggressive เท่านั้น) FFN width pruning  8192 → 6144

ตัด channel ใน intermediate ที่ "ไหลผ่านน้อยสุด" วัดจาก activation จริงบน calibration
(Wanda-style: importance = ‖activation ที่เข้า down_proj‖ ต่อ channel) แล้วตัดให้ตรงกันทั้ง gate/up (rows) และ down (cols)

ข้าม cell นี้ถ้า `RUN_WIDTH_PRUNING=False`""")
code(r"""
@torch.no_grad()
def ffn_channel_importance(texts):
    n = model.config.num_hidden_layers
    inter = model.config.intermediate_size
    imp = [torch.zeros(inter, dtype=torch.float64, device=device) for _ in range(n)]
    hooks = []
    def mk(i):
        def hook(mod, inp, out):
            x = inp[0].detach()                       # [B,T,inter] = act เข้า down_proj
            imp[i] += x.abs().float().sum(dim=(0,1)).double()
        return hook
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.down_proj.register_forward_hook(mk(i)))
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.to(device)
        model(ids)
    for h in hooks: h.remove()
    return imp

def slice_mlp(layer, keep_idx):
    h = model.config.hidden_size; k = len(keep_idx)
    idx = torch.as_tensor(keep_idx, device=device)
    g, u, d = layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj
    ng = nn.Linear(h, k, bias=False).to(g.weight.device, g.weight.dtype)
    nu = nn.Linear(h, k, bias=False).to(u.weight.device, u.weight.dtype)
    nd = nn.Linear(k, h, bias=False).to(d.weight.device, d.weight.dtype)
    ng.weight.data.copy_(g.weight.data[idx, :])
    nu.weight.data.copy_(u.weight.data[idx, :])
    nd.weight.data.copy_(d.weight.data[:, idx])
    layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj = ng, nu, nd

if RUN_WIDTH_PRUNING:
    imp = ffn_channel_importance(calib_texts)
    for i, layer in enumerate(model.model.layers):
        keep_idx = torch.topk(imp[i], NEW_INTERMEDIATE).indices.sort().values.tolist()
        slice_mlp(layer, keep_idx)
    model.config.intermediate_size = NEW_INTERMEDIATE
    gc.collect(); torch.cuda.empty_cache() if device=="cuda" else None
    p_after_ffn = sum(p.numel() for p in model.parameters())/1e9
    print(f"FFN {8192}→{NEW_INTERMEDIATE} | param: {p_after_ffn:.4f}B (Aggressive ~2.1B)")
    print(f"perplexity หลัง FFN prune: {perplexity(val):.3f}")
else:
    print("ข้าม width pruning (Moderate)")
""")

md("## ขั้น 6 — เซฟ + probe คุณภาพ (ก่อน recovery) + วิธี push")
code(r"""
scenario = "aggressive" if RUN_WIDTH_PRUNING else "moderate"
save_dir = os.path.join(OUT, f"model_pruned_{scenario}")
model.config.use_cache = True
model.save_pretrained(save_dir, safe_serialization=True)
tok.save_pretrained(save_dir)

# บันทึกสรุปตัวเลข
summary = {
    "scenario": scenario, "kept_layers": keep, "dropped_layers": drop,
    "num_hidden_layers": model.config.num_hidden_layers,
    "intermediate_size": model.config.intermediate_size,
    "params_B": round(sum(p.numel() for p in model.parameters())/1e9, 4),
    "ppl_base": round(ppl_base,3), "ppl_after_layer": round(ppl_layer,3),
    "block_influence": {int(i): round(bi[i],4) for i in range(N_LAYERS)},
}
json.dump(summary, open(os.path.join(save_dir,"prune_report.json"),"w"), ensure_ascii=False, indent=2)
print(json.dumps({k:summary[k] for k in ("scenario","params_B","num_hidden_layers","intermediate_size","ppl_base","ppl_after_layer")}, ensure_ascii=False, indent=2))

# probe: ตอบได้ไหม (คาดว่าคุณภาพตก รอ Phase 5)
ids = tok(tok.apply_chat_template([{"role":"user","content":"นิสิตลงทะเบียนเรียนได้ที่ไหน"}],
          add_generation_prompt=True, tokenize=False), return_tensors="pt").input_ids.to(device)
g = model.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
print("\nprobe:", tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True)[:200])
print(f"\n✅ เซฟ → {save_dir}")
""")

md(r"""---
## ✅ เสร็จ Phase 4 — ขั้นต่อไป

1. **Save Version** (Kaggle) → menu → *Create Dataset from Output* (หรืออัปเดต dataset เดิม) ให้มี `model_pruned_moderate/` (และ `_aggressive/` ถ้ารัน)
2. เทียบ `ppl_base` vs `ppl_after_layer` ของ Moderate/Aggressive — เลือกตัวที่จะเข้า Phase 5
3. **Phase 5 (`05_recovery_qlora.ipynb`)**: โหลด pruned model 4-bit → QLoRA บน `train.jsonl` → merge adapter → วัด ppl/quality กลับมา
4. อัปเดต `PROGRESS.md` + `experiments.csv` ด้วยตัวเลขจริง (param, ppl, layers ที่ตัด)

> เป้าหมายรวมหลัง Phase 4: Moderate(28→24) ~2.55B / Aggressive(+FFN) ~2.1B (จากฐาน Phase 3 = 2.9568B)""")

nb = {"cells": cells,
      "metadata": {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                   "language_info":{"name":"python","version":"3.11"}},
      "nbformat":4, "nbformat_minor":5}

out = os.path.join(os.path.dirname(__file__), "..", "notebooks", "04_pruning.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("เขียน", out, "|", len(cells), "cells")
