# -*- coding: utf-8 -*-
"""สร้าง notebooks/05_recovery_qlora.ipynb (พร้อมรัน Kaggle GPU) — mirror scripts/phase5_recovery.py
รัน:  .venv\\Scripts\\python.exe scripts\\_gen_nb05.py
"""
import json, os

cells = []
def md(s):  cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s):cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(keepends=True)})

md(r"""# Phase 5 — Recovery fine-tune (QLoRA)

กู้คุณภาพ **ในโดเมน** ที่เสียไปจาก pruning กลับมา ด้วย QLoRA บน `train.jsonl`
> ความสามารถ **นอกโดเมน** (โค้ด/คณิต) จะไม่ฟื้น = ตั้งใจให้เป็นแบบนั้น (specialization)

**Input (attach เป็น Kaggle Dataset):**
- โมเดล Phase 4 `pruned_moderate/` (หรือ `pruned_aggressive/`)
- `train.jsonl` + `val.jsonl`

**Output → Kaggle Dataset:** `recovered/` (โมเดล merge แล้ว) + `recovery_report.json`

---
## 🧭 ทำทีละขั้น
1. Kaggle: Accelerator = **GPU T4 ×1**. Add Input → attach dataset ที่มี `pruned_moderate/`, `train.jsonl`, `val.jsonl`
2. แก้ `DS` ให้ตรงชื่อ dataset
3. รันทุก cell: โหลด 4-bit + LoRA → เตรียม data (mask prompt) → ppl ก่อน → เทรน → ppl หลัง → merge + เซฟ
4. เทียบ ppl ก่อน/หลัง (ควรลงใกล้ฐานก่อน prune) → Save Version → Dataset ให้ Phase 6

> **หมายเหตุ:** ใช้ training loop เขียนเอง (ไม่พึ่ง `transformers.Trainer`/`datasets`) — พิสูจน์แล้วว่ารันได้ทั้ง Kaggle และ Windows local (เลี่ยง pyarrow segfault). บน Kaggle T4 16GB เพิ่ม `BATCH` ได้ (เช่น 4–8)""")

md("## ขั้น 0 — ติดตั้ง + knobs")
code(r"""
%pip install -q -U transformers peft bitsandbytes accelerate
import os, json, math, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

INPUT = "/kaggle/input"; OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
DS = "pholama-phase4"                         # <-- TODO ชื่อ dataset ที่ attach
MODEL_DIR = f"{INPUT}/{DS}/pruned_moderate"   # หรือ pruned_aggressive
TRAIN_FP  = f"{INPUT}/{DS}/train.jsonl"
VAL_FP    = f"{INPUT}/{DS}/val.jsonl"

EPOCHS, SEQ_LEN, BATCH, GRAD_ACCUM = 2, 512, 1, 16   # T4 16GB: BATCH 4–8 ได้
LR, RANK, ALPHA, PPL_N = 2e-4, 16, 32, 120
device = "cuda" if torch.cuda.is_available() else "cpu"
assert os.path.isdir(MODEL_DIR), f"ไม่พบ {MODEL_DIR} — attach dataset + แก้ DS"
print("device:", device)
""")

md("## ขั้น 1 — โหลด pruned model แบบ 4-bit (nf4) + LoRA")
code(r"""
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, quantization_config=bnb,
                                             dtype=torch.bfloat16, device_map="cuda")
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(
    r=RANK, lora_alpha=ALPHA, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
model.print_trainable_parameters()
print("VRAM after load:", round(torch.cuda.memory_allocated()/1e9,2), "GB")
""")

md(r"""## ขั้น 2 — เตรียม data (chat template + **mask prompt**)
เทรนเฉพาะส่วนคำตอบ (labels = -100 ที่ prompt) เพื่อให้โมเดลเรียน "ตอบ" ไม่ใช่ท่องคำถาม""")
code(r"""
def chat_full(ex):
    return tok.apply_chat_template(
        [{"role":"user","content":ex["question"]},
         {"role":"assistant","content":ex["answer"]}], tokenize=False)

def build(rows):
    out=[]
    for ex in rows:
        prompt = tok.apply_chat_template([{"role":"user","content":ex["question"]}],
                                         add_generation_prompt=True, tokenize=False)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        f_ids = tok(chat_full(ex), add_special_tokens=False).input_ids[:SEQ_LEN]
        labels=list(f_ids)
        for i in range(min(len(p_ids),len(labels))): labels[i]=-100
        out.append({"input_ids":f_ids,"labels":labels,"attention_mask":[1]*len(f_ids)})
    return out

def collate(batch):
    m=max(len(b["input_ids"]) for b in batch); pad=tok.pad_token_id
    ii,ll,am=[],[],[]
    for b in batch:
        n=m-len(b["input_ids"])
        ii.append(b["input_ids"]+[pad]*n); ll.append(b["labels"]+[-100]*n); am.append(b["attention_mask"]+[0]*n)
    return {"input_ids":torch.tensor(ii),"labels":torch.tensor(ll),"attention_mask":torch.tensor(am)}

train_rows=[json.loads(l) for l in open(TRAIN_FP,encoding="utf-8")]
val_rows=[json.loads(l) for l in open(VAL_FP,encoding="utf-8")]
train_ds=build(train_rows); print("train:",len(train_ds),"| val:",len(val_rows))
""")

md("## ขั้น 3 — perplexity ก่อนเทรน")
code(r"""
@torch.no_grad()
def ppl(n=PPL_N):
    model.eval(); nll=0.0; ntok=0
    for ex in val_rows[:n]:
        ids=tok(chat_full(ex),return_tensors="pt",truncation=True,max_length=SEQ_LEN,
                add_special_tokens=False).input_ids.to(device)
        if ids.shape[1]<2: continue
        o=model(ids,labels=ids); t=ids.shape[1]-1; nll+=o.loss.item()*t; ntok+=t
    return math.exp(nll/max(ntok,1))

ppl_before=ppl(); print("ppl_before =", round(ppl_before,3))
""")

md("## ขั้น 4 — เทรน QLoRA (manual loop)")
code(r"""
per_epoch=math.ceil(len(train_ds)/(BATCH*GRAD_ACCUM)); total_steps=int(per_epoch*EPOCHS)
trainable=[p for p in model.parameters() if p.requires_grad]
opt=torch.optim.AdamW(trainable,lr=LR)
sched=get_cosine_schedule_with_warmup(opt,int(0.03*total_steps),total_steps)
print(f"total_steps={total_steps} (per_epoch≈{per_epoch})")

model.train(); model.config.use_cache=False
step=micro=0; run=0.0; opt.zero_grad(); rng=random.Random(42); done=False; ep=0
while not done:
    ep+=1; order=list(range(len(train_ds))); rng.shuffle(order)
    for s in range(0,len(order),BATCH):
        batch=collate([train_ds[i] for i in order[s:s+BATCH]])
        batch={k:v.to(device) for k,v in batch.items()}
        loss=model(**batch).loss/GRAD_ACCUM; loss.backward(); run+=loss.item()*GRAD_ACCUM; micro+=1
        if micro%GRAD_ACCUM==0:
            torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step(); sched.step(); opt.zero_grad(); step+=1
            if step%10==0 or step==total_steps:
                print(f"  step {step}/{total_steps} | loss {run/(10*GRAD_ACCUM):.4f} | lr {sched.get_last_lr()[0]:.2e} | VRAM {torch.cuda.max_memory_allocated()/1e9:.2f}GB"); run=0.0
            if step>=total_steps: done=True; break
""")

md(r"""## ขั้น 5 — perplexity หลัง + merge + เซฟ + probe
> ⚠️ **สำคัญ:** ห้าม `merge_and_unload()` บนโมเดล 4-bit ตรงๆ — re-quantize ทำคุณภาพตกมาก (ppl 9→17)
> วิธีถูก: เซฟ **adapter** → โหลด base **bf16** (เต็มความละเอียด) → apply + merge""")
code(r"""
ppl_after = ppl(); print(f"ppl_after (4bit+adapter) = {ppl_after:.3f}  (ก่อน {ppl_before:.3f})")

import gc
from peft import PeftModel
adapter_dir = os.path.join(OUT, "adapter")
model.save_pretrained(adapter_dir); tok.save_pretrained(adapter_dir)   # LoRA adapter เท่านั้น
del model; gc.collect(); torch.cuda.empty_cache()

base = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to(device)
merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload(); merged.config.use_cache = True
""")
code(r"""
# วัด ppl ของ merged bf16 (ของจริงที่จะ deploy)
@torch.no_grad()
def ppl_of(m, n=PPL_N):
    m.eval(); nll=ntok=0
    for ex in val_rows[:n]:
        ids=tok(chat_full(ex),return_tensors="pt",truncation=True,max_length=SEQ_LEN,add_special_tokens=False).input_ids.to(device)
        if ids.shape[1]<2: continue
        o=m(ids,labels=ids); t=ids.shape[1]-1; nll+=o.loss.item()*t; ntok+=t
    return math.exp(nll/max(ntok,1))
ppl_merged = ppl_of(merged); print(f"ppl_merged (bf16, ของจริง) = {ppl_merged:.3f}")

save_dir = os.path.join(OUT, "recovered")
merged.save_pretrained(save_dir, safe_serialization=True); tok.save_pretrained(save_dir)
json.dump({"base":os.path.basename(MODEL_DIR),"ppl_before":round(ppl_before,3),
           "ppl_after_4bit_adapter":round(ppl_after,3),"ppl_merged_bf16":round(ppl_merged,3),
           "rank":RANK,"alpha":ALPHA,"epochs":EPOCHS,"seq_len":SEQ_LEN},
          open(os.path.join(save_dir,"recovery_report.json"),"w"), ensure_ascii=False, indent=2)

ids=tok(tok.apply_chat_template([{"role":"user","content":"นิสิตลงทะเบียนเรียนได้ที่ไหน"}],
        add_generation_prompt=True, tokenize=False), return_tensors="pt").input_ids.to(device)
g=merged.generate(ids,max_new_tokens=60,do_sample=False,pad_token_id=tok.eos_token_id)
print("probe:", tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True)[:250])
print("✅ เซฟ →", save_dir)
""")

md(r"""---
## ✅ เสร็จ Phase 5 — ขั้นต่อไป
1. ดู `ppl_after` ควร **ลงใกล้** ppl ฐานก่อน prune (Phase 1) → recovery สำเร็จ
2. Save Version → Dataset `recovered/` ให้ Phase 6
3. **Phase 6 (`06_quantize.ipynb`)**: แปลง `recovered/` เป็น GGUF (Q4_K_M/Q5_K_M/Q8) ผ่าน llama.cpp
4. อัปเดต `PROGRESS.md` + `experiments.csv` (ppl ก่อน/หลัง recovery)""")

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(__file__),"..","notebooks","05_recovery_qlora.ipynb")
json.dump(nb, open(out,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("เขียน", out, "|", len(cells), "cells")
