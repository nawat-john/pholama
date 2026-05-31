# PROGRESS — ติดตามความคืบหน้าโปรเจกต์ย่อ/ปรับแต่ง LLM

> เอกสารนี้แตก [`llm_pruning_project_plan.md`](./llm_pruning_project_plan.md) ออกเป็น subtask ย่อยที่ติ๊กได้
> ติ๊ก `[x]` เมื่อทำเสร็จ • อัปเดตคอลัมน์ Deliverable/หมายเหตุเมื่อมีผลลัพธ์จริง
> การรันจริงทำบน **Kaggle/Colab notebook** — ดูตาราง map notebook ด้านล่าง

---
## ▶️ ทำต่อ SESSION หน้า (อ่านตรงนี้ก่อน)

**ค้างที่:** Phase 7 — เหลือ **B2 baseline** (เดิม+QLoRA+quantize ไม่ prune) + **lm-eval ทางการ** (gsm8k/humaneval) — RAG + eval B1↔B3 + REPORT เสร็จแล้ว

**สถานะ ณ จบ session (2026-05-31):**
- ✅ **Phase 2** — 4,016 clean → 3212/401/403, leak check ผ่าน
- ✅ **Phase 3** — vocab trim **128,256 → 44,926** • **3.2127B → 2.9568B**
- ✅ **Phase 4** — Moderate 28→24 = **2.5541B** • Aggressive +FFN = **2.1011B**
- ✅ **Phase 5** — QLoRA recovery: **ppl 14.14 → 7.59** (bf16) → `artifacts/pruned_moderate_recovered/`
- ✅ **Phase 6** — GGUF: **Q4_K_M 1.53GB @ 23.5 tok/s CPU** → `artifacts/gguf/`
- [~] **Phase 7** — RAG (e5-small + faiss + llama-server) ✅ • in-domain eval **B1 35.0% vs B3 47.5%** (n40, RAG, Gemini judge) ✅ • OOD regression ยืนยัน ✅ • **`REPORT.md`** ครบทุก % ✅ — เหลือ B2 + lm-eval
- artifacts อยู่ใน `artifacts/` ไม่ commit (.gitignore)

> 🔑 **findings:** P4 ShortGPT 28→20 พัง→28→24 • P5 merge LoRA บน 4-bit เสียคุณภาพ→merge บน bf16 base • P6 trimmed tokenizer → patch convert เป็น llama-bpe • Windows: Trainer/datasets segfault (pyarrow), pip gguf reqs ทับ torch เป็น CPU

**Phase 7 ที่ทำเสร็จแล้ว (local):**
- ✅ **RAG** — `scripts/phase7_rag.py`: e5-small embed 361 chunks + faiss + llama-server (Q4_K_M) → `artifacts/rag/`
- ✅ **in-domain eval** — `scripts/phase7_eval.py` (B3) + `phase7_eval_b1.py` (B1), judge ด้วย Gemini → `artifacts/eval/` • **B1 35.0% vs B3 47.5%** (RAG, n40)
- ✅ **OOD regression probe** + **`REPORT.md`** (เทียบเดิม↔ใหม่ ครบทุก %)

**Phase 7 ที่เหลือ:**
1. ⚠️ **B2 baseline (บังคับ)** — original Typhoon2 + QLoRA + quantize (**ไม่ prune**) มาเทียบ:
   ```powershell
   .venv\Scripts\python.exe scripts\phase5_recovery.py --model-dir scb10x/llama3.2-typhoon2-3b-instruct --epochs 2 --out artifacts\b2_recovered
   # แล้ว convert+quantize เป็น GGUF + รัน phase7_eval แบบชี้ B2 → เทียบ B2 vs B3 (prune คุ้มไหม)
   ```
2. **lm-eval ทางการ** — gsm8k/humaneval × B1/B2/B3 (รัน Kaggle GPU; local ช้าเกิน) เก็บตัวเลข OOD เป็นทางการ
3. (ทางเลือก) Ollama deploy + README

**สคริปต์ที่ reproduce ได้ (local):**
```powershell
. .tools\activate.ps1; $env:PYTHONUTF8="1"
.venv\Scripts\python.exe scripts\phase3_vocab_trim.py --aggressive --target 40000 --with-model
.venv\Scripts\python.exe scripts\phase4_prune.py                                    # Moderate 28→24
.venv\Scripts\python.exe scripts\phase5_recovery.py --model-dir artifacts\pruned_moderate --epochs 2
# Phase 6 (local): convert + quantize
.venv\Scripts\python.exe .cache\llama.cpp\convert_hf_to_gguf.py artifacts\pruned_moderate_recovered --outfile artifacts\gguf\recovered-f16.gguf --outtype f16
.cache\llama-bin\llama-quantize.exe artifacts\gguf\recovered-f16.gguf artifacts\gguf\recovered-Q4_K_M.gguf Q4_K_M
```
> หมายเหตุ Phase 6: ต้อง patch `BPE pre-tokenizer was not recognized` → `res="llama-bpe"` ใน `.cache/llama.cpp/conversion/base.py` ก่อน convert • binary จาก llama.cpp release `b9442-bin-win-cpu-x64`

**ตัดสินใจค้างไว้:**
- เนื้อหา reg.chula เป็น snapshot วันที่ crawl (ปีการศึกษา 2568/2569) — Phase 7 ใช้ RAG ดึงของจริงแทน weight
- การคัด dataset เป็น "อัตโนมัติ" (heuristic) — ถ้าต้องการคุณภาพสูงสุดให้คัดด้วยมือก่อนเข้า Phase 5
- **vocab 40k ตัดอังกฤษ/โค้ดที่ใช้น้อยทิ้งด้วย** (สอดคล้อง specialization) — ถ้าอยากรักษาอังกฤษเต็มให้รัน conservative (ไม่ใส่ `--aggressive` → 104k, ตัด 18.8%)

**Gotchas เครื่องนี้ (สำคัญ):** ต้อง `truststore` (proxy MITM ทำ SSL fail) + `PYTHONUTF8=1` (กัน cp874 crash). ทั้งคู่จัดการใน `scripts/_bootstrap.py` แล้ว • gen ใช้ **multi-model fallback** สลับ model อัตโนมัติเมื่อ 429 (โควต้าฟรีรายวันแยกราย model-name, reset รายวัน)
---

## สถานะรวม (อัปเดตเมื่อ progress เปลี่ยน)

- [x] Phase 0 — ยืนยันสเปกโมเดลฐาน ✅ (3.213B ตรงแผน)
- [x] Phase 1 — Setup + baseline ✅ (B1 local 4-bit: 10.61 tok/s, VRAM 2.25GB) + experiment log (`experiments.csv`)
- [x] Phase 2 — สร้างชุดข้อมูลโดเมน ✅ (reg.chula.ac.th • 4,016 clean → 3212/401/403 • leak check ผ่าน)
- [x] Phase 3 — Vocabulary trimming ✅ (aggressive: 128,256→44,926 ตัด 65.1% • 3.2127B→2.9568B)
- [x] Phase 4 — Structured pruning ✅ — Moderate (28→24, **2.5541B**, ppl 13.2) + Aggressive (+FFN 6144, **2.1011B**, ppl 18.6)
- [x] Phase 5 — Recovery fine-tune (QLoRA) ✅ — Moderate: **ppl 14.14 → 7.59** (bf16) • adapter rank16 • local RTX3050
- [x] Phase 6 — Quantization ✅ — GGUF **Q4_K_M 1.53GB @ 23.5 tok/s CPU** (+ Q5_K_M/Q8_0/f16) • local llama.cpp
- [~] Phase 7 — Deploy + eval + report — **RAG + in-domain eval (B1 vs B3) + REPORT.md เสร็จ**; เหลือ B2 + lm-eval ทางการ

## Map: phase → Kaggle notebook

| Phase | Notebook | รันที่ | Compute |
|---|---|---|---|
| 0–1 | `notebooks/01_baseline.ipynb` | Kaggle/Colab | GPU T4/P100 16GB |
| 2 | `notebooks/02_dataset.ipynb` (+ คัดด้วยมือ) | Kaggle + เครื่องตัวเอง | CPU |
| 3 | `notebooks/03_vocab_trim.ipynb` | Kaggle/Colab หรือ CPU | CPU พอ |
| 4 | `notebooks/04_pruning.ipynb` | Kaggle/Colab | GPU |
| 5 | `notebooks/05_recovery_qlora.ipynb` | Kaggle/Colab | GPU 16GB |
| 6 | `notebooks/06_quantize.ipynb` | Kaggle/CPU | CPU |
| 7 | `notebooks/07_eval_deploy.ipynb` | Kaggle + เครื่อง deploy | GPU/CPU |

> หมายเหตุการทำงานข้าม Kaggle: artifact (โมเดล/dataset/tokenizer) แต่ละ phase ควร push เป็น **Kaggle Dataset** เพื่อให้ notebook phase ถัดไป attach เป็น input ได้ (notebook แต่ละตัวรันแยก session ไม่มี state ร่วมกัน)

---

## Phase 0 — ยืนยันสเปกโมเดลฐาน (ก่อนเริ่มทุกอย่าง)

- [ ] โหลด `config.json` ของ Typhoon 2 3B จริง
- [ ] ยืนยันค่า: hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim, intermediate_size, vocab_size, tie_word_embeddings
- [x] เทียบกับตัวเลขสมมติในแผน (Llama 3.2 3B) — **ตรงทุกค่า** ไม่ต้องแก้แผน
- [x] นับพารามิเตอร์จริง = **3.213B** (embedding 12.3% / transformer 87.7%) ⚠️ ใช้คำนวณจาก config ไม่ใช่ `numel()` เพราะ 4-bit pack ทำให้ numel นับครึ่งเดียว

**Deliverable:** ✅ `baseline_B1.json` (ราก project) + config ยืนยันตรงแผน

---

## Phase 1 — Setup + baseline (≈1 สัปดาห์) → `01_baseline.ipynb`

**Setup**
- [x] ติดตั้ง: PyTorch (cu124), transformers, peft, trl, bitsandbytes, datasets — env บน D (lm-eval/llama.cpp ค่อยลงตอน Phase 6/7)
- [x] โหลด Typhoon 2 3B (4-bit) + ยืนยัน inference ได้ (ตอบไทย/อังกฤษ/โค้ด/คณิต ปกติ)
- [x] ตั้งที่เก็บ experiment log — **`experiments.csv`** (ราก project) มี row B1 แล้ว เก็บเทียบทุก phase
- [x] โครงสร้าง notebooks/ ครบ 7 ไฟล์

**Baseline eval (3 แกน — เก็บไว้เทียบทุก phase)**
- [ ] วัดคุณภาพ **ในโดเมน** (รอ test set Phase 2)
- [~] นอกโดเมน — probe เชิงคุณภาพผ่าน (Fibonacci/คณิต ถูก); ตัวเลขทางการ (gsm8k/humaneval) **เลื่อนไป Phase 7 บน Kaggle GPU** — lm-eval บน RTX 3050 ช้าเกิน (หลายชม.) และ Phase 7 ต้องรัน B1/B2/B3 พร้อมกันอยู่แล้ว จึงรอบเดียวจบ
- [x] latency: **7.26 tok/s** (4-bit, RTX 3050)
- [x] VRAM ตอน inference: **2.25 GB** (4-bit, footprint 2.32 GB)
- [x] ขนาด weight: BF16 ทฤษฎี 6.43 GB / 4-bit จริง ~1.77 GB

**Deliverable:** ✅ `baseline_B1.json` (B1 local 4-bit) — repo reproduce ได้
**หมายเหตุ:** ตัวเลข B1 นี้เป็น **4-bit บน RTX 3050** ถ้าต้องการ B1 แบบ BF16 ตรงสเปกแผน ให้รัน `LOAD_4BIT=False` บน Kaggle 16GB

---

## Phase 2 — สร้างชุดข้อมูลโดเมน (≈3–4 สัปดาห์ — หนักสุด) → `02_dataset.ipynb` + `scripts/phase2_*.py`

> ความเสี่ยงสูงสุดของทั้งโปรเจกต์: ดาตาน้อย/คุณภาพต่ำ = พังทั้งงาน เผื่อเวลามากที่สุด
>
> **โดเมนจริง (ยืนยันแล้ว):** สำนักงานการทะเบียน จุฬาฯ — ตอบทุกอย่างใน https://www.reg.chula.ac.th/th/
> (ไม่ใช่ "กิจการนักศึกษา" ตามร่างแผนเดิม • ใช้คำว่า **"นิสิต"**) — ยังไม่มีเอกสารจริง จึง **crawl เว็บ** เป็นแหล่งข้อมูล
>
> **รันแบบ local** (ไม่ใช่ Kaggle): Phase 2 เป็น CPU ล้วน • สคริปต์อยู่ใน `scripts/` • LLM gen = **Gemini gemini-2.5-flash** (key ใน `.env`)
> หมายเหตุ env: ต้องใช้ `truststore` (proxy MITM) + `PYTHONUTF8=1` (กัน cp874 crash)

- [x] รวบรวมแหล่งจริง: `scripts/phase2_scrape.py` crawl reg.chula.ac.th/th/ (52 หน้า HTML + ~30 PDF) → **361 chunks** → `data/chunks.jsonl` + `data/corpus_raw.txt`
- [x] ออกแบบ prompt สร้าง Q&A สังเคราะห์ (grounded ในเว็บ, standalone, คละ ทางการ/ไม่ทางการ × ไทย/อังกฤษ/ปน)
- [x] generate คู่ถาม-ตอบ — `scripts/phase2_generate.py` (JSON mode, resumable, **multi-model fallback**) — **4,198 คู่ดิบ** ใน `data/qa_raw.jsonl` (358/361 chunks)
- [x] **คัดกรอง** — heuristic filter ใน `phase2_split.py` (self-ref/no-info/length + dedup): 4198 → 4144 → 4016 clean; คัดมือเพิ่มได้โดยแก้ `data/qa_raw.jsonl` (ทางเลือก)
- [x] แบ่ง train/val/test (80/10/10) — `scripts/phase2_split.py` → **3212 / 401 / 403**
- [x] ✅ ตรวจ **ไม่มี test รั่ว** ไปอยู่ใน calibration/training (assert ใน split script — ผ่าน)
- [x] เก็บ corpus ดิบแยกไว้สำหรับ vocab analysis (Phase 3) → `data/corpus_raw.txt`
- [x] เขียนเอกสารอธิบายวิธีสร้าง dataset → `data/dataset_card.md` (gen อัตโนมัติใน split script)
- [ ] push เป็น Kaggle Dataset (dataset v1 + corpus) — ทำตอนจะรัน Phase 3+ บน Kaggle

**Deliverable:** ✅ `data/{train,val,test}.jsonl` + `data/corpus_raw.txt` + `data/dataset_card.md`

---

## Phase 3 — Vocabulary trimming (≈1 สัปดาห์) → `03_vocab_trim.ipynb`

> อ้างอิง: Ushio et al. EMNLP 2023 (arXiv 2305.15020) — เหลือ vocab ~50% ยังรักษาคุณภาพได้

> **รันจริงแบบ local (CPU)** ด้วย `scripts/phase3_vocab_trim.py` — ไม่ต้อง Kaggle (embedding surgery ไม่ต้อง forward)

- [x] รัน tokenizer บน corpus + train + val (ไม่แตะ test) นับ token ที่ปรากฏ → **5,388 distinct**
- [x] กำหนดเซตที่เก็บ — **โหมด aggressive (ผู้ใช้เลือก)**: byte alphabet 256 + special 256 + ไทยทั้งหมด 2,462 + latin1 2,874 + punct 512 + corpus + อังกฤษ id ต่ำสุด (ใช้บ่อย) เติมถึง 40k + closure ของ merges
- [x] ลบแถว dead tokens จาก `embed_tokens`/`lm_head` (tied → ชุดเดียว) — ตัด 83,330 base tokens
- [x] สร้าง mapping `old_id → new_id` + rebuild tokenizer.json (vocab+merges+post_processor) → `vocab_map.json`
- [x] sanity test: roundtrip encode/decode ไทย/อังกฤษ/โค้ด/ภาษาที่ตัด ผ่านทั้งหมด + **generate ไทย/อังกฤษ coherent** (`phase3_sanity.py`)
- [x] นับ param หลังตัด: **3.2127B → 2.9568B (−256M)** ✅
- [ ] push โมเดล vocab-trimmed + tokenizer เป็น Kaggle Dataset (ทำก่อนเริ่ม Phase 4 บน Kaggle)

**Deliverable:** ✅ `artifacts/vocab_trimmed/` (model+tokenizer) + `vocab_trim_report.json`
**ผลจริง:** vocab 128,256 → **44,926** (ตัด 65.1%) • merges 280,147 → 103,362 • รวม **2.9568B**

> ⚠️ **ผลจริงต่างจากสมมติฐานแผน:** tokenizer Llama3.2 เป็น **English-dominant** — ASCII/อังกฤษ 97,718, ไทยแค่ 2,462, ภาษาที่ตายจริง (CJK/เกาหลี/อาหรับ/ซีริลลิก) แค่ 24,044. ตัด "ภาษาตายอย่างเดียว" ได้แค่ −18.8% (→104k) ไม่ถึงเป้าแผน ~40k. จะถึง 40k ต้องตัดอังกฤษ/โค้ดที่ใช้น้อยด้วย (เลือกโหมด aggressive) — สอดคล้องธีม specialization; ข้อความใดๆ ยัง encode ได้ผ่าน byte fallback (แค่ใช้ token มากขึ้น)

---

## Phase 4 — Structured pruning (≈2–3 สัปดาห์ — แกนโปรเจกต์) → `04_pruning.ipynb`

> ใช้ **dataset กิจการนักศึกษาเป็น calibration data** — กลไกที่ทำให้โมเดลเอนเอียงเข้าโดเมน

> **รันจริง local GPU (RTX 3050 ผ่าน sysmem fallback, ~2.7s/forward):** `scripts/phase4_prune.py` + `phase4_sweep.py`

**4a. Layer pruning (ShortGPT Block-Influence)**
- [x] วัดความสำคัญแต่ละ layer ด้วย calib (angular distance ระหว่าง hidden in/out) — BI: layer 0 (0.384) สำคัญสุด, 20–25 น้อยสุด
- [x] **sweep หาจำนวนที่ตัดได้** (แทน 28→20 ตายตัว): 2→ppl7.2 / 4→ppl**13.2** / 6→131 / 8→237 → เลือก **ตัด 4 (28→24)**
- [x] ประเมิน perplexity บน val ก่อน recovery: base 5.77 → **13.2** (coherent)

**4b. Width pruning (Aggressive)** ✅
- [x] ลด FFN intermediate 8,192 → 6,144 (−25%, activation importance Wanda-style) บนโมเดล 28→24
- [x] ประเมิน ppl: 13.5 (หลัง layer) → **18.6** (หลัง FFN) → param **2.1011B** • probe coherent บางส่วน (เริ่มวนท้ายประโยค)

**กลยุทธ์การทดลอง**
- [x] รัน **Moderate** (layer-only 28→24) → **2.5541B** ppl 13.2 ✅
- [x] รัน **Aggressive** (layer + FFN) → **2.1011B** ppl 18.6 ✅
- [x] เทียบสถานการณ์ (ตาราง sweep + Moderate/Aggressive)
- [ ] push โมเดล pruned เป็น Kaggle Dataset (ตอนรัน Phase 5 บน Kaggle)

**Deliverable:** ✅ `artifacts/pruned_moderate/` (2.5541B) + `artifacts/pruned_aggressive/` (2.1011B) + ตารางเทียบก่อน recovery

| สถานการณ์ | param | ppl (ก่อน recovery) | generate |
|---|---|---|---|
| ฐาน (vocab-trimmed) | 2.9568B | 5.77 | coherent |
| **Moderate** (28→24) | **2.5541B** | **13.2** | coherent |
| **Aggressive** (+FFN 6144) | **2.1011B** | **18.6** | coherent บางส่วน (วน) |

> ⚠️ **แผนเดิม 28→20 (~2.14B)/Aggressive 1.76B ใช้ไม่ได้จริง** — ตัด 8 layer ติดกันทำ ppl 237 generate ขยะ. ตัวเลขจริง: Moderate 2.55B / Aggressive 2.10B (ทั้งคู่ recover ไหว)

---

## Phase 5 — Recovery fine-tune QLoRA (≈1–2 สัปดาห์ — ห้ามข้าม) → `05_recovery_qlora.ipynb`

> เป้า: กู้คุณภาพ **ในโดเมน** กลับมา • ความสามารถนอกโดเมนจะไม่ฟื้น = ตั้งใจให้เป็นแบบนั้น

> **รันจริง local RTX 3050 (VRAM peak 2.4GB!)** ด้วย `scripts/phase5_recovery.py` — training loop เขียนเอง (เลี่ยง pyarrow segfault)

- [x] โหลด pruned 4-bit (nf4) + QLoRA rank 16/alpha 32, target = q/k/v/o/gate/up/down
- [x] seq 512, batch 1 + grad accum 16 (effective 16), 2 epoch (402 steps), cosine LR 2e-4
- [x] train + prompt masking (เทรนเฉพาะคำตอบ) — loss 2.5 → 0.6
- [x] **merge บน bf16 base** (ไม่ใช่ 4-bit!) — เซฟ adapter → โหลด base bf16 → merge
- [x] eval ก่อน/หลัง: **ppl 14.14 → 7.59** (bf16, −46%) • generate สะอาดขึ้น อ้างโดเมนถูก
- [ ] push โมเดล recovered เป็น Kaggle Dataset (ตอนรัน Phase 6/7 บน Kaggle)

**Deliverable:** ✅ `artifacts/pruned_moderate_recovered/` (bf16 2.55B) + `_adapter/` (85MB) + report
> ⚠️ **gotcha:** `merge_and_unload()` บนโมเดล 4-bit ทำ ppl 9→17 + เซฟเป็น 4-bit แปลง GGUF ไม่ได้ → ต้อง merge adapter บน **bf16 base** เสมอ
**RAM เทรนจริง:** peak **2.4GB** (ต่ำกว่าแผนคาด 6–12GB มาก เพราะ rank เล็ก + grad checkpoint + adamw ธรรมดา)

---

## Phase 6 — Quantization ✅ → `06_quantize.ipynb` + local llama.cpp

> **รันจริง local** (~5 นาที): convert ด้วย `convert_hf_to_gguf.py` (pip gguf) + quantize ด้วย binary `llama-quantize.exe` (llama.cpp release b9442 win-cpu-x64)

- [x] แปลงเป็น GGUF ผ่าน llama.cpp — `recovered-f16.gguf` (5.02GB)
- [x] สร้างหลายระดับ: Q8_0 (2.67GB) / Q5_K_M (1.79GB) / **Q4_K_M (1.53GB)** + f16 อ้างอิง
- [x] ⚠️ **patch trimmed tokenizer** (validated): convert error `BPE pre-tokenizer was not recognized` → `res="llama-bpe"` ใน `conversion/base.py`
- [x] เทียบคุณภาพ: Q8/Q4 probe ตอบถูกโดเมน ("...ทะเบียนคณะที่นิสิตสังกัด")
- [x] ยืนยันขนาด/RAM: **Q4_K_M 1.53GB + overhead ≈ 1.9GB ตรงเป้าแผน (1.5–2GB)** ✅ • speed CPU: Q4 **23.5 tok/s** / Q8 14.9 tok/s
- [ ] push ไฟล์ GGUF เป็น Kaggle Dataset (ตอน Phase 7)

**Deliverable:** ✅ `artifacts/gguf/` (f16/Q8_0/Q5_K_M/Q4_K_M) + `quant_report.json`

---

## Phase 7 — Deploy + eval + report → `07_eval_deploy.ipynb` + `scripts/phase7_*.py`

**Deploy (RAG)** — `scripts/phase7_rag.py`
- [x] vector DB **FAISS** + embedding **multilingual-e5-small** (เลี่ยง sentence_transformers ที่ segfault)
- [x] index 361 chunks → `artifacts/rag/` • retrieval แม่น (score 0.89)
- [x] serve ด้วย **llama-server** (Q4_K_M) + RAG context (OpenAI-compatible API)
- [x] demo ถาม-ตอบจริง — "ใช้คำร้อง จท. ... ยื่นที่เคาน์เตอร์สำนักงานการทะเบียน"

**Eval 3 แกน** — `scripts/phase7_eval.py` + `phase7_eval_b1.py`
- [x] ในโดเมน (n40, RAG, Gemini judge): **B1 35.0% → B3 47.5%** (+35.7% rel) — *ดีขึ้น*
- [x] ถดถอยนอกโดเมน: probe ยืนยันคณิต/โค้ดตก (17×23=411 ผิด) = specialization สำเร็จ
- [x] ประสิทธิภาพ: ครบ (size −20.5% / RAM −15.6% / speed +122%)
- [ ] `lm-eval-harness` ทางการ (gsm8k/humaneval) — เลื่อนไป Kaggle GPU

**3 Baselines**
- [x] B1: โมเดลเดิม ✅ (วัด in-domain + efficiency)
- [ ] B2: โมเดลเดิม + QLoRA + quantize (ไม่ prune) — **ยังไม่ทำ**
- [x] B3: pipeline เต็ม ✅
- [~] ตอบคำถามรายงาน: B1↔B3 ตอบแล้ว (REPORT.md); B2↔B3 รอ B2

**Deliverable:** ✅ RAG demo + **`REPORT.md`** (เทียบ B1↔B3 ครบ 3 แกน, ทุก %) + `artifacts/{rag,eval}/` — เหลือ B2 + lm-eval ทางการ

---

## Milestone / ตัวเลขเป้าหมายสรุป

- [x] 3.21B → **2.9568B** (หลัง Phase 3 — ตัด vocab 65.1%)
- [x] → **2.5541B Moderate** (28→24, ppl 13.2) / **2.1011B Aggressive** (+FFN6144, ppl 18.6) หลัง Phase 4 (แผนเดิม 2.14B/1.76B ไม่จริง — 28→20 พังโมเดล)
- [x] Phase 5 recovery: Moderate **ppl 14.14 → 7.59** (กู้คุณภาพในโดเมน, 2.55B bf16)
- [x] Moderate + Q4_K_M → **1.53GB weights, RAM รวม ~1.9GB** (หลัง Phase 6) — รันบน CPU **23.5 tok/s** บนแล็ปท็อปไม่มี GPU ✅
