# LLM Pruning Project — แชตบอตสำนักงานการทะเบียน จุฬาฯ

ย่อ + ปรับแต่ง **Typhoon 2 3B** (Llama 3.2 base) ให้เป็นโมเดลเฉพาะทางสำหรับถาม-ตอบ
**สำนักงานการทะเบียน จุฬาลงกรณ์มหาวิทยาลัย** (reg.chula.ac.th) — เล็กลง เร็วขึ้น รันบน CPU ได้

## ผลลัพธ์ (เดิม → ใหม่)

| | เดิม (Typhoon2 3B) | ใหม่ (pipeline เต็ม) | เปลี่ยน |
|---|---:|---:|---:|
| พารามิเตอร์ | 3.21B | 2.55B | **−20.5%** |
| ไฟล์ deploy | 6.43 GB (bf16) | **1.53 GB** (Q4_K_M) | **−76.2%** |
| RAM | 2.25 GB (GPU) | ~1.9 GB (CPU) | **−15.6%** |
| ความเร็ว | 10.6 tok/s (GPU) | **23.5 tok/s (CPU)** | **+122%** |
| คุณภาพในโดเมน | 35.0% | **47.5%** | **+35.7%** |
| โค้ด/คณิต (นอกโดเมน) | ปกติ | ตก | *ตั้งใจ (specialization)* |

📄 รายละเอียดเต็ม + ทุก % → **[REPORT.md](./REPORT.md)**

## Pipeline (7 ขั้น)

`Typhoon2 3B` → **vocab trim** (128k→45k vocab) → **layer prune** (28→24) → **QLoRA recovery** (ppl 14→7.6) → **quantize** (GGUF Q4_K_M 1.53GB) → **RAG deploy + eval**

ทุกขั้นรันจริงบน **RTX 3050 Laptop 4GB** (ไม่ต้องใช้คลาวด์)

## โครงสร้าง

```
data/            dataset โดเมน (train/val/test + corpus + chunks)
scripts/         phase3–7 (vocab trim / prune / recovery / quantize / rag / eval)
notebooks/       01–07 เวอร์ชัน Kaggle (พร้อมรันบน GPU ฟรี)
artifacts/       โมเดล + gguf + rag index + ผล eval (ไม่ commit — ดู .gitignore)
REPORT.md        รายงานเทียบเดิม↔ใหม่ครบ 3 แกน
PROGRESS.md      ติดตามความคืบหน้าทุก phase
experiments.csv  log ผลการทดลอง
llm_pruning_project_plan.md   แผนต้นฉบับ (ไทย)
```

## วิธีรัน (local)

```powershell
. .tools\activate.ps1            # env + venv (caches/HF อยู่บน D)
$env:PYTHONUTF8="1"
# ดูคำสั่งแต่ละ phase ใน REPORT.md (ภาคผนวก) หรือ PROGRESS.md
```

RAG demo:
```powershell
.venv\Scripts\python.exe scripts\phase7_rag.py --build        # index 361 chunks
# start llama-server (Q4_K_M) แล้ว:
.venv\Scripts\python.exe scripts\phase7_rag.py --ask "นิสิตขอทรานสคริปต์ทำอย่างไร"
```

## เทคนิคที่ใช้

PyTorch · Transformers · PEFT (QLoRA) · bitsandbytes · FAISS · llama.cpp (GGUF) ·
ShortGPT (layer pruning) · vocabulary trimming · multilingual-e5 (embedding) · Gemini (data gen + judge)

## สถานะ

Phase 2–6 ✅ เสร็จ • Phase 7: RAG + eval (B1↔B3) + REPORT ✅ • เหลือ (ทางเลือก) B2 baseline + lm-eval ทางการ
