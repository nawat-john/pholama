# LLM Pruning Project — Chula Registrar Chatbot

Compress + specialize **Typhoon 2 3B** (Llama 3.2 base) into a domain model for Q&A about the
**Office of the Registrar, Chulalongkorn University** (reg.chula.ac.th) — smaller, faster, runs on CPU.

## Results (original → new)

| | Original (Typhoon 2 3B) | New (full pipeline) | Change |
|---|---:|---:|---:|
| Parameters | 3.21B | 2.55B | **−20.5%** |
| Deploy file | 6.43 GB (bf16) | **1.53 GB** (Q4_K_M) | **−76.2%** |
| RAM | 2.25 GB (GPU) | ~1.9 GB (CPU) | **−15.6%** |
| Speed | 10.6 tok/s (GPU) | **23.5 tok/s (CPU)** | **+122%** |
| In-domain quality | 35.0% | **47.5%** | **+35.7%** |
| Code/math (out-of-domain) | normal | dropped | *intentional (specialization)* |

📄 Full details + all percentages → **[REPORT.md](./REPORT.md)**

## Pipeline (7 stages)

`Typhoon2 3B` → **vocab trim** (128k→45k vocab) → **layer prune** (28→24) → **QLoRA recovery** (ppl 14→7.6) → **quantize** (GGUF Q4_K_M 1.53 GB) → **RAG deploy + eval**

Every stage ran on a real **RTX 3050 Laptop 4 GB** (no cloud needed).

## Structure

```
data/            domain dataset (train/val/test + corpus + chunks)
scripts/         phase3–7 (vocab trim / prune / recovery / quantize / rag / eval)
notebooks/       01–07 Kaggle versions (ready to run on free GPU)
artifacts/       models + gguf + rag index + eval results (not committed — see .gitignore)
REPORT.md        original↔new comparison across all 3 axes
PROGRESS.md      per-phase progress tracker
experiments.csv  experiment log
llm_pruning_project_plan.md   original plan (Thai)
```

## How to run (local)

```powershell
. .tools\activate.ps1            # env + venv (caches/HF live on drive D)
$env:PYTHONUTF8="1"
# per-phase commands: see the appendix in REPORT.md or PROGRESS.md
```

RAG demo:
```powershell
.venv\Scripts\python.exe scripts\phase7_rag.py --build        # index 361 chunks
# start llama-server (Q4_K_M), then:
.venv\Scripts\python.exe scripts\phase7_rag.py --ask "How do I request a transcript?"
```

## Stack

PyTorch · Transformers · PEFT (QLoRA) · bitsandbytes · FAISS · llama.cpp (GGUF) ·
ShortGPT (layer pruning) · vocabulary trimming · multilingual-e5 (embedding) · Gemini (data gen + judge)

## Status

Phases 2–6 ✅ done • Phase 7: RAG + eval (B1↔B3) + REPORT ✅ • remaining (optional): B2 baseline + official lm-eval
