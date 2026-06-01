# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

An **implemented** Python project that runs the 7-phase LLM compression pipeline below (working language **Thai** — respond in Thai unless asked). Phases 2–6 done; Phase 7 (RAG + eval B1↔B3 + report) done, B2/lm-eval optional.

- `llm_pruning_project_plan.md` — แผนต้นฉบับ (Thai) • `PROGRESS.md` — สถานะทุก phase (อ่านบล็อก "ทำต่อ SESSION หน้า" ก่อน) • `REPORT.md` — รายงานผลเทียบเดิม↔ใหม่ • `README.md` — ภาพรวม • `experiments.csv` — log
- `scripts/phase{3..7}_*.py` — โค้ดรันจริง (local) • `notebooks/01–07` — เวอร์ชัน Kaggle • `data/` — dataset • `artifacts/` — โมเดล/gguf/eval (gitignored)

**Env/run:** venv อยู่ที่ `.venv` (เปิดด้วย `. .tools\activate.ps1`). ต้องตั้ง `$env:PYTHONUTF8="1"` เสมอ (กัน cp874 crash). คำสั่งต่อ phase ดูใน `REPORT.md` ภาคผนวก / `PROGRESS.md`. ไม่มี test/linter ทางการ — verify ด้วยการรันสคริปต์ + เช็ค ppl/generate

**ถาม-ตอบ (deliverable):** `app.py` = entry-point RAG (auto-start `llama-server` Q4_K_M → retrieve e5-small+faiss → ตอบ). รันตรง: `.venv\Scripts\python.exe app.py "คำถาม"` (flags: `--no-rag -k --max-tokens --show-sources`). มี **exe ห่อบางๆ**: `dist\pholama.exe "คำถาม"` (จาก `pholama_launcher.py`, stdlib ล้วน → shell out เข้า venv; ไม่ bundle torch). Build ใหม่: `build_exe.ps1` (หรือ `.venv\Scripts\python.exe -m PyInstaller --onefile --console --name pholama pholama_launcher.py`). หมายเหตุ: แต่ละครั้งโหลด torch+e5 ใหม่ (~นาที); server ค้างไว้ครั้งถัดไปไม่ต้องโหลด GGUF ซ้ำ

**Local runtime gotchas (สำคัญ — ดู memory ด้วย):** เครื่องนี้ RTX 3050 Laptop **4GB** — โมเดล bf16 รันได้ผ่าน CUDA **sysmem fallback** (spill ลง RAM, ช้ากว่า VRAM แต่เร็วกว่า CPU ~9×) • **ห้าม `import transformers.Trainer`/`datasets`** ในสคริปต์ local → **pyarrow segfault** (exit -1073741819) → เขียน training loop เอง • **merge LoRA ต้องทำบน bf16 base** ไม่ใช่ 4-bit (ไม่งั้น ppl เสีย) • **sentence_transformers ก็ segfault** → ใช้ transformers AutoModel ตรงๆ • pip install ของ gguf/llama.cpp อาจทับ torch เป็น CPU build

## The project the plan describes

The plan is a 7-phase pipeline to **compress and specialize an LLM**: take **Typhoon 2 3B** (a Llama 3.2 3B–based Thai model) and shrink it from ~3.21B params to ~2.1B (Moderate) or ~1.76B (Aggressive), targeting ~1.5–2 GB RAM after Q4 quantization, specialized for **Thai-English student-affairs (กิจการนักศึกษา) Q&A**.

Pipeline phases (each is a section in the plan):
1. **Setup + baseline** — measure original model quality (in-domain, out-of-domain), RAM, latency
2. **Domain dataset** — build 2k–10k Thai-English Q&A pairs; reused for calibration, recovery training, and eval. Strict train/val/test split; **never leak test into calibration/training**
3. **Vocabulary trimming** — drop dead (unused-language) tokens from the tied `embed_tokens`/`lm_head`; ~128k → ~40k vocab
4. **Structured pruning** (project core) — layer pruning (ShortGPT-style, 28→20 layers) and optional FFN width pruning, using the **domain dataset as calibration data** to bias the model toward the domain
5. **Recovery fine-tune** — QLoRA on the domain train set to recover in-domain quality lost to pruning, then merge the adapter
6. **Quantization** — export to GGUF (Q4_K_M / Q5_K_M / Q8) via llama.cpp
7. **Deploy + eval + report** — RAG (Ollama + FAISS/Chroma) so facts come from retrieval, not weights; evaluate on 3 axes (in-domain quality, out-of-domain regression, efficiency)

Intended tooling once implementation starts: PyTorch, Hugging Face `transformers`/`peft`/`bitsandbytes`/`datasets`, `lm-eval-harness`, `llama.cpp`, on free Colab/Kaggle GPUs.

## Key constraints baked into the plan

- **Three baselines are mandatory** for any results: (1) original model, (2) original + QLoRA + quantize (no prune), (3) full pipeline. The point is proving the full pipeline beats baseline (2) on size/speed for an acceptable quality cost — preserve this comparison framing in any report or experiment.
- **Out-of-domain regression is a goal, not a bug**: losing code/math ability is the intended proof that specialization worked. Don't "fix" it.
- Verify the real `config.json` of Typhoon 2 3B before relying on the plan's assumed Llama 3.2 3B numbers (hidden 3072, 28 layers, 24 heads / 8 KV heads, FFN 8192, vocab 128256, tied embeddings). **Confirmed: matches the plan (3.213B params).**
- **Phase 3 reality vs plan:** the plan assumed vocab trims 128k→~40k by dropping dead languages. Actual: the Llama 3.2 tokenizer is **English-dominant** (ASCII ~97.7k, Thai only ~2.5k, genuinely-dead langs CJK/Korean/Arabic/Cyrillic only ~24k). Dropping dead-only = −18.8% (→104k). Hitting ~40k requires the **aggressive** mode (also drop rarely-used English/code tokens by tiktoken frequency rank) — chosen for this project; result is 44,926 vocab, 2.9568B params. Any text still encodes via byte fallback. See `scripts/phase3_vocab_trim.py` (`--aggressive` flag) and `PROGRESS.md`.

## Working in this repo

- When editing `llm_pruning_project_plan.md`, keep its structure: per-phase sections with **เป้าหมาย / ขั้นตอนย่อย / Deliverable / Compute** and the running parameter-count math (the param/RAM tables are internally consistent — update dependent numbers together).
- If/when implementation code is added, update this file with the actual build/run/test commands and architecture.
