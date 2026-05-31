"""Phase 5 — Recovery fine-tune (QLoRA) กู้คุณภาพในโดเมนหลัง pruning

- โหลด pruned model แบบ 4-bit (nf4) + LoRA (q/k/v/o/gate/up/down)
- เทรนบน train.jsonl โดยจัด chat template + **mask prompt** (เทรนเฉพาะคำตอบ)
- gradient checkpointing + paged_adamw_8bit → ประหยัด VRAM (ลองบน RTX 3050 4GB)
- วัด val perplexity ก่อน/หลัง → merge adapter → เซฟ

รัน smoke test (ดูว่ารันได้ไหม):
  .venv\\Scripts\\python.exe scripts\\phase5_recovery.py --smoke
รันจริง:
  .venv\\Scripts\\python.exe scripts\\phase5_recovery.py --model-dir artifacts/pruned_moderate --epochs 2
"""
import argparse, json, math, os, random
from _bootstrap import bootstrap
bootstrap()
import torch
# หมายเหตุ: ห้าม import transformers.Trainer / datasets — มันดึง pyarrow ที่ segfault บน Windows
# จึงเขียน training loop เองด้วย PyTorch ล้วน (core QLoRA ทำงานได้ ยืนยันแล้ว)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.optimization import get_cosine_schedule_with_warmup

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
ART = os.path.join(ROOT, "artifacts")


def build_examples(tok, rows, max_len):
    """chat template + mask prompt (labels=-100 ที่ส่วน prompt)"""
    out = []
    for ex in rows:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": ex["question"]}],
            add_generation_prompt=True, tokenize=False)
        full = tok.apply_chat_template(
            [{"role": "user", "content": ex["question"]},
             {"role": "assistant", "content": ex["answer"]}], tokenize=False)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        f_ids = tok(full, add_special_tokens=False).input_ids[:max_len]
        labels = list(f_ids)
        for i in range(min(len(p_ids), len(labels))):
            labels[i] = -100
        out.append({"input_ids": f_ids, "labels": labels, "attention_mask": [1] * len(f_ids)})
    return out


class Collator:
    def __init__(self, pad_id):
        self.pad = pad_id
    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        ii, ll, am = [], [], []
        for b in batch:
            n = m - len(b["input_ids"])
            ii.append(b["input_ids"] + [self.pad] * n)
            ll.append(b["labels"] + [-100] * n)
            am.append(b["attention_mask"] + [0] * n)
        return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
                "attention_mask": torch.tensor(am)}


@torch.no_grad()
def perplexity(model, tok, rows, max_len, n):
    model.eval()
    nll, ntok = 0.0, 0
    for ex in rows[:n]:
        full = tok.apply_chat_template(
            [{"role": "user", "content": ex["question"]},
             {"role": "assistant", "content": ex["answer"]}], tokenize=False)
        ids = tok(full, return_tensors="pt", truncation=True, max_length=max_len,
                  add_special_tokens=False).input_ids.to(model.device)
        if ids.shape[1] < 2:
            continue
        out = model(ids, labels=ids); t = ids.shape[1] - 1
        nll += out.loss.item() * t; ntok += t
    return math.exp(nll / max(ntok, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(ART, "pruned_moderate"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--ppl-n", type=int, default=100)
    ap.add_argument("--smoke", action="store_true", help="ทดสอบเร็ว: 3 steps, seq 256, ตัวอย่างน้อย")
    args = ap.parse_args()

    if args.smoke:
        args.max_steps, args.seq_len, args.ppl_n, args.grad_accum = 3, 256, 8, 2

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print(f"device cuda: {torch.cuda.is_available()} | model: {args.model_dir}")
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, quantization_config=bnb, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    if torch.cuda.is_available():
        print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_rows = [json.loads(l) for l in open(os.path.join(DATA, "train.jsonl"), encoding="utf-8")]
    val_rows = [json.loads(l) for l in open(os.path.join(DATA, "val.jsonl"), encoding="utf-8")]
    if args.smoke:
        train_rows = train_rows[:16]
    train_ds = build_examples(tok, train_rows, args.seq_len)
    print(f"train examples: {len(train_ds)} | val: {len(val_rows)}")

    print("วัด ppl ก่อนเทรน …")
    ppl_before = perplexity(model, tok, val_rows, args.seq_len, args.ppl_n)
    print(f"  ppl_before = {ppl_before:.3f}")

    out_dir = args.out or (args.model_dir.rstrip("/\\") + "_recovered")
    collate = Collator(tok.pad_token_id)
    dev = model.device

    # จำนวน optimizer step รวม
    per_epoch = math.ceil(len(train_ds) / (args.batch * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else int(per_epoch * args.epochs)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total_steps), total_steps)
    print(f"train: total_steps={total_steps} (per_epoch≈{per_epoch}) | trainable tensors={len(trainable)}")

    # ---- manual training loop (PyTorch ล้วน) ----
    model.train()
    model.config.use_cache = False
    step, micro, run_loss = 0, 0, 0.0
    opt.zero_grad()
    rng = random.Random(42)
    done = False
    epoch = 0
    while not done:
        epoch += 1
        order = list(range(len(train_ds)))
        rng.shuffle(order)
        for bstart in range(0, len(order), args.batch):
            idxs = order[bstart:bstart + args.batch]
            batch = collate([train_ds[i] for i in idxs])
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            run_loss += out.loss.item()
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 5 == 0 or step == total_steps:
                    print(f"  step {step}/{total_steps} | loss {run_loss/ (5*args.grad_accum):.4f} "
                          f"| lr {sched.get_last_lr()[0]:.2e}"
                          + (f" | VRAM {torch.cuda.max_memory_allocated()/1e9:.2f}GB" if dev.type=='cuda' else ""))
                    run_loss = 0.0
                if step >= total_steps:
                    done = True; break
        if epoch > 1000:
            break

    print("วัด ppl หลังเทรน …")
    ppl_after = perplexity(model, tok, val_rows, args.seq_len, args.ppl_n)
    print(f"  ppl_after = {ppl_after:.3f}  (ก่อน {ppl_before:.3f})")

    if args.smoke:
        print("\n✅ SMOKE ผ่าน — รันได้ในเครื่องนี้ (ไม่ OOM)")
        if torch.cuda.is_available():
            print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
        return

    # ---- เซฟ adapter ก่อน แล้ว merge บน base bf16 (ไม่ใช่ 4-bit!) ----
    # ⚠️ merge_and_unload บนโมเดล 4-bit ทำคุณภาพตกมาก (re-quantize rounding: ppl 9→17)
    #    วิธีถูก = เซฟ LoRA adapter → โหลด base bf16 เต็มความละเอียด → apply+merge
    adapter_dir = out_dir + "_adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"เซฟ adapter → {adapter_dir}")

    import gc
    from peft import PeftModel
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("โหลด base bf16 + merge adapter …")
    base = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).to(
        "cuda" if torch.cuda.is_available() else "cpu")
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.config.use_cache = True

    ppl_merged = perplexity(merged, tok, val_rows, args.seq_len, args.ppl_n)
    print(f"  ppl_merged (bf16, ของจริง) = {ppl_merged:.3f}")

    # save state_dict เอง (เลี่ยง revert_weight_conversion ใน transformers 5.9)
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in merged.state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    merged.config.torch_dtype = "bfloat16"
    merged.config.save_pretrained(out_dir)
    if getattr(merged, "generation_config", None) is not None:
        merged.generation_config.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    json.dump({"base": os.path.basename(args.model_dir), "ppl_before": round(ppl_before, 3),
               "ppl_after_4bit_adapter": round(ppl_after, 3), "ppl_merged_bf16": round(ppl_merged, 3),
               "rank": args.rank, "alpha": args.alpha, "epochs": args.epochs, "seq_len": args.seq_len},
              open(os.path.join(out_dir, "recovery_report.json"), "w"), ensure_ascii=False, indent=2)
    print(f"✅ เซฟ recovered (bf16) → {out_dir}")


if __name__ == "__main__":
    main()
