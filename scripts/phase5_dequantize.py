"""แปลงโมเดล recovered ที่เซฟเป็น bnb 4-bit → bf16 มาตรฐาน (สำหรับ Phase 6 GGUF)

merge_and_unload บนโมเดล 4-bit เซฟกลับเป็น 4-bit (U8 + absmax) ซึ่ง convert_hf_to_gguf
อ่านไม่ได้ → ต้อง dequantize เป็น bf16 ก่อน. ค่าตัวเลขเท่าเดิม (ppl ไม่เปลี่ยน)

รัน:  .venv\\Scripts\\python.exe scripts\\phase5_dequantize.py
"""
import argparse, os, json
from _bootstrap import bootstrap
bootstrap()
import torch, torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.join(os.path.dirname(__file__), "..")
ART = os.path.join(ROOT, "artifacts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=os.path.join(ART, "pruned_moderate_recovered"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or (args.in_dir.rstrip("/\\") + "_bf16")

    print("โหลดโมเดล 4-bit …")
    tok = AutoTokenizer.from_pretrained(args.in_dir)
    model = AutoModelForCausalLM.from_pretrained(args.in_dir, device_map="cuda")  # อ่าน quant config เอง

    # แทน Linear4bit ทุกตัวด้วย nn.Linear bf16 (weights dequantized)
    n = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, bnb.nn.Linear4bit):
            w = bnb.functional.dequantize_4bit(mod.weight.data, mod.weight.quant_state).to(torch.bfloat16)
            lin = nn.Linear(w.shape[1], w.shape[0], bias=mod.bias is not None)
            lin.weight = nn.Parameter(w.cpu(), requires_grad=False)
            if mod.bias is not None:
                lin.bias = nn.Parameter(mod.bias.detach().cpu().to(torch.bfloat16), requires_grad=False)
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            setattr(parent, name.rsplit(".", 1)[-1], lin)
            n += 1
    print(f"dequantize {n} Linear4bit → bf16")

    # เคลียร์ flag bnb ก่อน (ไม่งั้น .to(dtype) ถูกบล็อก) + ลบ quantization_config
    for attr in ("is_quantized", "is_loaded_in_4bit", "is_loaded_in_8bit"):
        if hasattr(model, attr):
            setattr(model, attr, False)
    model.quantization_method = None      # guard ใน .to() เช็คตัวนี้
    if hasattr(model, "hf_quantizer"):
        model.hf_quantizer = None
    for cfgobj in (model.config, getattr(model, "config", None)):
        if cfgobj is not None and hasattr(cfgobj, "quantization_config"):
            del cfgobj.quantization_config
    model = model.to(torch.bfloat16)
    model.config.torch_dtype = "bfloat16"
    model.config.use_cache = True

    os.makedirs(out_dir, exist_ok=True)
    # save_pretrained ติด revert_weight_conversion ของ bnb (transformers 5.9) → save state_dict เอง
    from safetensors.torch import save_file
    sd = {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in model.state_dict().items()}
    sd = {k: v for k, v in sd.items() if "quant" not in k and "absmax" not in k}  # กันเศษ bnb
    save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    model.config.save_pretrained(out_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    # คัด report เดิมมาด้วย
    rep_fp = os.path.join(args.in_dir, "recovery_report.json")
    if os.path.exists(rep_fp):
        rep = json.load(open(rep_fp, encoding="utf-8"))
        rep["note"] = "dequantized 4bit->bf16 for GGUF; ppl เท่าเดิม"
        json.dump(rep, open(os.path.join(out_dir, "recovery_report.json"), "w"),
                  ensure_ascii=False, indent=2)
    sz = os.path.getsize(os.path.join(out_dir, "model.safetensors")) / 1e9
    print(f"✅ เซฟ bf16 → {out_dir} ({sz:.2f}GB)")

    # sanity generate
    ids = tok(tok.apply_chat_template(
        [{"role": "user", "content": "นิสิตลงทะเบียนเรียนได้ที่ไหน"}],
        add_generation_prompt=True, tokenize=False), return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        g = model.generate(ids, max_new_tokens=50, do_sample=False, pad_token_id=tok.eos_token_id)
    print("probe:", tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True)[:220])


if __name__ == "__main__":
    main()
