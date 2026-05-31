"""ตรวจ env: import ครบ + CUDA + bitsandbytes 4-bit forward บน GPU จริง"""
import importlib, os
os.environ.setdefault("HF_HOME", r"D:\Code\pholama\.cache\hf")

mods = ["numpy", "torch", "transformers", "accelerate", "datasets", "peft",
        "trl", "bitsandbytes", "pdfplumber", "google.generativeai",
        "faiss", "sentence_transformers"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"OK   {m:<24} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"FAIL {m:<24} {type(e).__name__}: {str(e)[:80]}")

import torch
ok = torch.cuda.is_available()
print(f"\nCUDA available: {ok}")
if ok:
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} | VRAM: {p.total_memory/1e9:.2f} GB")
    # ทดสอบ 4-bit linear บน GPU (จุดที่มักพังบน Windows)
    from bitsandbytes.nn import Linear4bit
    lin = Linear4bit(64, 64, bias=False).cuda()
    y = lin(torch.randn(2, 64, device="cuda", dtype=torch.float16))
    torch.cuda.synchronize()
    print(f"bitsandbytes 4-bit forward OK | out {tuple(y.shape)} | VRAM {torch.cuda.memory_allocated()/1e6:.1f} MB")
