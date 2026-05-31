import os
os.environ.setdefault("HF_HOME", r"D:\Code\pholama\.cache\hf")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "scb10x/llama3.2-typhoon2-3b-instruct"
out = []
def log(*a):
    s = " ".join(str(x) for x in a)
    out.append(s); print(s)

log("transformers test")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
model.eval()

prompt = "สวัสดีครับ ขอวิธีขอทุนการศึกษาสั้นๆ"

# --- วิธี A: chat template (return_dict) ---
log("\n=== A: apply_chat_template return_dict ===")
enc = tok.apply_chat_template([{"role":"user","content":prompt}],
                              add_generation_prompt=True, return_tensors="pt", return_dict=True)
log("keys:", list(enc.keys()), "| input_ids shape:", tuple(enc["input_ids"].shape))
log("decoded prompt:", repr(tok.decode(enc["input_ids"][0])[:300]))
enc = {k: v.to(model.device) for k, v in enc.items()}
with torch.no_grad():
    o = model.generate(**enc, max_new_tokens=60, do_sample=False)
log("OUT-A:", repr(tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)))

# --- วิธี B: plain encode no template ---
log("\n=== B: plain text, no chat template ===")
ids = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    o = model.generate(**ids, max_new_tokens=60, do_sample=False)
log("OUT-B:", repr(tok.decode(o[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)))

# --- วิธี C: sampling instead of greedy ---
log("\n=== C: chat template + sampling ===")
enc2 = tok.apply_chat_template([{"role":"user","content":prompt}],
                               add_generation_prompt=True, return_tensors="pt", return_dict=True)
enc2 = {k: v.to(model.device) for k, v in enc2.items()}
with torch.no_grad():
    o = model.generate(**enc2, max_new_tokens=60, do_sample=True, temperature=0.7, top_p=0.9)
log("OUT-C:", repr(tok.decode(o[0, enc2["input_ids"].shape[1]:], skip_special_tokens=True)))

open(r"D:\Code\pholama\.cache\diag_gen.txt", "w", encoding="utf-8").write("\n".join(out))
log("\nDONE")
