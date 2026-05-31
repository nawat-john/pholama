import os, sys
os.environ.setdefault("HF_HOME", r"D:\Code\pholama\.cache\hf")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers, bitsandbytes
print("transformers", transformers.__version__, "| bnb", bitsandbytes.__version__)

MODEL_ID = "scb10x/llama3.2-typhoon2-3b-instruct"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
prompt = "ขอวิธีขอทุนการศึกษาสั้นๆ"
enc = tok.apply_chat_template([{"role":"user","content":prompt}],
                              add_generation_prompt=True, return_tensors="pt", return_dict=True)

def gen(model, tag):
    e = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        o = model.generate(**e, max_new_tokens=40, do_sample=False)
    txt = tok.decode(o[0, e["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[{tag}] {txt!r}")
    return txt

mode = sys.argv[1] if len(sys.argv) > 1 else "8bit"

if mode == "8bit":
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
    m.eval(); gen(m, "8bit")
elif mode == "4bit_nodq":
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                             device_map="auto", torch_dtype=torch.float16)
    m.eval(); gen(m, "4bit_fp16_nodq")
elif mode == "cpu":
    # โหลด fp32 บน CPU — ช้าแต่ชี้ขาดว่าตัวโมเดลปกติไหม
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="cpu")
    m.eval(); gen(m, "cpu_fp32")
print("DONE", mode)
