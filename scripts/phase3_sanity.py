"""ยืนยันโมเดล vocab-trimmed: โหลด + generate ภาษาไทย (เช็ค embedding↔tokenizer align)"""
import os
from _bootstrap import bootstrap
bootstrap()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.join(os.path.dirname(__file__), "..")
DIR = os.path.join(ROOT, "artifacts", "vocab_trimmed")

tok = AutoTokenizer.from_pretrained(DIR)
print("tokenizer len:", len(tok), "| vocab_size:", tok.vocab_size)
model = AutoModelForCausalLM.from_pretrained(DIR, dtype=torch.bfloat16)
print("model vocab_size:", model.config.vocab_size,
      "| embed rows:", model.get_input_embeddings().weight.shape[0])
assert model.get_input_embeddings().weight.shape[0] == len(tok), "❌ embed≠tokenizer!"

for prompt in ["นิสิตจุฬาฯ ลงทะเบียนเรียนได้ที่ไหน",
               "What is Chulalongkorn University Office of the Registrar?"]:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=40, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"\nQ: {prompt}\nA: {gen.strip()[:200]}")
print("\n✅ โมเดล vocab-trimmed generate ได้ (embedding↔tokenizer align)")
