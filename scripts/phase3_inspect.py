"""Phase 3 — ขั้นสำรวจ: โหลด tokenizer จริง + ดูโครงสร้าง tokenizer.json
รัน:  .venv\\Scripts\\python.exe scripts\\phase3_inspect.py
"""
import json
import os

from _bootstrap import bootstrap
bootstrap()

from transformers import AutoTokenizer

MODEL_ID = "scb10x/llama3.2-typhoon2-3b-instruct"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
print("loaded:", type(tok).__name__, "| is_fast:", tok.is_fast)
print("vocab_size:", tok.vocab_size, "| len(tok):", len(tok))
print("n_special:", len(tok.all_special_ids))
print("special ids sample:", sorted(tok.all_special_ids)[:5], "...", sorted(tok.all_special_ids)[-3:])
print("bos/eos/pad:", tok.bos_token_id, tok.eos_token_id, tok.pad_token_id)

# dump tokenizer.json structure (top-level keys + model keys)
tj_path = os.path.join(tok.name_or_path if os.path.isdir(tok.name_or_path) else "", "")
# find cached tokenizer.json
from transformers.utils import cached_file
tjf = cached_file(MODEL_ID, "tokenizer.json")
tj = json.load(open(tjf, encoding="utf-8"))
print("\ntokenizer.json top keys:", list(tj.keys()))
print("model keys:", list(tj["model"].keys()))
print("model.type:", tj["model"].get("type"))
print("n vocab in model:", len(tj["model"]["vocab"]))
print("n merges:", len(tj["model"]["merges"]))
print("merges[0]:", repr(tj["model"]["merges"][0]))
print("n added_tokens:", len(tj.get("added_tokens", [])))
print("added_tokens[0]:", tj["added_tokens"][0])
print("added_tokens[-1] id:", tj["added_tokens"][-1]["id"], tj["added_tokens"][-1]["content"][:40])
# is special token also in model.vocab?
sp = tj["added_tokens"][0]["content"]
print(f"is added '{sp}' in model.vocab? ->", sp in tj["model"]["vocab"])
print("pre_tokenizer:", json.dumps(tj.get("pre_tokenizer"), ensure_ascii=False)[:300])
print("\ncached tokenizer.json at:", tjf)
