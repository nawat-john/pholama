"""วิเคราะห์ breakdown vocab: Thai / Latin-ASCII / punct / อื่นๆ + การกระจาย id"""
import json
from _bootstrap import bootstrap
bootstrap()
from transformers import AutoTokenizer
from transformers.utils import cached_file
from phase3_vocab_trim import token_bytes, BYTE_LEVEL_CHARS, count_corpus_tokens, MODEL_ID

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tj = json.load(open(cached_file(MODEL_ID, "tokenizer.json"), encoding="utf-8"))
vocab = tj["model"]["vocab"]

def classify(s):
    bs = token_bytes(s)
    if bs is None:
        return "weird"
    if s in BYTE_LEVEL_CHARS:
        return "byte"
    has_thai_lead = any(b == 0xE0 for b in bs)
    has_latin1 = any(b in (0xC2, 0xC3) for b in bs)
    has_punct = any(b == 0xE2 for b in bs)
    bad = any(b >= 0xC0 and b not in (0xC2, 0xC3, 0xE0, 0xE2) for b in bs)
    allascii = all(b < 0x80 for b in bs)
    if bad:
        return "other_lang"          # มี lead byte ของภาษาที่ตัด
    if has_thai_lead:
        return "thai"
    if allascii:
        return "ascii"
    if has_latin1:
        return "latin1"
    if has_punct:
        return "punct"
    return "cont_only"               # continuation-only fragment (กำกวม)

from collections import Counter
cnt = Counter()
thai_ids, ascii_ids = [], []
for s, i in vocab.items():
    c = classify(s)
    cnt[c] += 1
    if c == "thai":
        thai_ids.append(i)
    elif c == "ascii":
        ascii_ids.append(i)

print("breakdown (base 128000):")
for k, v in cnt.most_common():
    print(f"  {k:12s}: {v}")

corpus = {i for i in count_corpus_tokens(tok) if i < 128000}
print(f"\ncorpus distinct base tokens: {len(corpus)}")

# floor ถ้า aggressive: bytes + specials + thai + corpus
floor = set(thai_ids) | corpus
print(f"floor (thai+corpus, ไม่รวม byte/special): {len(floor)}")
print(f"ascii tokens: {len(ascii_ids)} | min id {min(ascii_ids)} max id {max(ascii_ids)}")

# ถ้าจะ hit ~40k base: floor + ascii ที่ id ต่ำสุด เติมจนครบ
for target in (40000, 50000, 60000):
    need = target - len(floor) - 256  # 256 bytes
    asc_sorted = sorted(i for i in ascii_ids if i not in floor)
    take = asc_sorted[:max(need, 0)]
    cutoff = take[-1] if take else 0
    print(f"target base ~{target}: เติม ascii id<{cutoff} ({len(take)} ตัว) → รวม ~{len(floor)+256+len(take)}")
