"""Phase 3 — Vocabulary trimming (ทำงานจริง บน CPU)

ตัด dead tokens (ภาษาที่ไม่ใช้: CJK/เกาหลี/อาหรับ/ซีริลลิก/อินดิก/emoji) ออกจาก
tied embedding (embed_tokens = lm_head) ของ Typhoon2 3B แล้ว reindex vocab ใหม่

กลยุทธ์เลือก token ที่ "เก็บ":
  1. byte-level alphabet ครบ 256 ตัว        → กัน encode พัง (ทุก byte ต้อง map ได้)
  2. ทุก special/added token (256 ตัว)       → กัน chat template พัง
  3. token ที่ทุก byte อยู่ในกลุ่มอนุญาต      → ASCII + Thai(0xE0) + Latin-1(0xC2/C3) + punct(0xE2)
  4. token ที่ "ปรากฏจริง" ใน corpus/train/val → กัน token โดเมนหลุด
  + closure ของ merges (ถ้าเก็บ token ผล ต้องเก็บ parent) → รักษา segmentation

ขั้นตอน:
  (A) วิเคราะห์ + สร้าง tokenizer ใหม่ + ทดสอบ roundtrip  [default — ไม่ต้องโหลดโมเดล 6.4GB]
  (B) --with-model : โหลดโมเดล ตัด embedding ตาม mapping เซฟโมเดลใหม่

รัน:
  .venv\\Scripts\\python.exe scripts\\phase3_vocab_trim.py            # (A) tokenizer เท่านั้น
  .venv\\Scripts\\python.exe scripts\\phase3_vocab_trim.py --with-model
"""
import argparse
import json
import os
import shutil

from _bootstrap import bootstrap
bootstrap()

MODEL_ID = "scb10x/llama3.2-typhoon2-3b-instruct"
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "artifacts", "vocab_trimmed")  # tokenizer (+model ถ้า --with-model)


# ---- byte-level (GPT-2/Llama3 ByteLevel) ----
def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}  # unicode-char -> byte


CHAR2BYTE = bytes_to_unicode()
BYTE_LEVEL_CHARS = set(CHAR2BYTE.keys())  # 256 single chars = byte alphabet

# lead/continuation bytes ที่อนุญาต (UTF-8 ของ ASCII + Latin-1 + Thai + punct/symbols)
#   ASCII           0x00-0x7F
#   continuation    0x80-0xBF (ใช้ร่วมหลายสคริปต์ → อนุญาตไว้กัน Thai fragment หลุด)
#   Latin-1 lead    0xC2,0xC3
#   Thai lead       0xE0
#   punct/symbol/currency lead 0xE2  (U+2000–2FFF)
ALLOWED_BYTES = set(range(0x00, 0xC0)) | {0xC2, 0xC3, 0xE0, 0xE2}


def token_bytes(tokstr):
    """แปลง token string (byte-level encoded) กลับเป็น bytes จริง"""
    try:
        return bytes(CHAR2BYTE[c] for c in tokstr)
    except KeyError:
        return None  # มีตัวอักษรนอก byte-alphabet (ไม่ควรเกิดกับ BPE vocab)


def allowed_script(tokstr):
    bs = token_bytes(tokstr)
    if bs is None:
        return False
    return all(b in ALLOWED_BYTES for b in bs)


def categorize(tokstr):
    """แยกประเภท token สำหรับโหมด aggressive"""
    bs = token_bytes(tokstr)
    if bs is None:
        return "weird"
    if tokstr in BYTE_LEVEL_CHARS:
        return "byte"
    if any(b >= 0xC0 and b not in (0xC2, 0xC3, 0xE0, 0xE2) for b in bs):
        return "other_lang"          # ภาษาที่ตัดทิ้ง
    if any(b == 0xE0 for b in bs):
        return "thai"
    if all(b < 0x80 for b in bs):
        return "ascii"               # อังกฤษ/โค้ด (id ต่ำ = ใช้บ่อย)
    if any(b in (0xC2, 0xC3) for b in bs):
        return "latin1"
    if any(b == 0xE2 for b in bs):
        return "punct"
    return "cont_only"               # fragment กำกวม (เก็บไว้กันไทยหลุด)


def count_corpus_tokens(tok):
    """นับ token id ที่ปรากฏจริงใน corpus + train + val (ไม่แตะ test)"""
    seen = set()

    def feed(text):
        for tid in tok.encode(text, add_special_tokens=False):
            seen.add(tid)

    cp = os.path.join(DATA_DIR, "corpus_raw.txt")
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as f:
            for line in f:
                feed(line)
    for name in ("train.jsonl", "val.jsonl"):
        fp = os.path.join(DATA_DIR, name)
        if os.path.exists(fp):
            for line in open(fp, encoding="utf-8"):
                ex = json.loads(line)
                feed(ex.get("question", ""))
                feed(ex.get("answer", ""))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-model", action="store_true", help="โหลดโมเดล + ตัด embedding + เซฟ")
    ap.add_argument("--aggressive", action="store_true",
                    help="ตัด token อังกฤษ/โค้ดที่ใช้น้อยด้วย → เข้าใกล้ --target")
    ap.add_argument("--target", type=int, default=40000,
                    help="(aggressive) จำนวน base vocab เป้าหมายก่อน closure")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from transformers.utils import cached_file

    print("โหลด tokenizer …")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tj_path = cached_file(MODEL_ID, "tokenizer.json")
    tj = json.load(open(tj_path, encoding="utf-8"))
    snap_dir = os.path.dirname(tj_path)

    vocab = tj["model"]["vocab"]            # tokstr -> id (0..127999)
    merges = tj["model"]["merges"]          # [[left,right], ...]
    added = tj["added_tokens"]              # 256 specials, id 128000..128255
    id2str = {i: s for s, i in vocab.items()}
    n_base = len(vocab)
    print(f"base vocab={n_base} | merges={len(merges)} | specials={len(added)}")

    # ---- เลือก keep set (เฉพาะ base ids) ----
    corpus_ids = count_corpus_tokens(tok)
    corpus_base = {i for i in corpus_ids if i < n_base}
    print(f"token ที่ปรากฏจริงใน corpus/train/val: {len(corpus_ids)} (base {len(corpus_base)})")

    keep = set()
    if not args.aggressive:
        # ---- conservative: ตัดเฉพาะภาษาที่ตายจริง (เก็บ ASCII/Latin/Thai ทั้งหมด) ----
        n_byte, n_script, n_corpus = 0, 0, 0
        for s, i in vocab.items():
            k = False
            if s in BYTE_LEVEL_CHARS:           # 1. byte alphabet
                k = True; n_byte += 1
            elif allowed_script(s):             # 3. สคริปต์อนุญาต (incl. อังกฤษทั้งหมด)
                k = True; n_script += 1
            if i in corpus_base and not k:      # 4. ปรากฏจริง
                k = True; n_corpus += 1
            if k:
                keep.add(i)
        print(f"keep(base) [conservative] ก่อน closure: {len(keep)} "
              f"[byte={n_byte} script={n_script} corpus+={n_corpus}]")
    else:
        # ---- aggressive: floor = byte+thai+latin1+punct+cont+corpus, เติม ascii id ต่ำสุด ----
        ascii_ids = []
        cat_cnt = {}
        for s, i in vocab.items():
            c = categorize(s)
            cat_cnt[c] = cat_cnt.get(c, 0) + 1
            if c in ("byte", "thai", "latin1", "punct", "cont_only"):
                keep.add(i)
            elif c == "ascii":
                ascii_ids.append(i)
        keep |= corpus_base                      # corpus เก็บเสมอ (รวม ascii โดเมน)
        floor = len(keep)
        need = max(args.target - floor, 0)
        ascii_fill = sorted(i for i in ascii_ids if i not in keep)[:need]  # id ต่ำ=ใช้บ่อย
        keep |= set(ascii_fill)
        print(f"keep(base) [aggressive target={args.target}] ก่อน closure: {len(keep)}")
        print(f"  floor(byte+thai+latin1+punct+cont+corpus)={floor} + ascii_fill={len(ascii_fill)} "
              f"(เก็บ ascii id<{ascii_fill[-1] if ascii_fill else 0})")
        print(f"  cat: {cat_cnt}")

    # ---- closure ของ merges: เก็บ token ผล → ต้องเก็บ parent ----
    str2id = vocab
    result_to_parents = {}
    for a, b in merges:
        res = a + b
        if res in str2id:
            result_to_parents[str2id[res]] = (str2id.get(a), str2id.get(b))
    added_closure = 0
    stack = list(keep)
    while stack:
        tid = stack.pop()
        pa = result_to_parents.get(tid)
        if not pa:
            continue
        for p in pa:
            if p is not None and p not in keep:
                keep.add(p); stack.append(p); added_closure += 1
    print(f"closure เพิ่ม: {added_closure} → keep(base)={len(keep)}")

    # ---- reindex: base kept = 0..K-1 (เรียงตาม old id), specials = K..K+255 ----
    kept_base = sorted(keep)
    K = len(kept_base)
    old2new_base = {old: new for new, old in enumerate(kept_base)}
    new_vocab = {id2str[old]: new for old, new in old2new_base.items()}
    kept_str = set(new_vocab.keys())

    # filter merges: left/right/result ต้องอยู่ครบ
    new_merges = [[a, b] for a, b in merges
                  if a in kept_str and b in kept_str and (a + b) in kept_str]
    print(f"merges: {len(merges)} → {len(new_merges)}")

    # specials: เก็บครบ 256, id ใหม่ = K + (oldid-128000)
    new_added = []
    for t in added:
        t2 = dict(t)
        t2["id"] = K + (t["id"] - n_base)
        new_added.append(t2)

    # ---- ประกอบ tokenizer.json ใหม่ ----
    tj["model"]["vocab"] = new_vocab
    tj["model"]["merges"] = new_merges
    tj["added_tokens"] = new_added
    # อัปเดต id ใน post_processor (Llama3 อ้าง <|begin_of_text|> ตาม id)
    content2new = {t["content"]: t["id"] for t in new_added}
    pp = tj.get("post_processor")
    if pp:
        pj = json.dumps(pp)
        # special_tokens ใน TemplateProcessing เก็บ id เป็น list → patch ผ่าน object walk
        def patch(o):
            if isinstance(o, dict):
                if "id" in o and isinstance(o.get("id"), str) and o["id"] in content2new and "ids" in o:
                    o["ids"] = [content2new[o["id"]]]
                for v in o.values():
                    patch(v)
            elif isinstance(o, list):
                for v in o:
                    patch(v)
        patch(pp)

    new_total = K + len(new_added)
    print(f"\n=== สรุป vocab ===")
    print(f"  เดิม: {n_base + len(added)} (base {n_base} + special {len(added)})")
    print(f"  ใหม่: {new_total} (base {K} + special {len(new_added)})")
    print(f"  ตัดออก: {n_base - K} base tokens ({(n_base-K)/n_base*100:.1f}%)")

    # ---- เขียน tokenizer dir ใหม่ (copy config แล้วทับ tokenizer.json) ----
    os.makedirs(args.out, exist_ok=True)
    for fn in ("tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
        src = os.path.join(snap_dir, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, fn))
    json.dump(tj, open(os.path.join(args.out, "tokenizer.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    # บันทึก mapping เผื่อ debug / phase ถัดไป
    json.dump({"kept_base_old_ids": kept_base, "n_base_old": n_base, "K": K},
              open(os.path.join(args.out, "vocab_map.json"), "w"), )
    print(f"เซฟ tokenizer ใหม่ → {args.out}")

    # ---- ทดสอบ roundtrip ----
    print("\n=== roundtrip test (tokenizer ใหม่) ===")
    new_tok = AutoTokenizer.from_pretrained(args.out)
    tests = [
        "การลงทะเบียนเรียนของนิสิตใหม่ ปีการศึกษา 2569",
        "student dormitory registration deadline",
        "นิสิตต้องชำระค่าเล่าเรียนภายในวันที่ 15 ก.ค. 2568 (fee 25,000 บาท)",
        "Q: How to request transcript? A: ยื่นคำร้องที่ reg.chula.ac.th",
        # เคสยาก: อังกฤษหายาก/โค้ด/สัญลักษณ์ → aggressive ตัด token พวกนี้ ต้อง encode ผ่าน byte/subword
        "def fibonacci(n): return n if n<2 else fibonacci(n-1)+fibonacci(n-2)",
        "Pneumonoultramicroscopicsilicovolcanoconiosis & ❤ © → α β",
        "ภาษาอื่นที่ตัดทิ้ง: 你好 こんにちは 안녕하세요 (ต้องยัง decode กลับได้)",
    ]
    ok = True
    for s in tests:
        a = tok.encode(s, add_special_tokens=False)
        b = new_tok.encode(s, add_special_tokens=False)
        da = tok.decode(a); db = new_tok.decode(b)
        match = db.strip() == s.strip()
        ok = ok and match
        print(f"  {'✅' if match else '❌'} [{len(a)}→{len(b)} tok] {s[:45]}")
        if not match:
            print(f"      orig: {repr(da)}\n      new : {repr(db)}")
    # chat template ยังทำงาน?
    try:
        msgs = [{"role": "user", "content": "สวัสดีครับ"}]
        ct = new_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        print(f"  ✅ chat template ใช้ได้ ({len(ct)} chars)")
    except Exception as e:
        ok = False
        print(f"  ❌ chat template พัง: {e}")
    print("ผล roundtrip:", "ผ่านทั้งหมด ✅" if ok else "มีปัญหา ❌")

    # save report
    report = {
        "mode": "aggressive" if args.aggressive else "conservative",
        "target": args.target if args.aggressive else None,
        "n_base_old": n_base, "n_special": len(added),
        "n_base_new": K, "vocab_total_old": n_base + len(added),
        "vocab_total_new": new_total, "merges_old": len(merges), "merges_new": len(new_merges),
        "base_dropped": n_base - K, "base_dropped_pct": round((n_base - K) / n_base * 100, 1),
        "roundtrip_ok": ok, "closure_extra": added_closure,
    }
    json.dump(report, open(os.path.join(args.out, "vocab_trim_report.json"), "w"), indent=2)
    print(f"\nรายงาน → {os.path.join(args.out, 'vocab_trim_report.json')}")

    if args.with_model:
        trim_model(args.out, kept_base, n_base)


def trim_model(out_dir, kept_base, n_base):
    import torch
    from transformers import AutoModelForCausalLM
    print("\n=== (B) โหลดโมเดล + ตัด embedding ===")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    tie = getattr(model.config, "tie_word_embeddings", True)
    print(f"tie_word_embeddings = {tie}")
    # keep ids สำหรับ embedding: base kept (เรียง) + specials (128000..128255)
    keep_emb = list(kept_base) + list(range(n_base, n_base + 256))
    idx = torch.tensor(keep_emb, dtype=torch.long)

    p0 = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight.data
    new_emb = emb[idx].clone()
    model.resize_token_embeddings(len(keep_emb))
    model.get_input_embeddings().weight.data.copy_(new_emb)
    out = model.get_output_embeddings()
    if out is not None and not tie:
        old_lm = out.weight.data  # ก่อน resize ถูกแทนแล้ว — โหลดใหม่ถ้าจำเป็น
        out.weight.data.copy_(new_emb)  # tied → เหมือนกัน (ถ้าไม่ tie ควรดึง lm_head เดิมแยก)
    model.config.vocab_size = len(keep_emb)
    # อัปเดต special token ids (bos/eos/pad) เป็น id ใหม่ ทั้งใน config และ generation_config
    K = len(kept_base)
    def remap(old):
        return K + (old - n_base) if isinstance(old, int) and old >= n_base else old
    for cfg in (model.config, getattr(model, "generation_config", None)):
        if cfg is None:
            continue
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
            old = getattr(cfg, attr, None)
            if isinstance(old, list):
                setattr(cfg, attr, [remap(x) for x in old])
            elif old is not None:
                setattr(cfg, attr, remap(old))

    p1 = sum(p.numel() for p in model.parameters())
    print(f"param: {p0/1e9:.4f}B → {p1/1e9:.4f}B (ลด {(p0-p1)/1e6:.1f}M)")
    model.save_pretrained(out_dir, safe_serialization=True)
    print(f"เซฟโมเดล vocab-trimmed → {out_dir}")


if __name__ == "__main__":
    main()
