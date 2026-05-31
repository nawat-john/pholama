"""Phase 2 — ขั้น 3: filter + dedup + split train/val/test

อ่าน data/qa_raw.jsonl แล้ว:
1. heuristic filter — ตัดที่สั้นเกิน/ยาวเกิน, คำถาม self-referential, ตอบ "ไม่มีข้อมูล"
2. dedup — ตามคำถาม normalize
3. split 80/10/10 (seed=42) + ✅ leak check (ห้าม test ซ้ำใน train/val)
4. เขียน train/val/test.jsonl + dataset_card.md

หมายเหตุ: นี่คือการคัด "อัตโนมัติ" — แผนแนะนำให้คัดด้วยมืออีกชั้น
เปิด data/qa_raw.jsonl ใน editor ลบที่ไม่ดี แล้วรัน --input data/qa_clean.jsonl ทับได้

รัน:  .venv\Scripts\python.exe scripts\phase2_split.py
"""
import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter

from _bootstrap import bootstrap
bootstrap()

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")

# คำถามที่อ้างถึงตัวเอกสาร = ไม่ standalone ตัดทิ้ง
SELF_REF = re.compile(r"(จากเอกสาร|ตามข้อความ|ข้างต้น|ในข้อความนี้|จากบทความ|above text|this document|the document|the passage)", re.I)
# คำตอบที่บอกว่าไม่มีข้อมูล = ไร้ค่า
NO_INFO = re.compile(r"(ไม่มีข้อมูล|ไม่ได้ระบุ|ไม่ปรากฏ|not (specified|mentioned|provided|available)|no information)", re.I)


def norm_q(q):
    return re.sub(r"\s+", "", q).lower()


def keep(ex):
    q, a = ex.get("question", ""), ex.get("answer", "")
    if not (8 <= len(q) <= 300):
        return False
    if not (3 <= len(a) <= 1200):
        return False
    if SELF_REF.search(q) or SELF_REF.search(a):
        return False
    if NO_INFO.search(a):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(DATA_DIR, "qa_raw.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    n_raw = len(rows)

    filtered = [r for r in rows if keep(r)]
    n_filtered = len(filtered)

    seen, clean = set(), []
    for r in filtered:
        h = hashlib.md5(norm_q(r["question"]).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            clean.append(r)
    n_clean = len(clean)

    random.seed(args.seed)
    random.shuffle(clean)
    n = n_clean
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    splits = {
        "train": clean[:n_train],
        "val": clean[n_train:n_train + n_val],
        "test": clean[n_train + n_val:],
    }

    for name, data in splits.items():
        fp = os.path.join(DATA_DIR, f"{name}.jsonl")
        with open(fp, "w", encoding="utf-8") as f:
            for ex in data:
                # ตัด field ภายในออกจาก deliverable
                f.write(json.dumps({k: v for k, v in ex.items() if not k.startswith("_")},
                                   ensure_ascii=False) + "\n")
        print(f"{name}: {len(data)}")

    # ✅ leak check
    test_q = {norm_q(e["question"]) for e in splits["test"]}
    train_val_q = {norm_q(e["question"]) for e in splits["train"] + splits["val"]}
    leak = test_q & train_val_q
    assert not leak, f"❌ พบ test รั่ว {len(leak)} ข้อ!"
    print("✅ ไม่มี test รั่วไป train/val")

    lang = Counter(e.get("lang", "?") for e in clean)
    style = Counter(e.get("style", "?") for e in clean)
    src = Counter(e.get("source", "?") for e in clean)

    card = f"""# Dataset Card — reg.chula Q&A (v1)

**โดเมน:** สำนักงานการทะเบียน จุฬาลงกรณ์มหาวิทยาลัย (https://www.reg.chula.ac.th/th/)
**ภาษา:** ไทย / อังกฤษ / ปนกัน — ใช้คำว่า "นิสิต"

## ที่มา
สังเคราะห์ด้วย Gemini (multi-model fallback: gemini-2.5-flash-lite / flash-lite-latest /
flash-latest / 2.5-flash — สลับเมื่อโควต้าฟรีรายวันหมด) จากข้อความที่ crawl จากเว็บ
reg.chula.ac.th (หน้า HTML + PDF ระเบียบ/ประกาศ) คำตอบ grounded ในเนื้อหาเว็บเท่านั้น

## จำนวน
- raw ที่ gen ได้: {n_raw}
- หลัง heuristic filter: {n_filtered}
- หลัง dedup (clean): {n_clean}
- train / val / test: {len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])} (80/10/10, seed={args.seed})

## การกระจาย (clean)
- lang: {dict(lang)}
- style: {dict(style)}
- จำนวน source page/pdf: {len(src)}

## การใช้งานในไปป์ไลน์
- calibration (Phase 4) + recovery training (Phase 5): ใช้ **train** เท่านั้น
- evaluation (Phase 1/7): ใช้ **test**
- ⚠️ test แยกเด็ดขาด ห้ามรั่วเข้า calibration/training (ตรวจแล้วผ่าน)

## ข้อจำกัด / ที่ควรทำต่อ
- เป็นการคัด "อัตโนมัติ" — แผนแนะนำคัดด้วยมืออีกชั้น (เปิด qa_raw.jsonl ลบที่ผิด/กำกวม)
- เนื้อหาวันเวลา (ปีการศึกษา 2568/2569) เป็น snapshot ณ วันที่ crawl — Phase 7 ใช้ RAG ดึงของจริงแทน
"""
    with open(os.path.join(DATA_DIR, "dataset_card.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print("\n" + card)


if __name__ == "__main__":
    main()
