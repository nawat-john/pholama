"""Phase 7 — eval in-domain (B3) + OOD regression probe

in-domain: รัน test.jsonl ผ่าน B3 (RAG vs no-RAG) → Gemini เป็น judge (correct/partial/wrong)
OOD: ถามโค้ด/คณิต → ดูว่าตกจริง (พิสูจน์ specialization)
ต้อง start llama-server (Q4_K_M) ก่อน

รัน:  .venv\\Scripts\\python.exe scripts\\phase7_eval.py --n 40
"""
import argparse, json, os, time, re
from _bootstrap import bootstrap
bootstrap()
from phase7_rag import ask
import google.generativeai as genai

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "artifacts", "eval")

JUDGE_PROMPT = """คุณเป็นผู้ตรวจคำตอบ เทียบ "คำตอบของระบบ" กับ "คำตอบอ้างอิง" สำหรับคำถามเดียวกัน
ให้คะแนนความถูกต้องเชิงข้อเท็จจริง:
- correct = ตรงกับอ้างอิงในสาระสำคัญ
- partial = ถูกบางส่วน/กว้างไป/ขาดรายละเอียดสำคัญ
- wrong = ผิด/ขัดแย้ง/ไม่ตอบ

คำถาม: {q}
คำตอบอ้างอิง: {ref}
คำตอบของระบบ: {ans}

ตอบเป็น JSON: {{"verdict": "correct|partial|wrong", "reason": "สั้นๆ"}}"""

OOD_PROBES = [
    "Write a Python function to compute the nth Fibonacci number.",
    "What is 17 * 23? Show steps.",
    "Solve for x: 2x + 5 = 13",
    "เขียนโค้ด Python สำหรับเรียงลำดับ list จากน้อยไปมาก",
]


def make_judge(models):
    genai.configure(api_key=os.environ["GEMINI_API_KEY"], transport="rest")
    chain = [m.strip() for m in models.split(",")]
    state = {"i": 0}
    def judge(q, ref, ans):
        for _ in range(len(chain)):
            try:
                m = genai.GenerativeModel(chain[state["i"]],
                        generation_config={"response_mime_type": "application/json", "temperature": 0})
                r = m.generate_content(JUDGE_PROMPT.format(q=q, ref=ref, ans=ans),
                                       request_options={"timeout": 60})
                return json.loads(r.text).get("verdict", "wrong")
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    state["i"] = (state["i"] + 1) % len(chain); time.sleep(2); continue
                return "error"
        return "quota_out"
    return judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--judge-models",
                    default="gemini-2.5-flash-lite,gemini-flash-lite-latest,gemini-flash-latest")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    test = [json.loads(l) for l in open(os.path.join(DATA, "test.jsonl"), encoding="utf-8")][:args.n]
    judge = make_judge(args.judge_models)

    print(f"=== in-domain eval (B3 Q4_K_M, n={len(test)}) ===")
    rows = []
    tally = {"rag": {"correct": 0, "partial": 0, "wrong": 0, "error": 0, "quota_out": 0},
             "norag": {"correct": 0, "partial": 0, "wrong": 0, "error": 0, "quota_out": 0}}
    for i, ex in enumerate(test):
        q, ref = ex["question"], ex["answer"]
        a_rag, _ = ask(q, k=4, use_rag=True)
        a_no, _ = ask(q, use_rag=False)
        v_rag = judge(q, ref, a_rag); time.sleep(1.5)
        v_no = judge(q, ref, a_no); time.sleep(1.5)
        tally["rag"][v_rag] = tally["rag"].get(v_rag, 0) + 1
        tally["norag"][v_no] = tally["norag"].get(v_no, 0) + 1
        rows.append({"q": q, "ref": ref, "ans_rag": a_rag, "ans_norag": a_no,
                     "verdict_rag": v_rag, "verdict_norag": v_no})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test)} | RAG {tally['rag']} | noRAG {tally['norag']}")

    def acc(t):
        n = sum(t.values()) or 1
        return round((t["correct"] + 0.5 * t["partial"]) / n * 100, 1)
    summary = {"n": len(test), "rag": tally, "norag": tally,
               "score_rag": acc(tally["rag"]), "score_norag": acc(tally["norag"])}
    summary["rag"], summary["norag"] = tally["rag"], tally["norag"]
    print(f"\nin-domain score (correct + 0.5*partial): RAG={summary['score_rag']}% | noRAG={summary['score_norag']}%")

    # OOD probe
    print("\n=== out-of-domain probe (B3, ควรตก = specialization สำเร็จ) ===")
    ood = []
    for p in OOD_PROBES:
        a, _ = ask(p, use_rag=False, max_tokens=120)
        ood.append({"q": p, "ans": a})
        print(f"\nQ: {p}\nA: {a[:160]}")

    json.dump({"in_domain": summary, "rows": rows, "ood": ood},
              open(os.path.join(OUT, "eval_b3.json"), "w"), ensure_ascii=False, indent=2)
    print(f"\n✅ → {os.path.join(OUT, 'eval_b3.json')}")


if __name__ == "__main__":
    main()
