"""Phase 7 — eval B1 (โมเดลเดิม Typhoon2 3B) in-domain ด้วยชุด+judge เดียวกับ B3
เพื่อเทียบคุณภาพ เดิม vs ใหม่ (ความคุ้มค่า)

รัน:  .venv\\Scripts\\python.exe scripts\\phase7_eval_b1.py --n 40
"""
import argparse, json, os, time
from _bootstrap import bootstrap
bootstrap()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from phase7_rag import retrieve, SYS_RAG
from phase7_eval import make_judge

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "artifacts", "eval")
B1 = "scb10x/llama3.2-typhoon2-3b-instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--judge-models",
                    default="gemini-2.5-flash-lite,gemini-flash-lite-latest,gemini-flash-latest")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(B1)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(B1, quantization_config=bnb,
                                                 dtype=torch.bfloat16, device_map="cuda").eval()

    @torch.no_grad()
    def gen(q, use_rag=True, max_new=200):
        if use_rag:
            ctx = "\n\n".join(f"[{i+1}] {c['text'][:600]}" for i, c in enumerate(retrieve(q, 4)))
            msgs = [{"role": "system", "content": SYS_RAG},
                    {"role": "user", "content": f"บริบท:\n{ctx}\n\nคำถาม: {q}"}]
        else:
            msgs = [{"role": "user", "content": q}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    test = [json.loads(l) for l in open(os.path.join(DATA, "test.jsonl"), encoding="utf-8")][:args.n]
    judge = make_judge(args.judge_models)
    tally = {"rag": {}, "norag": {}}
    rows = []
    print(f"=== B1 in-domain (n={len(test)}) ===")
    for i, ex in enumerate(test):
        q, ref = ex["question"], ex["answer"]
        a_rag = gen(q, True); a_no = gen(q, False)
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
        return round((t.get("correct", 0) + 0.5 * t.get("partial", 0)) / n * 100, 1)
    out = {"model": "B1_original", "n": len(test), "rag": tally["rag"], "norag": tally["norag"],
           "score_rag": acc(tally["rag"]), "score_norag": acc(tally["norag"]), "rows": rows}
    print(f"\nB1 in-domain: RAG={out['score_rag']}% | noRAG={out['score_norag']}%")
    json.dump(out, open(os.path.join(OUT, "eval_b1.json"), "w"), ensure_ascii=False, indent=2)
    print(f"✅ → {os.path.join(OUT, 'eval_b1.json')}")


if __name__ == "__main__":
    main()
