"""Phase 4 — sweep หาจำนวน layer ที่ตัดได้พอดี (reuse BI เดิม)

อ่าน block_influence จาก artifacts/pruned_moderate/prune_report.json (ไม่ต้องคำนวณ BI ใหม่)
วน drop = [2,4,6] วัด ppl + probe → เลือก k มากสุดที่ ppl < THRESH และ generate coherent → เซฟ

รัน:  .venv\\Scripts\\python.exe scripts\\phase4_sweep.py
"""
import json, math, os, gc
from _bootstrap import bootstrap
bootstrap()
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(ROOT, "artifacts", "vocab_trimmed")
DATA = os.path.join(ROOT, "data")
ART = os.path.join(ROOT, "artifacts")

DROPS = [2, 4, 6]          # 8 รู้แล้วว่าพัง (ppl 236)
PPL_N = 80
MAX_LEN = 1024
PPL_THRESH = 20.0          # k มากสุดที่ ppl < นี้ = เลือก

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
val = [json.loads(l) for l in open(os.path.join(DATA, "val.jsonl"), encoding="utf-8")]
rep = json.load(open(os.path.join(ART, "pruned_moderate", "prune_report.json"), encoding="utf-8"))
bi = [rep["block_influence"][str(i)] for i in range(28)]


def chat(ex):
    return tok.apply_chat_template(
        [{"role": "user", "content": ex["question"]},
         {"role": "assistant", "content": ex["answer"]}], tokenize=False)


def load_full():
    m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to(device).eval()
    m.config.use_cache = False
    return m


@torch.no_grad()
def ppl(m, n=PPL_N):
    nll, ntok = 0.0, 0
    for ex in val[:n]:
        ids = tok(chat(ex), return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.to(device)
        if ids.shape[1] < 2:
            continue
        out = m(ids, labels=ids); t = ids.shape[1] - 1
        nll += out.loss.item() * t; ntok += t
    return math.exp(nll / max(ntok, 1))


def prune_layers(m, drop):
    keep = [i for i in range(m.config.num_hidden_layers) if i not in drop]
    m.model.layers = nn.ModuleList([m.model.layers[i] for i in keep])
    m.config.num_hidden_layers = len(keep)
    for new_i, layer in enumerate(m.model.layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.layer_idx = new_i
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = new_i
    return keep


@torch.no_grad()
def probe(m):
    ids = tok(tok.apply_chat_template(
        [{"role": "user", "content": "นิสิตลงทะเบียนเรียนได้ที่ไหน"}],
        add_generation_prompt=True, tokenize=False), return_tensors="pt").input_ids.to(device)
    g = m.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    txt = tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True)
    # coherent ถ้าไม่วนซ้ำหนัก (unique 4-gram ratio)
    toks = txt.split()
    uniq = len(set(txt[i:i+8] for i in range(0, max(len(txt)-8, 1)))) / max(len(txt)-8, 1)
    return txt[:120], uniq


order = sorted(range(28), key=lambda i: bi[i])
protected = {0, 27}
cand = [i for i in order if i not in protected]

results = []
for k in DROPS:
    m = load_full()
    drop = sorted(cand[:k])
    keep = prune_layers(m, drop)
    p = sum(pp.numel() for pp in m.parameters()) / 1e9
    pl = ppl(m)
    txt, uniq = probe(m)
    coherent = uniq > 0.5
    results.append({"k": k, "drop": drop, "params_B": round(p, 4),
                    "ppl": round(pl, 3), "coherent": coherent, "probe": txt})
    print(f"\n=== drop {k} (28→{28-k}) ===")
    print(f"  layers ตัด: {drop}")
    print(f"  param={p:.4f}B | ppl={pl:.3f} | coherent={coherent} (uniq={uniq:.2f})")
    print(f"  probe: {txt}")
    del m; gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

# เพิ่มผล k=8 ที่รู้แล้ว
results.append({"k": 8, "drop": [17, 18, 20, 21, 22, 23, 24, 25], "params_B": 2.1514,
                "ppl": rep["ppl_after_layer"], "coherent": False, "probe": "การศภ... (ขยะ)"})

print("\n\n=== สรุป sweep ===")
print(f"{'drop':>4} {'layers':>8} {'param':>8} {'ppl':>9} {'coherent':>9}")
for r in results:
    print(f"{r['k']:>4} {28-r['k']:>6}   {r['params_B']:>7.3f}B {r['ppl']:>9.2f} {str(r['coherent']):>9}")

# เลือก k มากสุดที่ coherent และ ppl < THRESH
ok = [r for r in results if r["coherent"] and r["ppl"] < PPL_THRESH]
best = max(ok, key=lambda r: r["k"]) if ok else min(results, key=lambda r: r["ppl"])
print(f"\n→ เลือก drop {best['k']} (28→{28-best['k']}) ppl={best['ppl']} param={best['params_B']}B")

# เซฟตัวเลือก
m = load_full()
keep = prune_layers(m, best["drop"])
m.config.use_cache = True
save_dir = os.path.join(ART, "pruned_moderate")
m.save_pretrained(save_dir, safe_serialization=True)
tok.save_pretrained(save_dir)
out = {"scenario": "moderate", "kept_layers": keep, "dropped_layers": best["drop"],
       "num_hidden_layers": m.config.num_hidden_layers, "intermediate_size": m.config.intermediate_size,
       "params_B": best["params_B"], "ppl_base": rep["ppl_base"], "ppl_after_layer": best["ppl"],
       "ppl_after_ffn": None, "sweep": [{kk: r[kk] for kk in ("k", "drop", "params_B", "ppl", "coherent")} for r in results],
       "block_influence": rep["block_influence"]}
json.dump(out, open(os.path.join(save_dir, "prune_report.json"), "w"), ensure_ascii=False, indent=2)
print(f"✅ เซฟ → {save_dir} (drop {best['k']})")
