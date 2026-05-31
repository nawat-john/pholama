"""Phase 4 — Structured pruning (รัน local บน GPU RTX 3050 ผ่าน sysmem fallback)

มิเรอร์ลอจิกใน notebooks/04_pruning.ipynb แต่ใช้ path local + CLI
- ShortGPT Block-Influence → ตัด layer สำคัญน้อย (28→20)
- (--aggressive) FFN width prune 8192→6144 ด้วย activation importance
- วัด perplexity บน val ก่อน/หลัง

รัน:
  .venv\\Scripts\\python.exe scripts\\phase4_prune.py                 # Moderate (layer-only)
  .venv\\Scripts\\python.exe scripts\\phase4_prune.py --aggressive    # + FFN prune
  เพิ่ม --cpu เพื่อบังคับ CPU (ช้ากว่า ~9 เท่า)
"""
import argparse, json, math, os, gc
from _bootstrap import bootstrap
bootstrap()
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(ROOT, "artifacts", "vocab_trimmed")
DATA = os.path.join(ROOT, "data")
ART = os.path.join(ROOT, "artifacts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggressive", action="store_true", help="ตัด FFN width ด้วย (Aggressive)")
    ap.add_argument("--target-layers", type=int, default=20)
    ap.add_argument("--new-intermediate", type=int, default=6144)
    ap.add_argument("--n-calib", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--ppl-n", type=int, default=150)
    ap.add_argument("--protect-first-last", action="store_true", default=True)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"device={device}", f"| {torch.cuda.get_device_name(0)}" if device == "cuda" else "")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to(device).eval()
    model.config.use_cache = False
    N = model.config.num_hidden_layers
    print(f"layers={N} hidden={model.config.hidden_size} inter={model.config.intermediate_size} "
          f"| param={sum(p.numel() for p in model.parameters())/1e9:.4f}B")

    def chat(ex):
        return tok.apply_chat_template(
            [{"role": "user", "content": ex["question"]},
             {"role": "assistant", "content": ex["answer"]}], tokenize=False)

    calib = [json.loads(l) for l in open(os.path.join(DATA, "train.jsonl"), encoding="utf-8")][:args.n_calib]
    calib_texts = [chat(e) for e in calib]
    val = [json.loads(l) for l in open(os.path.join(DATA, "val.jsonl"), encoding="utf-8")]
    print(f"calib={len(calib_texts)} val={len(val)}")

    # ---- Block Influence (ShortGPT) ----
    @torch.no_grad()
    def block_influence(texts):
        score = torch.zeros(N, dtype=torch.float64); ntok = 0
        for k, t in enumerate(texts):
            ids = tok(t, return_tensors="pt", truncation=True, max_length=args.max_len).input_ids.to(device)
            hs = model(ids, output_hidden_states=True).hidden_states
            for i in range(N):
                a, b = hs[i][0].float(), hs[i + 1][0].float()
                cos = F.cosine_similarity(a, b, dim=-1).clamp(-1, 1)
                score[i] += (torch.arccos(cos) / math.pi).sum().item()
            ntok += ids.shape[1]
            if (k + 1) % 50 == 0:
                print(f"  BI {k+1}/{len(texts)}")
        return (score / max(ntok, 1)).tolist()

    @torch.no_grad()
    def ppl(samples):
        nll, ntok = 0.0, 0
        for ex in samples[:args.ppl_n]:
            ids = tok(chat(ex), return_tensors="pt", truncation=True, max_length=args.max_len).input_ids.to(device)
            if ids.shape[1] < 2:
                continue
            out = model(ids, labels=ids); t = ids.shape[1] - 1
            nll += out.loss.item() * t; ntok += t
        return math.exp(nll / max(ntok, 1))

    print("วัด Block Influence …")
    bi = block_influence(calib_texts)
    order = sorted(range(N), key=lambda i: bi[i])
    print("BI (สำคัญน้อย→มาก):", [(i, round(bi[i], 3)) for i in order])

    protected = {0, N - 1} if args.protect_first_last else set()
    cand = [i for i in order if i not in protected]
    n_drop = N - args.target_layers
    drop = sorted(cand[:n_drop])
    keep = [i for i in range(N) if i not in drop]
    print(f"ตัด {n_drop} layers: {drop} | เก็บ: {keep}")

    print("วัด perplexity ฐาน …")
    ppl_base = ppl(val); print(f"  ppl_base = {ppl_base:.3f}")

    # ---- ตัด layer + reassign layer_idx ----
    model.model.layers = nn.ModuleList([model.model.layers[i] for i in keep])
    model.config.num_hidden_layers = len(keep)
    for new_i, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.layer_idx = new_i
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = new_i
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    p_layer = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"หลังตัด layer: {model.config.num_hidden_layers} layers | param={p_layer:.4f}B")
    ppl_layer = ppl(val); print(f"  ppl_after_layer = {ppl_layer:.3f} (+{ppl_layer-ppl_base:.2f})")

    # ---- (Aggressive) FFN width prune ----
    p_ffn = None; ppl_ffn = None
    if args.aggressive:
        print("FFN width pruning …")
        inter = model.config.intermediate_size
        imp = [torch.zeros(inter, dtype=torch.float64, device=device) for _ in range(len(keep))]
        hooks = []
        def mk(i):
            def hook(mod, inp, out):
                imp[i] += inp[0].detach().abs().float().sum(dim=(0, 1)).double()
            return hook
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.down_proj.register_forward_hook(mk(i)))
        with torch.no_grad():
            for t in calib_texts:
                ids = tok(t, return_tensors="pt", truncation=True, max_length=args.max_len).input_ids.to(device)
                model(ids)
        for h in hooks:
            h.remove()
        h_size = model.config.hidden_size; k = args.new_intermediate
        for i, layer in enumerate(model.model.layers):
            idx = torch.topk(imp[i], k).indices.sort().values
            g, u, d = layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj
            ng = nn.Linear(h_size, k, bias=False).to(g.weight.device, g.weight.dtype)
            nu = nn.Linear(h_size, k, bias=False).to(u.weight.device, u.weight.dtype)
            nd = nn.Linear(k, h_size, bias=False).to(d.weight.device, d.weight.dtype)
            ng.weight.data.copy_(g.weight.data[idx, :])
            nu.weight.data.copy_(u.weight.data[idx, :])
            nd.weight.data.copy_(d.weight.data[:, idx])
            layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj = ng, nu, nd
        model.config.intermediate_size = k
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        p_ffn = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"FFN {inter}→{k} | param={p_ffn:.4f}B")
        ppl_ffn = ppl(val); print(f"  ppl_after_ffn = {ppl_ffn:.3f}")

    # ---- save ----
    scenario = "aggressive" if args.aggressive else "moderate"
    save_dir = os.path.join(ART, f"pruned_{scenario}")
    model.config.use_cache = True
    model.save_pretrained(save_dir, safe_serialization=True)
    tok.save_pretrained(save_dir)
    report = {
        "scenario": scenario, "kept_layers": keep, "dropped_layers": drop,
        "num_hidden_layers": model.config.num_hidden_layers,
        "intermediate_size": model.config.intermediate_size,
        "params_B": round(sum(p.numel() for p in model.parameters()) / 1e9, 4),
        "ppl_base": round(ppl_base, 3), "ppl_after_layer": round(ppl_layer, 3),
        "ppl_after_ffn": round(ppl_ffn, 3) if ppl_ffn else None,
        "block_influence": {int(i): round(bi[i], 4) for i in range(N)},
    }
    json.dump(report, open(os.path.join(save_dir, "prune_report.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"\n✅ เซฟ → {save_dir}")
    print(json.dumps({k: report[k] for k in
                      ("scenario", "params_B", "num_hidden_layers", "intermediate_size",
                       "ppl_base", "ppl_after_layer", "ppl_after_ffn")}, ensure_ascii=False, indent=2))

    # probe
    ids = tok(tok.apply_chat_template(
        [{"role": "user", "content": "นิสิตลงทะเบียนเรียนได้ที่ไหน"}],
        add_generation_prompt=True, tokenize=False), return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        g = model.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    print("probe:", tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True)[:200])


if __name__ == "__main__":
    main()
