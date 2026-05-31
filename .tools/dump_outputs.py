import json
nb = json.load(open("notebooks/01_baseline.ipynb", encoding="utf-8"))
buf = []
ci = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    txt = ""
    for o in c.get("outputs", []):
        t = o.get("output_type")
        if t == "stream":
            txt += "".join(o.get("text", []))
        elif t in ("execute_result", "display_data"):
            txt += "".join(o.get("data", {}).get("text/plain", []))
        elif t == "error":
            txt += f"ERROR {o.get('ename')}: {o.get('evalue')}\n"
            txt += "\n".join(o.get("traceback", []))[-1500:]
    src = "".join(c["source"])
    buf.append(f"\n########## code cell #{ci} ##########")
    buf.append("--- SRC (first 3 lines) ---")
    buf.append("\n".join(src.splitlines()[:3]))
    buf.append("--- OUTPUT ---")
    buf.append(txt.rstrip() if txt.strip() else "(no output)")
    ci += 1
open(".cache/nb_outputs.txt", "w", encoding="utf-8").write("\n".join(buf))
print("ok")
