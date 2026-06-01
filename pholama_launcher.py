"""pholama_launcher.py — ตัวห่อบางๆ (stdlib ล้วน) สำหรับ compile เป็น pholama.exe

ไม่ทำงานหนักเอง แค่:
  1. หา project root (โฟลเดอร์ที่มี .venv\\Scripts\\python.exe)
  2. ตั้ง env ให้เหมือน .tools\\activate.ps1 (PYTHONUTF8, HF_HOME, cache บน D)
  3. เรียก  .venv\\Scripts\\python.exe app.py  <args ทั้งหมดที่ผู้ใช้ส่งมา>

แยกจาก app.py เพื่อให้ exe เล็ก (ไม่ต้อง bundle torch/transformers) — งานหนักรันใน venv
"""
import os
import sys

# ราก default ของเครื่องนี้ (fallback ถ้าหาแบบ relative ไม่เจอ)
DEFAULT_ROOT = r"D:\Code\pholama"


def find_root():
    # 1) ไล่ขึ้นจากตำแหน่ง exe/สคริปต์ หา .venv
    base = os.path.dirname(os.path.abspath(
        sys.executable if getattr(sys, "frozen", False) else __file__))
    d = base
    for _ in range(6):
        if os.path.exists(os.path.join(d, ".venv", "Scripts", "python.exe")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # 2) fallback ตำแหน่งที่รู้
    if os.path.exists(os.path.join(DEFAULT_ROOT, ".venv", "Scripts", "python.exe")):
        return DEFAULT_ROOT
    sys.exit("❌ หา .venv ไม่เจอ — วาง pholama.exe ไว้ในโฟลเดอร์โปรเจกต์ "
             "หรือแก้ DEFAULT_ROOT")


def main():
    root = find_root()
    py = os.path.join(root, ".venv", "Scripts", "python.exe")
    app = os.path.join(root, "app.py")

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"                          # กัน cp874 crash
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("HF_HOME", os.path.join(root, ".cache", "hf"))
    env.setdefault("PIP_CACHE_DIR", os.path.join(root, ".cache", "pip"))
    env.setdefault("UV_CACHE_DIR", os.path.join(root, ".cache", "uv"))
    tmp = os.path.join(root, ".cache", "tmp")
    if os.path.isdir(tmp):
        env["TMP"] = env["TEMP"] = tmp

    args = sys.argv[1:]
    if not args:
        print('ใช้:  pholama.exe "ข้อความคำถาม"', file=sys.stderr)
        print('ตัวอย่าง:  pholama.exe "นิสิตยื่นคำร้องลาพักการศึกษาอย่างไร"', file=sys.stderr)
        sys.exit(2)

    # ส่งต่อ stdio ตรงๆ; ใช้ os.execv ไม่ได้บน Windows (จะคืน control ทันที) → subprocess
    import subprocess
    sys.exit(subprocess.call([py, app, *args], cwd=root, env=env))


if __name__ == "__main__":
    main()
