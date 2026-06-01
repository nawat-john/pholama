"""app.py — entry-point ถาม-ตอบ RAG (รันด้วย .venv python)

รับคำถามเป็น argument แล้วตอบโดยอิงข้อมูลสำนักงานการทะเบียน จุฬาฯ
- ดูแล llama-server ให้เอง: ถ้ายังไม่รันบน 127.0.0.1:8080 จะ start ให้ (โหลด GGUF Q4_K_M)
  แล้วปล่อยค้างไว้ → ครั้งถัดไปตอบเร็ว
- retrieve ด้วย e5-small + faiss → ส่งเข้า llama-server (OpenAI-compatible)

ใช้ผ่าน pholama.exe (launcher) หรือเรียกตรง:
  .venv\\Scripts\\python.exe app.py "นิสิตยื่นคำร้องลาพักการศึกษาอย่างไร"
  .venv\\Scripts\\python.exe app.py --show-sources "ลงทะเบียนสายทำอย่างไร"
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

GGUF = os.path.join(ROOT, "artifacts", "gguf", "recovered-Q4_K_M.gguf")
LLAMA_BIN = os.path.join(ROOT, ".cache", "llama-bin")
SERVER_EXE = os.path.join(LLAMA_BIN, "llama-server.exe")
HOST, PORT = "127.0.0.1", 8080
SERVER_LOG = os.path.join(ROOT, "artifacts", "llama-server.log")


def _server_ready():
    """True ถ้า llama-server โหลดโมเดลเสร็จพร้อมรับงาน"""
    try:
        with urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_server(timeout=180):
    """start llama-server (detached) ถ้ายังไม่รัน แล้วรอจน /health ok"""
    if _server_ready():
        return
    if not os.path.exists(SERVER_EXE):
        sys.exit(f"❌ ไม่พบ llama-server: {SERVER_EXE}")
    if not os.path.exists(GGUF):
        sys.exit(f"❌ ไม่พบโมเดล GGUF: {GGUF}")

    print("⏳ กำลังเปิด llama-server (โหลดโมเดลครั้งแรก ~10-30 วิ)…", file=sys.stderr)
    os.makedirs(os.path.dirname(SERVER_LOG), exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED_PROCESS
    with open(SERVER_LOG, "ab") as log:
        subprocess.Popen(
            [SERVER_EXE, "-m", GGUF, "--host", HOST, "--port", str(PORT),
             "-c", "4096", "--no-warmup"],
            cwd=LLAMA_BIN,            # ให้หา ggml/*.dll ในโฟลเดอร์เดียวกันเจอ
            stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            creationflags=flags, close_fds=True,
        )
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _server_ready():
            print("✅ llama-server พร้อมแล้ว", file=sys.stderr)
            return
        time.sleep(1.0)
    sys.exit(f"❌ llama-server ไม่พร้อมใน {timeout}s — ดู log: {SERVER_LOG}")


def main():
    ap = argparse.ArgumentParser(description="ถาม-ตอบ RAG สำนักงานการทะเบียน จุฬาฯ")
    ap.add_argument("question", nargs="+", help="ข้อความคำถาม")
    ap.add_argument("--no-rag", action="store_true", help="ถามตรงๆ ไม่ดึงบริบท")
    ap.add_argument("-k", type=int, default=4, help="จำนวน chunk ที่ดึง (default 4)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--show-sources", action="store_true", help="แสดงแหล่งอ้างอิงที่ดึงมา")
    args = ap.parse_args()
    question = " ".join(args.question).strip()
    if not question:
        sys.exit("❌ กรุณาใส่ข้อความคำถาม")

    ensure_server()

    from phase7_rag import ask  # import หลัง bootstrap ใน module (truststore/utf-8)
    ans, ctx = ask(question, k=args.k, use_rag=not args.no_rag,
                   max_tokens=args.max_tokens, server=f"http://{HOST}:{PORT}")

    print(ans)
    if args.show_sources and ctx:
        print("\n— แหล่งอ้างอิง —", file=sys.stderr)
        for i, c in enumerate(ctx, 1):
            print(f"[{i}] ({c['score']:.2f}) {c.get('title', '')[:60]} | "
                  f"{c.get('source', '')[:70]}", file=sys.stderr)


if __name__ == "__main__":
    main()
