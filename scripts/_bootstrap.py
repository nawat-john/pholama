"""Shared bootstrap for Phase 2 local scripts.

จัดการ 3 เรื่องที่เครื่อง Windows + corporate proxy ต้องใช้:
1. truststore — ใช้ Windows trust store แก้ปัญหา SSL ของ proxy ที่ MITM
2. โหลด .env (GEMINI_API_KEY) เข้า os.environ
3. บังคับ stdout เป็น UTF-8 (เลี่ยง cp874 crash ตอน print ภาษาไทย/emoji)
"""
import os
import sys


def bootstrap():
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception as e:  # pragma: no cover
        print("warn: truststore ไม่ทำงาน:", e, file=sys.stderr)

    # โหลด .env (รูปแบบ KEY=VALUE บรรทัดละตัว)
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    # stdout/stderr เป็น UTF-8 เสมอ
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
