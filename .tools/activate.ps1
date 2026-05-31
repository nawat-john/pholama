# activate.ps1 — เปิด env ต่อ 1 session (ตั้ง cache ลง D + activate venv)
# ใช้:  . .tools\activate.ps1     (มีจุดนำหน้า เพื่อให้ env var ติดกับ shell ปัจจุบัน)
$Root = "D:\Code\pholama"
$env:HF_HOME       = "$Root\.cache\hf"      # โมเดล HuggingFace ลง D (~6.4GB/ตัว)
$env:UV_CACHE_DIR  = "$Root\.cache\uv"
$env:PIP_CACHE_DIR = "$Root\.cache\pip"
$env:TMP  = "$Root\.cache\tmp"
$env:TEMP = "$Root\.cache\tmp"
& "$Root\.venv\Scripts\Activate.ps1"
Write-Host "env active — Python/caches/HF_HOME อยู่บน D ทั้งหมด" -ForegroundColor Green
