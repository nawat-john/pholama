# =====================================================================
# setup_env.ps1 — สร้าง Python environment ทั้งหมดบนไดรฟ์ D (ไม่แตะ C)
# รันครั้งเดียว:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .tools\setup_env.ps1
# =====================================================================
$ErrorActionPreference = "Stop"
$Root  = "D:\Code\pholama"
$uv    = "$Root\.tools\uv.exe"
$py    = "$Root\.venv\Scripts\python.exe"

# --- บังคับ cache/temp/Python install ทุกตัวลง D (ห้ามไปลง C ที่เหลือน้อย) ---
$env:UV_CACHE_DIR          = "$Root\.cache\uv"
$env:UV_PYTHON_INSTALL_DIR = "$Root\.python"
$env:TMP  = "$Root\.cache\tmp"
$env:TEMP = "$Root\.cache\tmp"
$env:PIP_CACHE_DIR = "$Root\.cache\pip"
$env:HF_HOME       = "$Root\.cache\hf"
New-Item -ItemType Directory -Force -Path `
  $env:UV_CACHE_DIR,$env:UV_PYTHON_INSTALL_DIR,$env:TMP,$env:PIP_CACHE_DIR,$env:HF_HOME | Out-Null

Write-Host "[1/4] สร้าง venv (Python 3.11 standalone บน D) ..." -ForegroundColor Cyan
& $uv venv "$Root\.venv" --python 3.11

Write-Host "[2/4] ติดตั้ง PyTorch (CUDA cu124) ..." -ForegroundColor Cyan
& $uv pip install --python $py torch --index-url https://download.pytorch.org/whl/cu124

Write-Host "[3/4] ติดตั้ง deps ที่เหลือ ..." -ForegroundColor Cyan
& $uv pip install --python $py -r "$Root\requirements.txt"

Write-Host "[4/4] ตรวจ GPU ..." -ForegroundColor Cyan
& $py -c "import torch; print('torch', torch.__version__); ok=torch.cuda.is_available(); print('CUDA available:', ok); print('GPU:', torch.cuda.get_device_name(0) if ok else 'CPU only'); print('VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9,2) if ok else 0)"

Write-Host "`nเสร็จ! เปิด env ต่อ session ด้วย:  . .tools\activate.ps1" -ForegroundColor Green
