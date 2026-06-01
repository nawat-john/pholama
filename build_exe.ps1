# build_exe.ps1 — สร้าง dist\pholama.exe จาก pholama_launcher.py
# รัน:  powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Py   = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Py)) { throw "ไม่พบ .venv python ที่ $Py" }

Write-Host "==> ตรวจ/ติดตั้ง pyinstaller" -ForegroundColor Cyan
& $Py -c "import PyInstaller" 2>$null
if (-not $?) { & $Py -m pip install pyinstaller }

Write-Host "==> build (onefile, console)" -ForegroundColor Cyan
& $Py -m PyInstaller --noconfirm --onefile --console `
    --name pholama `
    --distpath (Join-Path $Root "dist") `
    --workpath (Join-Path $Root ".cache\pyi-build") `
    --specpath (Join-Path $Root ".cache\pyi-build") `
    (Join-Path $Root "pholama_launcher.py")

$Exe = Join-Path $Root "dist\pholama.exe"
if (Test-Path $Exe) {
    Write-Host "`n✅ เสร็จ: $Exe" -ForegroundColor Green
    Write-Host 'ลองใช้:  .\dist\pholama.exe "นิสิตยื่นคำร้องลาพักการศึกษาอย่างไร"'
} else {
    throw "build ไม่สำเร็จ — ไม่พบ $Exe"
}
