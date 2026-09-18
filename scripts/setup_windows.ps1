# 桌面端（Windows）一键构建 + 运行准备脚本
# 用法：在仓库根目录打开 PowerShell，执行  .\scripts\setup_windows.ps1
# 默认构建 libultrahdr，并把 ultrahdr_app.exe 放到根目录 build/ 下。
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "==> 1. 拉取 libultrahdr 子模块" -ForegroundColor Cyan
git submodule update --init

Write-Host "==> 2. 构建 ultrahdr_app (CMake/MSVC)" -ForegroundColor Cyan
Set-Location (Join-Path $Root "libultrahdr")
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUHDR_WRITE_ISO=ON -DUHDR_WRITE_XMP=ON
cmake --build build --config Release --target ultrahdr_app

# 定位产物并复制到根目录 build/
$exe = Get-ChildItem -Path (Join-Path $Root "libultrahdr\build") -Recurse -Filter "ultrahdr_app.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $exe) { throw "未找到 ultrahdr_app.exe，请检查构建输出目录" }
New-Item -ItemType Directory -Force -Path (Join-Path $Root "build") | Out-Null
Copy-Item $exe.FullName (Join-Path $Root "build\ultrahdr_app.exe") -Force
Write-Host "    已复制 => build\ultrahdr_app.exe" -ForegroundColor Green

Write-Host "==> 3. Python 虚拟环境 + 依赖" -ForegroundColor Cyan
Set-Location $Root
if (-not (Test-Path ".venv")) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pillow opencv-python-headless requests
pip install torch torchvision transformers fastapi uvicorn python-multipart

Write-Host "==> 4. 前端依赖 (seg_ui)" -ForegroundColor Cyan
Set-Location (Join-Path $Root "seg_ui")
if (-not (Test-Path "node_modules")) { npm install }

Write-Host ""
Write-Host "完成。下一步：" -ForegroundColor Green
Write-Host "  cd seg_ui && npm run dev                    # 前端"
Write-Host "  python -m uvicorn backend.app:app --host 127.0.0.1 --port 7860   # 后端(SAM2+/erase)"
Write-Host "  远程访问用 Tailscale: tailscale serve --bg 7860                # 只暴露给 tailnet"
