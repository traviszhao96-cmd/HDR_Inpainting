# 桌面端（Windows, RTX 4070 TiS）搭建与运行

本仓库已把 `libultrahdr` 作为 **git 子模块** 纳入，因此台式机 `git clone` 后即可拿到并构建项目所需的
`ultrahdr_app`（Ultra HDR 编解码）。本机（Mac, M4）负责开发/前端/轻活；**台式机作为"单一 GPU 服务"**
（SAM2 分割 + 内容生成 + 重建 HDR）。

---

## 1. 前提（Windows）

- Git
- Python 3.10+（独立 venv）
- CMake ≥ 3.16
- C++ 编译器：**Visual Studio 2019/2022 Build Tools（MSVC）** 或 MinGW
- 显卡驱动对应 CUDA（4070 TiS 建议 CUDA 12.x）

## 2. 克隆 + 拉子模块

```bat
git clone git@github.com:traviszhao96-cmd/HDR_Inpainting.git
cd HDR_Inpainting
git submodule update --init
```
> 说明：`libultrahdr` 子模块指向 google/libultrahdr @ 93c2349；（含上游测试图，首次拉取约几百 MB）。
> 只想快速构建可改用浅拉：`git submodule update --init --depth 1`（更小，但可能非锁定提交）。

## 3. 构建 ultrahdr_app

```bat
cd libultrahdr
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target ultrahdr_app
REM 产物：libultrahdr/build/Release/ultrahdr_app.exe
```
复制到根目录 `build/` 下（管线默认在 `build/ultrahdr_app` 找）：
```bat
mkdir ..\build 2>nul
copy build\Release\ultrahdr_app.exe ..\build\ultrahdr_app.exe
```
> 注：管线里 `ULTRAHDR_APP = build/ultrahdr_app`；Windows 下是 `ultrahdr_app.exe`。
> 若你直接改 `hdr_ai_erase.py` / `seg_ui/backend/app.py` 里的 `ULTRAHDR_APP` 指向该 exe 更省事。

## 4. Python 环境

```bat
python -m venv .venv && .venv\Scripts\activate
pip install numpy pillow opencv-python-headless requests
REM 若要运行 seg_ui 后端（SAM2 分割 + /erase）：
pip install torch torchvision transformers fastapi uvicorn python-multipart
```

## 5. 启动（本机监听 + Tailscale 私有访问）

```bat
REM 后台常驻：python -m uvicorn backend.app:app --host 127.0.0.1 --port 7860
REM 前端（seg_ui）：
npm install && npm run dev
```
- 安全：后端/前端都绑 `127.0.0.1`，**不要** `--listen 0.0.0.0` 裸暴露。
- 远程访问用 **Tailscale**：`tailscale serve --bg 7860`（把本机 7860 暴露给 tailnet，走加密通道，不开放公网端口）。
- `seg_ui` 默认从 `http://127.0.0.1:7860` 拉后端；外部用 Tailscale 地址访问 `http://<台式机>.ts.net:7860`。

## 6. 内容引擎

- seg_ui 的 `/erase` 支持 **OpenCV 本地（免费，验证通路）** 或 **gpt-image-2（云端，需 API Key）**。
- 台式机上跑 `hdr_ai_erase.py` 也一样：`python hdr_ai_erase.py --inpainter opencv --mask mask.png`。

## 7. 泄漏防护（重要）

- `.qq_smtp.env`、`*.env`、`*.key`、`*token*` 均被 `.gitignore` 忽略——**不要**把任何 API Key / 授权码提交。
- `output/`、`build/`、`*.raw` 等大体积产物不入库。
- ComfyUI 无鉴权、节点可执行代码：永远绑 `127.0.0.1`，远程走 Tailscale，别用 `0.0.0.0`/UPnP/明文端口暴露。
