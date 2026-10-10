# HDR Mask Lab

中英双语项目说明、原理、模型与安装步骤已统一到仓库根目录：

**[README · 中文 / English](../README.md)**

The bilingual overview, architecture, model requirements, setup instructions, and
limitations are maintained in the root README linked above.

## 本地开发 / Local development

首次使用请按主 README 构建带补丁的 libultrahdr，并设置自己的 `COMFYUI_URL`。
For first-time setup, build the patched libultrahdr and configure your own
`COMFYUI_URL` as described in the root README.

```bash
# From this directory / 在本目录运行
./run-local.sh
```

前端 / UI: <http://localhost:3000> · 后端 / API: <http://127.0.0.1:7860>

已安装依赖后，可在两个终端分别启动；远程服务环境变量须设置在后端终端中。
After installing dependencies, run the processes separately if desired.
Set remote-service environment variables in the backend terminal.

```bash
# Terminal 1: backend / 后端
. .venv/bin/activate
python -m uvicorn backend.app:app --host 127.0.0.1 --port 7860

# Terminal 2: frontend / 前端
npm run dev
```

当前网页默认 FLUX Fill＋消除 LoRA、生成式 gainmap；OpenCV 和平滑插值为备选。
The UI defaults to FLUX Fill with removal LoRA and generative gainmap repair;
OpenCV and smooth interpolation are alternatives.
