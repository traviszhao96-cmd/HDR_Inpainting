# HDR 对象消除

在浏览器中圈选对象，由本地 SAM 2.1 生成全分辨率单通道蒙版，然后选择：

- **OpenCV 本地**：免费验证“分割 → 消除 → HDR 重建”的完整通路。
- **gpt-image-2 云端**：把圈选附近的 SDR 局部缩至最长边 1024 px 后付费调用 OpenAI；返回后只在蒙版内拼接，并在本机重建单通道 gain map 的 Ultra HDR。

原始 HDR、gain map 和蒙版外的完整画面不会发送给 OpenAI。

## Run

```bash
cd /Users/travis.zhao/ultrahdr/seg_ui
./run-local.sh
```

Open <http://localhost:3000>. The first segmentation downloads Meta's
`facebook/sam2.1-hiera-tiny` weights (about 156 MB); later runs use the local cache.

The backend automatically selects CUDA, Apple MPS, or CPU. Downloaded masks are full-resolution
grayscale PNG files with white as the edit region, compatible with `hdr_ai_erase.py --mask`.

## OpenAI 付费 API 模式

ChatGPT Plus 与 OpenAI API 是两套独立计费。`gpt-image-2` 不支持免费 API 层级，需要在 OpenAI API 平台单独添加付款方式或购买额度。

推荐在启动后端前设置环境变量，这样 Key 不会写入网页或仓库：

```bash
export OPENAI_API_KEY="sk-..."
cd /Users/travis.zhao/ultrahdr/seg_ui
./run-local.sh
```

也可以只在本机网页的密码框中临时填写；它只会提交给 `127.0.0.1:7860`，不会由前端保存。长期使用建议采用环境变量，并为这个项目单独创建 Project Key。

使用步骤：圈选对象 → `继续消除` → 内容引擎选 `gpt-image-2` → 检查提示词 → 开始验证。结果保存在 `seg_ui/results/<job_id>/`。

为了控制费用，云端模式固定采用：对象周围 ROI、最长边 1024 px、`quality=low`、单张输出。完整分辨率拼接与 Ultra HDR 重建均在本地完成。

## Separate development processes

```bash
# Terminal 1
. .venv/bin/activate
python -m uvicorn backend.app:app --host 127.0.0.1 --port 7860

# Terminal 2
npm run dev
```
