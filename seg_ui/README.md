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

## 当前交互与测试流程

网页默认使用 `comfyui-flux-fill`：FLUX.1 Fill-dev Q4_K_S +
ObjectRemovalFluxFill v2，图像与蒙版都进入专用 inpaint 条件。
圈选确认后执行 SDR 消除；HDR 输入随后用单通道平滑插值重建 gainmap
并编码 UHDR。当前网页不运行 gainmap 生成模型或 ControlNet。
进度阶段是时间估算，实际较大的图像可能显著超过 30 秒。

从项目根目录运行 `seg_ui/.venv/bin/python seg_ui/backend/test_fill_pipeline.py`
可通过与页面相同的 `/erase` 接口测试五张非教堂样张。它保留已保存的
蒙版，使用当前无人背景提示词和 Fill 引擎，结果存入 `seg_ui/test_runs/`。
提示词不保证拒绝所有新增人物，结果仍需目视检查。
# 循环编辑与本地作品集

消除成功后自动以结果作为新底图，清空旧蒙版后可继续圈选。编辑器展示 SDR 预览，但下一轮提交完整 UHDR 文件（普通 SDR 则提交 PNG），并清除原测试样张标识，避免重新编辑旧图。

点击「保存当前作品」直接保存到 `seg_ui/portfolio/<job_id>/`，不经过浏览器下载：`work.jpg` 为完整 UHDR，`work.png` 为普通 SDR，`preview.png` 为预览。重复保存同一个结果不会产生重复作品。点击作品可重新载入编辑，后续结果不会覆盖已保存作品。每轮原始结果仍保留在 `seg_ui/results/`。
# HDR 导出格式（2026-09-18）

正式消除输出使用手机实测通过的 Display P3 JPEG HDR：SDR/HDR 工作数组
直接按 P3 编码（`-c 1 -C 1`），不额外执行 709→P3 矩阵；单通道 gainmap。
这是当前工作流确认的输出策略，不表示任意 sRGB 文件都可以直接重标 P3，
也不能仅凭视觉更饱和就证明此前发生了重复转换。

编码器必须同时写入 ISO 21496-1 和 Adobe/Google XMP gainmap 元数据：

```sh
cmake -S libultrahdr -B build -DUHDR_WRITE_ISO=ON -DUHDR_WRITE_XMP=ON
cmake --build build -j 6
```

以上命令从仓库根目录运行。原图 EXIF 通过编码器 `-x` 写入，MPF 索引、
主图长度与 gainmap 长度由编码器生成；不再在编码后手工插入 APP 段。
作品集保存直接复制结果 JPEG，连续编辑重新解码该 JPEG；均沿用相同输出。
历史结果不会自动重写。独立 `convert_uhdr_display_p3.py` 仅用于旧文件对照实验。

本地 libultrahdr 源码还修复了两处问题，同步构建时需一并保留子模块改动：
`examples/ultrahdr_app.cpp` 的 EXIF 输入长度未赋值，以及
`lib/src/dsp/arm/gainmapmath_neon.cpp` 中 P3/BT.2020 的 RGB→YUV 系数分支颠倒，
同时将 P3 JPEG 的 YCbCr 系数与标量实现的 BT.601 系数对齐。
离线回归检查（不调用生成模型）：
`seg_ui/.venv/bin/python -m unittest seg_ui.backend.test_hdr_export`。
