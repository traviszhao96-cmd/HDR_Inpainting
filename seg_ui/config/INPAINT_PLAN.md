# 专用消除模型验证路线

## 当前状态

- 页面仍使用现有 Klein 内容引擎，尚未切到 FLUX Fill。
- 独立实验 `backend/run_mask_probe.py` 已验证 mask 进入 Klein 采样；
  对照全黑 mask，两名人物可移除，但色块接缝和未选中的阴影仍存在。
- 2026-09-14：Fill 与 ObjectRemovalFluxFill v2 均已下载，文件大小及
  SHA256 重算通过；校验信息见 `flux_fill_download.json` 和下文。
- GGUF 节点安装、服务重启及首次 Fill / Fill + LoRA 推理已完成。
- 仍未切换网页默认引擎，首次测试仅验证 SDR，不含 gainmap/HDR 重建。

## LoRA 固定版本

- 官方仓库：https://huggingface.co/lrzjason/ObjectRemovalFluxFill
- revision: `867f2e6adb15232efd8d07bbb289e0f9f7551791`
- 文件：`removal_timestep_alpha-2-1740.safetensors`
- 字节数：`89746016`
- SHA256：`9a51336bca7c6cf85a611209c4b01977a55c2add16ec4d0de2d3d7a2e00aa0fd`
- 目标：`ComfyUI/models/loras/`。
- 适配 FLUX.1 Fill dev，不是 Klein；作者提示大面积蒙版存在局限。
- 量化底模搭配 LoRA 的实际加载兼容性和内存峰值还需测试。

## 下一步验收

1. 两个权重先比对大小及 SHA256；未验证前保留 `.part`。
2. 检查、安装 ComfyUI-GGUF，空闲时重启并确认节点可用。
3. 使用 Fill 专用条件工作流，不复用 Klein 的 ReferenceLatent 工作流。
4. 同一张 SDR、mask、随机种子及采样参数，对比 Fill 与 Fill + LoRA。
5. 保存原始生成结果、实际工作流、种子、时间及显存/内存信息。
6. 检查是否彻底消除、是否重新生成对象、边缘色差和阴影残留。
7. SDR 达标后再验证 gainmap/HDR，最后接入页面。

测试照片、蒙版、生成结果和私有 API Key 仅保留在本机，不随 Git 同步。
新机器需自行放置测试图并准备凭据。模型许可沿用各自上游要求；不能
将研究验证通过视为已取得商业部署授权。

## 2026-09-14 首次 A/B 实测

- ComfyUI-GGUF commit：`6ea2651e7df66d7585f6ffee804b20e92fb38b8a`。
- 远端安装回报：gguf 0.19.0；torch/CUDA 未升级，服务保持 127.0.0.1:8188。
- 主模型：FLUX.1 Fill dev Q4_K_S；T5 FP8 + CLIP-L；FLUX VAE BF16。
- 输入：教堂双人样张，工作尺寸 512×768，蒙版在工作尺寸外扩 16 像素。
- seed 20260911，20 steps，Euler / normal，CFG 1，FluxGuidance 30。
- InpaintModelConditioning 同时输入原图与蒙版；noise_mask=true。
- LoRA strength 0 与 1 的两组；除此之外 prompt、图、mask、seed、采样参数相同。
- 原版 Fill：62.48 秒（含首次加载/网络等待），原目标被替换为两名新人，消除失败。
- Fill + LoRA：45.65 秒，两名行人被移除，但补出了远处坐着的人，阴影仍在。
- 时间不能直接用于比较 LoRA 速度，第二组复用了缓存；未测内存峰值。
- 本次仅一个 seed、一个场景，不代表整个图集通过。
- 本地完整工作流/原始结果目录：`test_runs/20260914-104919-mask-probe-ae8a99/`。

复现命令（从仓库根目录，需已有该本地样张结果）：

```sh
seg_ui/.venv/bin/python seg_ui/backend/run_mask_probe.py \
  --engine fill --url https://win-hm9ig3vhnaa.tailfff622.ts.net \
  --result-dir seg_ui/results/5a678dbc579c4aaeb7e5b672208c7067 \
  --mask-grow 16 \
  --prompt 'An empty stone floor with continuous paving lines and soft colored light from stained glass windows, a stone wall and seated visitors in the background, photorealistic, matching the surrounding architecture and lighting.'
```

后续先补全阴影范围并测试更多固定样张；对“生成额外背景人物”单独评估
提示词及采样设置，再决定接入网页。不要把这次实验称为 HDR 消除已修复。
