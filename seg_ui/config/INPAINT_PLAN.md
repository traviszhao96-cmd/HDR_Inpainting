# 专用消除模型验证路线

## 当前状态

- 页面仍使用现有 Klein 内容引擎，尚未切到 FLUX Fill。
- 独立实验 `backend/run_mask_probe.py` 已验证 mask 进入 Klein 采样；
  对照全黑 mask，两名人物可移除，但色块接缝和未选中的阴影仍存在。
- FLUX.1 Fill dev Q4_K_S 已在 Windows 挂断点下载，校验信息见
  `flux_fill_download.json`。下载完成不等于安装或推理通过。
- 选择 ObjectRemovalFluxFill v2 LoRA 作第一轮消除专用适配。

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
