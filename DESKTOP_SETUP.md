# 部署与运行（M4 前端/管线 + Windows 台式机 GPU 服务）

> 更新时间：2026-09-18
> 现状架构：**Mac(M4) 跑 seg_ui（前端 + 后端 + SAM 分割 + gainmap 重建 + HDR 编码）**，
> **Windows 台式机(RTX 4070 TiS) 跑 ComfyUI 作为内容生成 GPU 服务**，两者通过 **Tailscale** 内网互访。

---

## 0. 角色分工

| 机器 | 职责 |
|------|------|
| **Mac (M4)** | seg_ui 前端(3000) + 后端(7860)；SAM2.1 分割(MPS)；gainmap 重建；libultrahdr 编码；本地作品集 |
| **Windows 台式机 (4070 TiS)** | ComfyUI(8188)。内容引擎：FLUX.1 Fill、SD1.5-inpaint、LaMa 等；生成式 gainmap |
| **连接** | Tailscale（`https://win-hm9ig3vhnaa.tailfff622.ts.net`），无需公网暴露、无需付费远程桌面 |

---

## 1. 克隆 + 子模块 + 本机补丁

```bash
git clone git@github.com:traviszhao96-cmd/HDR_Inpainting.git
cd HDR_Inpainting
git submodule update --init
# 应用本机对 libultrahdr 的两处修复（子模块无法 push，必须打补丁）
cd libultrahdr && git apply ../scripts/libultrahdr-local-fixes.patch && cd ..
```
> `libultrahdr` 指向 `google/libultrahdr @ 93c2349`（含上游测试图，首次拉取约几百 MB）。
> 补丁内容见 `scripts/README.md`：EXIF 长度、NEON P3/BT.2020 系数分支。

## 2. 构建 ultrahdr_app（必须开 ISO + XMP 元数据）

HDR 输出为 **Display P3 JPEG HDR**，编码器需同时写 ISO 21496-1 与 XMP gainmap 元数据：

```bash
cmake -S libultrahdr -B build -DUHDR_WRITE_ISO=ON -DUHDR_WRITE_XMP=ON
cmake --build build -j 6            # 产物 build/ultrahdr_app（Windows 为 ultrahdr_app.exe）
```
> Windows 下构建 `ultrahdr_app` 后复制到仓库根 `build/`（管线默认路径 `build/ultrahdr_app`）。

## 3. Windows 台式机：ComfyUI GPU 服务

1. 安装 ComfyUI + 自定义节点（GGUF、AusBoss 等），放置模型：
   - `models/unet/`：`flux-1-fill-dev-Q4_K_S.gguf`（或对应名）+ Klein 变体
   - `models/loras/`：`removal_timestep_alpha-2-1740.safetensors`（ObjectRemovalFluxFill v2）
   - `models/checkpoints/`：`sd-v1-5-inpainting.safetensors`
   - `models/lama/`：`big-lama.pt`（AusBoss LaMa 节点用）
2. 启动（**绑定 0.0.0.0 以便 Tailscale 访问**）：
   ```bat
   python main.py --listen 0.0.0.0 --port 8188
   ```
3. 用 Tailscale 让 M4 可达：`tailscale up`（同一 tailnet），M4 用
   `https://win-hm9ig3vhnaa.tailfff622.ts.net` 访问其 `/system_stats`、`/prompt`。
4. 校验：浏览器打开该地址能进 ComfyUI；`curl <url>/system_stats` 返回 JSON。

## 4. M4：seg_ui（前端 + 后端）

```bash
cd seg_ui
./run-local.sh        # 自动建 venv、装依赖、导出 COMFYUI_URL、起后端 7860 + 前端 3000
```
`run-local.sh` 已带以下环境变量（可用同名 shell 变量覆盖）：
```
COMFYUI_URL=https://win-hm9ig3vhnaa.tailfff622.ts.net
COMFYUI_ENGINE=sd15          # 前端默认会覆盖为 comfyui-flux-fill
COMFYUI_CHECKPOINT=sd-v1-5-inpainting.safetensors
COMFYUI_LAMA_MODEL=big-lama.pt
COMFYUI_WORK_SIZE=768
```
打开 <http://localhost:3000>：圈选对象 → 选内容引擎（**FLUX.1 Fill-dev + 消除 LoRA v2** 为默认）→ 继续消除。

## 5. 内容引擎与 gainmap 模式

- **内容引擎**（前端下拉）：`comfyui-flux-fill`（默认）/ `sd15` / `lama` / `opencv`。
- **gainmap 模式**（前端下拉）：
  - `generative`（**默认**）：FLUX Fill 8 步、最长边 576，实验性（非 HDR 训练模型）；
  - `smooth-interpolation`：谐波/Poisson 插值，约 0.85s，稳定但平淡。
- 输出：单通道 gainmap 的 **Display P3 JPEG HDR**，EXIF 经编码器 `-x` 写入。

## 6. 离线回归（不调用生成模型）

```bash
seg_ui/.venv/bin/python -m unittest seg_ui.backend.test_hdr_export
seg_ui/.venv/bin/python -m unittest seg_ui.backend.test_sdr_composite
```

## 7. 隐私与泄漏防护（重要）

- **仅留本机、不入库**：`seg_ui/test_images/*`（除 README）、`test_masks/`、`test_runs/`、`results/`、`portfolio/`、`output/`、`build/`、`*.raw`。
- **密钥**：`qwen.env`、`qwen.env.save`、`*.env`、`*.key`、`*token*` 均被 `.gitignore` 忽略；不要写进代码或提交。
- **ComfyUI 无鉴权、节点可执行代码**：台式机 ComfyUI 只绑内网（Tailscale），不要 `0.0.0.0` 暴露到公网、不要开 UPnP；只用信任的自定义节点。
- 论文/评审文档中引用的样张、原始输出保存在本机 `test_runs/`，不随 Git 同步。