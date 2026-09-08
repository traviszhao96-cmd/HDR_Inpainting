# HDR 物体消除 — 长期方案规划

> 更新时间：2026-09（基于当前已完成工作与讨论收敛）
> 概述：做一个支持 **Ultra HDR** 的 AI 物体消除工具。核心方法论：**"内容交给模型，HDR 交给数学"**。

---

## 0. 一句话总纲

> 在 `inpaint_sdr(sdr, mask)` 一个函数里切换"内容生成"后端，其余（gainmap 用 Poisson、只在擦除区重建 HDR、再编码）全程本地且与后端无关。长期把内容层落到本机 **RTX 4070 TiS（16GB）**，前端用**交互画范围 + SAM 自动分割**，得到零云成本、精确 mask、HDR 一致的产品。

---

## 1. 目标

用户给一个 mask（交互绘制 + SAM 自动分割），把物体从画面擦掉，同时：
- 擦除区 **HDR 不穿帮**（物体被擦后，该区域 HDR 反应仍与周围一致）；
- **未擦除区 HDR 零损失**（保留原始 gainmap）；
- 最终输出仍是标准的 Ultra HDR JPEG。

---

## 2. 核心架构（已验证无泄漏）

```
Ultra HDR 输入
  └─ 解码 → SDR(8-bit) + HDR(10-bit HLG)          [libultrahdr: -o3 -O3 / -o1 -O5]
        ├─ gainmap = HDR_lin / SDR_lin             (低频色调映射)
        ├─ mask  = 交互画范围 + SAM 自动分割
        ├─ 内容层  inpaint_sdr(SDR, mask)          ← 唯一需要 GPU/花钱的步骤
        ├─ HDR层   Poisson 修 gainmap(mask 区)      ← 本地、免费、纯数学
        ├─ 只重建擦除区 HDR = SDR修复_lin × gainmap修复，并入整体 HDR
        └─ libultrahdr 重编码 → Ultra HDR JPEG
```

**关键点（对齐 Google Photos 的 Ultra HDR 编辑做法）：**
1. 内容动了（AI 消除）→ 只在**改动区**重建 gainmap；**未改动区保留原 gainmap**，故 HDR 全局不丢。
2. **gainmap 是低频**（实测：90% 像素梯度 < 0.9，B 通道近乎恒定）→ **Poisson（Laplace）在数学上最优**，无需 AI 模型。
3. **所有 gainmap / HDR 逻辑留在本地**，与内容后端无关；只有"内容生成"依赖模型/GPU。

### 为什么不能让编码器从"旧 HDR"重算 gainmap（重点）
直接把"含物体的原始 HDR + 已擦除的 SDR"喂给编码器，会让擦除区的 gainmap = `物体HDR ÷ 背景SDR`，即**物体 HDR 泄漏**（此前的 bug）。正确做法是先用 Poisson 修 gainmap，再用 `SDR修复_lin × gainmap修复` **重建擦除区 HDR**，然后才编码。

---

## 3. 各层方案

| 层 | 短期（当前/验证） | 长期（目标） | 说明 |
|----|----------------|-------------|------|
| **内容层**（SDR 消除） | **OpenAI gpt-image-1**（`inpaint_sdr()` 已插好，OpenCV 兜底） | **本机 4070 TiS + SD-inpaint / LaMa** | 免费、精确 mask、快；16GB 足以跑 SDXL-inpaint |
| **mask** | 手画（测试用, `draw_lamp_mask`） | **交互画范围 + SAM 自动分割** | 参考 [Inpaint-Anything](https://github.com/USTC-IMCL/Inpaint-Anything) 链路 |
| **gainmap 层** | Poisson | 维持 Poisson；必要时升级 **GMODiff** 学习式精修 | 低频场景 Poisson 即最优 |
| **拼接/编码** | libultrahdr | 同左 | 不变 |

---

## 4. 分阶段路线

### 阶段 1 —— 封装 + 通路验证（已完成）
- `hdr_ai_erase.py` 封装为 `inpaint_sdr(sdr, mask, backend=...)`，提供 `opencv` / `openai` 两个后端。
- 用 OpenCV 替身跑通全链路：gainmap Poisson + 重建 + 编码，验证 **HDR 通道健康**（std ≈ 283/284/285）、**无 gainmap 泄漏**。
- 用 **gpt-image-1** 做一次真实内容消除，验证"效果通路"（已提供 `--content-only` + `--sdr-erased` 两步）。

### 阶段 2 —— 本地 GPU 内容层（主攻）
- 在 **RTX 4070 TiS**（16GB）上跑内容生成：**SDXL-inpaint**（求质量）或 **LaMa**（求快/大 mask）。
- 用 **OpenClaw**（现有远程 AI 控制工具）在台式机装环境、常驻服务；用 **Tailscale / SSH** 让 M4/浏览器访问其 ComfyUI API（免费，无需付费远程桌面软件）。
- 内容层仍填入同一个 `inpaint_sdr()`。

### 阶段 3 —— 产品化 / 升级
- 若 Poisson 不足 → 接 **GMODiff** 学习式 gainmap 精修（需数据、工程量大）。
- 部署形态二选一：
  - **台式机当常驻 GPU 服务**（免费、自控，适合自用）；
  - **云端**（省心、按张付费，适合对外/量小）。

---

## 5. 关键已知事实（决策依赖）

1. **硬件**
   - 本机（Mac）：Apple M4 / 24GB / MPS —— 适合开发 + 轻活，跑大型 SD 偏慢。
   - **台式机：RTX 4070 Ti Super / 16GB** —— 跑 SD-inpaint 完全够（SD1.5-Inpaint 轻松、SDXL-Inpaint 舒适）。
2. **付款**
   - **Replicate** 走 Stripe，**不直接支持支付宝/微信/银联**；国内用户需**国际卡或虚拟卡**。预付费充值（1 年有效、不可退），自动充值阈值 ≥$5、补货 ≥$15。
   - **国内云**（支付宝即可）：阿里云 `qwen-image-edit-plus` ≈¥0.21/张；字节 Seedream 5.0 Pro ≈¥0.3 起。
3. **平台限制**
   - **liblib 开放 API**：只有 `文生图/图生图/ControlNet`，**不能提交自定义 ComfyUI 工作流**；其在线 ComfyUI 也无稳定公开 API。
   - 要"上传工作流 JSON + REST 调用"的**云端 ComfyUI**：**ComfyDeploy**（最稳，锁版本）/ RunComfy / RunPod / ViewComfy。
4. **gpt-image-1**（OpenAI `/v1/images/edits`）
   - mask 语义：**alpha=0（透明）= 编辑区**（与我们的 mask 相反，需转换，脚本内 `OPENAI_EDIT_ALPHA`）。
   - 输入上限 ~1024 级，需缩放后回放；**标准 $0.04/张、高 $0.08/张**，延迟 ~10–30s。

---

## 6. 代码结构（当前落点）

`hdr_ai_erase.py`：
```
inpaint_sdr(sdr, mask, backend="opencv"|"openai", prompt=..., out_png=...)
  ├─ OpenCVInpainter   (Telea, 免费替身)
  └─ OpenAIInpainter   (gpt-image-1 edits, 读 OPENAI_API_KEY)
poisson_inpaint(image, mask)        # 包围盒加速的 Jacobi-Laplace
srgb_to_linear / hlg_to_linear / linear_to_hlg
compute_gainmap(hdr, sdr)
load/save_rgba1010102 / rgba8888
decode_uhdr / encode_uhdr / build_clean_input_uhdr   # 用 build/ultrahdr_app
```

CLI：
```bash
# 全链路，OpenCV 兜底
python hdr_ai_erase.py --inpainter opencv --mask output/lamp_mask.png

# 只跑内容步（单独测 OpenAI），存 erased SDR
python hdr_ai_erase.py --content-only --inpainter openai \
    --sdr-in sdr.png --mask mask.png --out erased_sdr.png --prompt "..."

# 用已有 erased SDR 跑后半段（gainmap+重建+编码）
python hdr_ai_erase.py --sdr-erased erased_sdr.png --mask mask.png
```

关键输入文件：
- 干净输入 UHDR：`output/clean_input_uhdr.jpg`（绕开 `building_uhdr_hlg_output_jpegsave_encoded.jpg` 的蓝通道损坏）。
- 干净 HDR 源：`libultrahdr/tests/photo/building_hlgRGBA1010102_output.raw`（= `building_hlg_output.raw` = `building_decoded.raw`）。
- 测试 mask：`output/lamp_mask.png`（路灯）。

---

## 7. 待定决策（下一步）

1. **内容层最终环境**（当前唯一悬而未决）：
   - A. 台式机本地 SD-inpaint / LaMa（免费、精确、自控）——**推荐**；
   - B. 云端（省心、按张付费）。
2. 台式机**是否已装 CUDA/Python/ComfyUI**，以及 OpenClaw 如何在台式机执行常驻服务。
3. 是否需要用 **Tailscale/SSH** 打通 M4 ↔ 台式机，让 M4 管线直接调用台式机 ComfyUI `/prompt`。

---

## 8. 参考

- [libultrahdr（Google）](https://github.com/google/libultrahdr) —— Ultra HDR 参考编解码库；打包约定：`R=bit0-9, G=10-19, B=20-29, A=30-31`。
- [Android 编辑 Ultra HDR 指南](https://developer.android.com/media/grow/ultra-hdr/edit) —— 编辑时"保留 vs 重建 gainmap"的思路。
- [GMODiff（arXiv 2512.16357）](https://ar5iv.labs.arxiv.org/html/2512.16357) —— 用扩散模型一步精修 gainmap（HDR = (L+α)·exp2(1+G·Qmax)）。
- [LaMa（WACV 2022）](https://ar5iv.labs.arxiv.org/html/2109.07161) —— 大 mask、分辨率鲁棒的内容填充，行业"消物体"默认。
- [Inpaint-Anything](https://github.com/USTC-IMCL/Inpaint-Anything) —— SAM + LaMa/SD 的"点选→分割→消除"，交互+自动分割参考。
- [Google Photos 保留 Ultra HDR 的复杂编辑](https://9to5google.com/2024/10/16/google-photos-ultra-hdr-edit/) —— 商业先例，验证此链路可行。
