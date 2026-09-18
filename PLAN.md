# HDR 物体消除 — 方案与进度

> 更新时间：2026-09-18
> 目标：做一个支持 **Ultra HDR** 的 AI 物体消除工具。方法论沉淀为：**"内容交给模型，HDR 交给数学"**。

---

## 0. 一句话总纲

> 浏览器圈选 → 本地 SAM 出全分辨率 mask → **Windows 台式机(4070 TiS) 的 ComfyUI**（经 Tailscale）用
> **FLUX.1 Fill + 消除 LoRA** 等模型消除 SDR → 本地重建 gainmap/合成 HDR → 编码 **Display P3 Ultra HDR**。
> 内容层可插拔；gainmap/HDR 逻辑与内容后端无关。

---

## 1. 目标

用户给一个 mask（交互绘制 + SAM 自动分割），把物体从画面擦掉，同时：
- 擦除区 **HDR 不穿帮**；
- **未擦除区 HDR 零损失**（保留原始 gainmap）；
- 最终输出仍是标准 Ultra HDR JPEG（当前为 Display P3）。

---

## 2. 当前架构（已落地）

```
浏览器圈选 (localhost:3000)
  └─ SAM2.1 出全分辨率 mask (M4 / MPS)
        ├─ 内容层  ComfyUI 消除 SDR   ← Windows 4070 TiS，经 Tailscale
        │    引擎: flux-fill(默认, FLUX.1 Fill-dev Q4_K_S + ObjectRemovalFluxFill v2)
        │          / sd15(sd-v1-5-inpainting) / lama(big-lama)
        ├─ gainmap 层
        │    generative          : FLUX Fill 生成 gainmap（默认，实验性）
        │    smooth-interpolation: 谐波/Poisson 插值（稳定备选）
        ├─ 只重建擦除区 HDR = SDR修复_lin × gainmap修复，并入整体 HDR
        └─ libultrahdr 编码 → Display P3 JPEG HDR（ISO 21496-1 + XMP + EXIF）
```

**关键点（对齐 Google Photos 的 Ultra HDR 编辑做法）：**
1. 只在**改动区**重建 gainmap；**未改动区保留原 gainmap**，HDR 全局不丢。
2. **gainmap 低频** → Poisson 插值在数学上最优；生成式 gainmap 仅作实验对照。
3. **所有 gainmap/HDR 逻辑留在本地**；只有"内容生成"依赖 GPU（台式机）。

### 为什么不能让编码器从"旧 HDR"重算 gainmap（重点）
把"含物体的原始 HDR + 已擦除的 SDR"喂给编码器，会让擦除区 gainmap = `物体HDR ÷ 背景SDR`（**物体 HDR 泄漏**）。
正确做法：先修 gainmap，再用 `SDR修复_lin × gainmap修复` **重建擦除区 HDR**，然后编码。

---

## 3. 各层实现

| 层 | 现状 | 说明 |
|----|------|------|
| **内容层**（SDR 消除） | **FLUX.1 Fill-dev + ObjectRemovalFluxFill v2**（默认），sd15/lama/opencv 可选 | 跑在 Windows 台式机 ComfyUI（Tailscale）；`/erase` 可插拔 |
| **mask** | 浏览器圈选 + **本地 SAM2.1** 自动分割（MPS） | 全分辨率灰度 PNG，白=编辑区 |
| **gainmap 层** | **generative**（默认）/ **smooth-interpolation** | 生成式受网络与模型缓存影响大（端到端约 98–146s） |
| **拼接/编码** | libultrahdr（**已打本机补丁**） | Display P3、单通道 gainmap、ISO+XMP+EXIF |

---

## 4. 进度

### 阶段 1 —— 封装 + 通路验证（✅ 已完成）
- `hdr_ai_erase.py` 封装 `inpaint_sdr(sdr, mask, backend=...)`（opencv/openai）。
- OpenCV 替身跑通全链路，验证 HDR 通道健康、无 gainmap 泄漏。
- gpt-image-1 内容消除验证通路。

### 阶段 2 —— 本地 GPU 内容层（✅ 已完成）
- Windows 台式机 (4070 TiS) 跑 ComfyUI，经 **Tailscale** 被 M4 调用。
- 内容引擎升级为 **FLUX.1 Fill + 消除 LoRA**；端到端 `分割→消除→gainmap→P3 HDR` 打通。

### 阶段 3 —— 质量攻坚（🚧 进行中）
诚实的现状（见 `seg_ui/config/*REVIEW*.md`）：**接口成功 ≠ 质量通过**。
- **SDR 接缝/阴影**：外扩 10px + 有界 RGB 残差校正 + 6px 内缩，改进有限；人形接缝、阴影残留仍在。
- **邻近物体误伤**：外扩蒙版覆盖旁人；需"对象蒙版 + 阴影蒙版分开、提交前确认"。
- **物体替换失败模式**：建筑类会"换成另一栋楼"而非清空（路灯→建筑状、高楼→矮楼）。
- **生成式 gainmap**：可能臆造"人形"增益细节；256×384 会破坏增益结构。
- **HDR 观感**：编码成功不代表在 HDR 显示器上验证过亮度。

---

## 5. 关键已知事实

1. **硬件**：Mac M4/24GB/MPS（开发、SAM、gainmap 重建、编码）；Windows RTX 4070 TiS/16GB（内容生成）。
2. **连通**：Tailscale 内网（`win-hm9ig3vhnaa.tailfff622.ts.net`），不开公网端口。
3. **内容引擎成本**：台式机本地免费；`gpt-image-2` 云端按张付费（需 OpenAI 额度）。
4. **libultrahdr 本机修复**（子模块无法 push）：导出为 `scripts/libultrahdr-local-fixes.patch`，
   clone 后必须 `git apply`（EXIF 长度、NEON P3/BT.2020 系数）。
5. **参考模型/论文**：GMODiff（扩散精修 gainmap）、LaMa、Inpaint-Anything。

---

## 6. 代码落点

- **应用**：`seg_ui/`（前端 page.tsx + 后端 app.py；`/segment` 分割、`/erase` 消除+重建）。
- **CLI/管线**：`hdr_ai_erase.py`（`inpaint_sdr`、`poisson_inpaint`、gainmap、编解码）。
- **实验脚本**：`seg_ui/backend/run_mask_probe.py`、`run_fill_gallery.py`、`run_gainmap_probe.py`、`run_gainmap_sweep.py`、`sdr_composite.py`、`review_sdr_seam.py`、`convert_uhdr_display_p3.py` 等。
- **评审文档**：`seg_ui/config/{INPAINT_PLAN,FILL_GALLERY_REVIEW,GAINMAP_SPEED_REVIEW,SDR_SEAM_REVIEW,PIPELINE_REVIEW,GENERATIVE_GAINMAP_UI_*,GAINMAP_EXPANDED_MASK_REVIEW}*.md`。
- **离线回归**：`seg_ui/backend/test_hdr_export.py`、`test_sdr_composite.py`。

---

## 7. 下一步（待定/攻坚）

1. **SDR 质量**：同种子受控对照，分离"合成算法"与"原始生成"；渐变域边界匹配（未验证）。
2. **遮挡与阴影**：对象蒙版与阴影蒙版分开，提交前让用户确认总范围；近邻实例保护。
3. **物体替换失败**：建筑类增加种子/提示词对照，改进提示与蒙版。
4. **gainmap**：gemerative 仅作实验；默认保留 smooth，生成式仅在达标场景启用。
5. **HDR 观感**：在真实 HDR 显示器上评估 UHDR 亮度/色彩。
6. **文档**：本文件与 `DESKTOP_SETUP.md` 已更新到当前架构。

---

## 8. 参考

- [libultrahdr（Google）](https://github.com/google/libultrahdr) —— Ultra HDR 参考编解码库；打包约定 `R=bit0-9,G=10-19,B=20-29,A=30-31`。
- [Android 编辑 Ultra HDR 指南](https://developer.android.com/media/grow/ultra-hdr/edit) —— "保留 vs 重建 gainmap"。
- [GMODiff（arXiv 2512.16357）](https://ar5iv.labs.arxiv.org/html/2512.16357) —— 扩散模型一步精修 gainmap。
- [LaMa（WACV 2022）](https://ar5iv.labs.arxiv.org/html/2109.07161) —— 大 mask、分辨率鲁棒的内容填充。
- [Inpaint-Anything](https://github.com/USTC-IMCL/Inpaint-Anything) —— SAM + LaMa/SD 的"点选→分割→消除"。
- [ObjectRemovalFluxFill](https://huggingface.co/lrzjason/ObjectRemovalFluxFill) —— FLUX Fill 的消除 LoRA（revision `867f2e6a…`）。
- [Google Photos 保留 Ultra HDR 的复杂编辑](https://9to5google.com/2024/10/16/google-photos-ultra-hdr-edit/) —— 商业先例。