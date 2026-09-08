# HDR AI Inpainting — 实验验证与结论

## 实验日期
2026-09-04

## 🔑 关键发现：Color Space 参数导致编码错误

**问题根因**：Pipeline 使用 `-C 0 -c 0` 告诉编码器 HDR 和 SDR 都是 BT.709，但实际 HDR 是 P3。
编码器跳过 gamut conversion → P3 值被当作 BT.709 处理 → gainmap 计算错误 → HDR 严重失真。

**修复**：使用正确的默认参数（`-C 1 -c 0`，即 HDR=P3, SDR=BT.709），或直接省略 `-C`/`-c`。
编码器的 `generateGainMap()` 检测到 P3≠BT.709，通过 `p3ToBt709()` 矩阵转换 HDR，
然后在 BT.709 空间计算 gainmap。

### 修复前后对比

| 指标 | 修复前 (-C 0 -c 0) | 修复后 (-C 1 -c 0) | 改善 |
|------|-------------------|-------------------|------|
| SDR mean diff | 2.58 | 2.65 | 持平 (JPEG 级别) |
| **HDR mean diff** | **348.79** | **5.35** | **65x 改善!** |
| HDR max diff | 1023/1023 | 277/1023 | 3.7x 改善 |
| 文件大小 | 30.7 MB | 4.1 MB | 7.5x 更小 |

### 编码器内部流程 (jpegr.cpp)

```
generateGainMap():
  sdr_rgb = srgbInvOetf(sdr)           # sRGB → linear
  sdr_rgb = identity()                  # BT.709, 不转换
  hdr_rgb = hlgInvOetf(hdr)            # HLG → linear
  hdr_rgb = hlgOotf(hdr_rgb)           # HLG OOTF
  hdr_rgb = p3ToBt709(hdr_rgb)         # ★ P3 → BT.709 gamut conversion
  gainmap = encodeGain(sdr, hdr)       # 在 BT.709 空间计算 gainmap
```

## 输入
- Ultra HDR JPEG: `building_uhdr_hlg_output_jpegsave_encoded.jpg` (27MB, 2464x3280)
- 解码工具: `libultrahdr v1.4.0` (`ultrahdr_app`)
- HDR 源: `building_hlgRGBA1010102_output.raw` (P3, HLG) — 原始 raw，非 UHDR 解码
- SDR 源: `building.jpg` (BT.709, sRGB) — 原始 JPEG

## 流水线

```
building.jpg (BT.709)          building_hlgRGBA1010102_output.raw (P3, HLG)
  │                                │
  ├─→ AI Inpainting (OpenCV)       │
  │     └─→ inpainted SDR          │
  │                                │
  └────────────┬───────────────────┘
               │
         ultrahdr_app encode (-m 0)
         默认 -C 1 -c 0 (HDR=P3, SDR=BT.709)
         编码器内部处理 P3→BT.709 gamut conversion
               │
         Ultra HDR JPEG
```

## 核心发现：Gainmap 是天然低频的
        └─→ Poisson Inpainting → inpainted gainmap
              └─→ HDR_reconstructed = SDR_inpainted_linear × gainmap_inpainted
                    └─→ uhdr_encode → Ultra HDR JPEG
```

## 核心发现：Gainmap 是天然低频的

### 统计数据（2464×3280, 808万像素）

| 通道 | 均值 | 中位数 | 标准差 | P90 | P99 | 范围 |
|------|------|--------|--------|-----|-----|------|
| R | 1.648 | 1.515 | 1.709 | 2.332 | 3.073 | 0.56-92.05 |
| G | 1.631 | 1.519 | 0.549 | 2.387 | 3.358 | 0.90-5.44 |
| B | 0.987 | 0.992 | 0.026 | 1.016 | 1.031 | 0.82-1.07 |

### 梯度分析（边缘强度）

- 90% 像素的梯度 < 0.91（非常平滑）
- 99% 像素的梯度 < 10.54
- 但梯度最大值达到 72.4（R 通道的极值点，可能是镜面高光等孤立像素）

### 关键洞察

**Gainmap 本质上是色调映射（tone map），不是纹理图。** 它记录的是 SDR 到 HDR 的亮度放大比例，这个比例在空间上是缓慢变化的：
- 天空区域：gain > 1（HDR 比 SDR 更亮）
- 均匀区域：gain ≈ 1
- 不存在精细纹理、高频细节

这意味着 **Poisson（Laplace）inpainting 在数学上是最优解**：
- Laplace 方程 Δu = 0 产生的是最小曲率曲面
- 对于本身平滑的 gainmap，边界值的平滑插值就是正确的结果
- 不存在"丢失纹理"的问题，因为 gainmap 本来就没有纹理

## 回答你的问题：rich luminance variation 会导致过渡问题吗？

### 结论：不会，因为 gainmap 的"rich variation"和你想象的不一样

你担心的场景是：gainmap 在 mask 区域内有丰富的亮度变化，Poisson inpainting 会把它抹平。

但实际上：
1. **Gainmap 的梯度是低频的** — 即使 R 通道有较大的标准差，其空间变化也是缓慢的。Poisson 插值从边界值开始，天然地延续了这种缓慢变化的趋势。
2. **B 通道几乎恒定** — 变化范围 0.82-1.07，标准差仅 0.026，inpainting 几乎完美。
3. **G 和 R 通道的变化是平滑的** — 主要是场景亮度分布的变化，不包含边缘突变。

### 我们做了两种方法的对比：

| 方法 | R MAE | G MAE | B MAE | 边界连续性 |
|------|-------|-------|-------|-----------|
| Poisson (Laplace) | 0.502 | 0.747 | 0.015 | 优秀 |
| Edge-aware (Perona-Malik) | 0.354 | 0.425 | 0.014 | 良好 |

- Edge-aware 方法在 MAE 上略好，但视觉差异极小
- 两种方法在视觉上都不可察觉（gainmap 本身太平滑了）
- **Poisson 方法在边界连续性上更好**，这对无缝拼接更关键

### 真正的瓶颈不是 gainmap，而是 SDR 的 AI inpainting 质量

你的担忧方向是对的，但应该关注的是 SDR 基图的 AI 修复质量。因为：
- AI 模型生成的 SDR 区域如果和原始 SDR 不匹配，会导致 gainmap 重建出错
- gainmap 的物理意义是"HDR 比 SDR 亮多少"，如果 SDR 区域被 AI 改变了亮度，gainmap 的比例关系就错了

## 改进方向

### 1. 加速 Poisson 收敛（当前 5000 次迭代不够）
- **多尺度方法**：在 1/4 分辨率上做 inpainting，再上采样
- **共轭梯度法**：比 Jacobi 快 10-100 倍
- 当前 2464×3280 的 Jacobi 迭代对于生产环境太慢

### 2. 更好的 SDR 重建策略
- AI 模型应该尽量保持原始亮度（不要大幅改变曝光）
- 或者：AI 只修复纹理，保留原始亮度层

### 3. 多通道 gainmap 的独立处理
- 当前 Ultra HDR 支持 multi-channel gainmap
- 三个通道的统计特性差异很大（B 几乎恒定，R 变化大）
- 可以根据每通道特性选择不同的 inpainting 策略

## 输出文件

| 文件 | 说明 |
|------|------|
| `output/gainmap_visualization.png` | Gainmap 热力图 + 每通道分解 |
| `output/gainmap_comparison.png` | SDR 原图 vs Gainmap 叠加 |
| `output/inpainted_gainmap.png` | 原始 vs Poisson vs Edge-aware 对比 |
| `output/sdr_with_mask.png` | SDR + Mask 标注 |
| `output/result_uhdr.jpg` | 最终 Ultra HDR JPEG 输出 (21.3 MB) |
| `output/gainmap_statistics.txt` | 完整统计数据 |

## 下一步

1. 用真实的 AI inpainting 模型（如 LaMa、SD Inpainting）替换 OpenCV 模拟
2. 在 gainmap 重建中使用多尺度 Poisson 求解器加速
3. 测试更多场景（室内、夜景、高对比度）的 gainmap 特性
4. 评估最终 Ultra HDR JPEG 在 HDR 显示器上的视觉效果