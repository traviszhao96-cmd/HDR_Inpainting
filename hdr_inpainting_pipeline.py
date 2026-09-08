#!/usr/bin/env python3
"""
HDR AI Inpainting Pipeline — Gainmap Analysis & Poisson Inpainting

Dependencies: numpy, PIL (Pillow), OpenCV (cv2) — no scipy, no matplotlib.

Pipeline:
  1. Load SDR (RGBA8888) and HDR (RGBA1010102) decoded from Ultra HDR JPEG
  2. Convert both to linear floating-point
  3. Compute gainmap = HDR_linear / SDR_linear (per-channel)
  4. Visualize gainmap characteristics (heatmap, histogram, edge analysis)
  5. Apply a synthetic mask (simulating AI erasure region)
  6. Poisson inpaint the gainmap in the masked region
  7. Reconstruct HDR from inpainted SDR x inpainted gainmap
  8. Write back to 10-bit RGBA1010102 raw format
  9. Re-encode to Ultra HDR JPEG via libultrahdr

Core insight: Gainmap is low-frequency and smooth, making Poisson inpainting viable.
For regions with rich luminance variation, we use edge-aware weighting.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import struct
import os
import sys
import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
WIDTH = 2464
HEIGHT = 3280
CHANNELS = 4  # RGBA

WORK_DIR = Path("/Users/travis.zhao/ultrahdr")
OUTPUT_DIR = WORK_DIR / "output"
ULTRAHDR_APP = WORK_DIR / "build" / "ultrahdr_app"

# Input: Ultra HDR encoded JPEG
UHDR_INPUT = WORK_DIR / "libultrahdr/tests/photo/building_uhdr_hlg_output_jpegsave_encoded.jpg"
# Original SDR JPEG (before UHDR encoding) — use this as SDR reference
# because the SDR decoded from UHDR has color shifts from the encoding process
SDR_ORIGINAL_JPG = WORK_DIR / "libultrahdr/tests/photo/building.jpg"

# Decoded raw files
SDR_RAW = OUTPUT_DIR / "sdr_decoded.raw"

# HDR source — CRITICAL: do NOT decode this from the source UHDR.
# building_uhdr_hlg_output_jpegsave_encoded.jpg was created with a saturated blue
# channel (B pinned near 1001/1023, std ~11), so decoding it yields a color-corrupt
# HDR (SDR and HDR both lose the blue channel). The pristine HDR raw below pairs
# with building.jpg, produces a clean low-frequency gainmap, and round-trips
# losslessly through the encoder (R/G/B std ≈ 282/284/284 after encode+decode).
HDR_RAW = WORK_DIR / "libultrahdr/tests/photo/building_hlgRGBA1010102_output.raw"

# Output files
GAINMAP_PNG = OUTPUT_DIR / "gainmap_visualization.png"
GAINMAP_HIST_TXT = OUTPUT_DIR / "gainmap_statistics.txt"
MASK_PNG = OUTPUT_DIR / "mask.png"
SDR_OVERLAY_PNG = OUTPUT_DIR / "sdr_with_mask.png"
INPAINTED_GAINMAP_PNG = OUTPUT_DIR / "inpainted_gainmap.png"
COMPARISON_PNG = OUTPUT_DIR / "gainmap_comparison.png"
SDR_INPAINTED_RAW = OUTPUT_DIR / "sdr_inpainted.raw"
HDR_RECONSTRUCTED_RAW = OUTPUT_DIR / "hdr_reconstructed.raw"
UHDR_RESULT = OUTPUT_DIR / "result_uhdr.jpg"


# =============================================================================
# 1. Raw Data I/O
# =============================================================================

def load_rgba8888(path, width, height):
    """Load RGBA8888 raw data (4 bytes per pixel, 8 bits per channel)."""
    data = np.fromfile(path, dtype=np.uint8)
    expected = width * height * 4
    if len(data) != expected:
        raise ValueError(f"Expected {expected} bytes, got {len(data)}")
    return data.reshape((height, width, 4)).astype(np.float32)


def load_rgba1010102(path, width, height):
    """Load RGBA1010102 raw data (4 bytes per pixel, 10 bits per channel packed)."""
    data = np.fromfile(path, dtype=np.uint32)
    expected = width * height
    if len(data) != expected:
        raise ValueError(f"Expected {expected} pixels, got {len(data)}")
    img = data.reshape((height, width))
    r = (img & 0x3FF).astype(np.float32)
    g = ((img >> 10) & 0x3FF).astype(np.float32)
    b = ((img >> 20) & 0x3FF).astype(np.float32)
    return np.stack([r, g, b], axis=-1)


def save_rgba1010102(rgb_10bit, path, width, height):
    """Save 10-bit RGB data as RGBA1010102 raw format."""
    r = np.clip(np.round(rgb_10bit[:,:,0]), 0, 1023).astype(np.uint32)
    g = np.clip(np.round(rgb_10bit[:,:,1]), 0, 1023).astype(np.uint32)
    b = np.clip(np.round(rgb_10bit[:,:,2]), 0, 1023).astype(np.uint32)
    a = np.full((height, width), 0x3, dtype=np.uint32)
    packed = r | (g << 10) | (b << 20) | (a << 30)
    packed.tofile(path)


def save_rgba8888(rgb_8bit, path, width, height):
    """Save 8-bit RGB data as RGBA8888 raw format."""
    r = np.clip(np.round(rgb_8bit[:,:,0]), 0, 255).astype(np.uint8)
    g = np.clip(np.round(rgb_8bit[:,:,1]), 0, 255).astype(np.uint8)
    b = np.clip(np.round(rgb_8bit[:,:,2]), 0, 255).astype(np.uint8)
    a = np.full((height, width), 255, dtype=np.uint8)
    rgba = np.stack([r, g, b, a], axis=-1).ravel()
    rgba.tofile(path)


# =============================================================================
# 2. Color Space Conversion: HLG <-> Linear, sRGB <-> Linear, Display P3 <-> sRGB
# =============================================================================

# Display P3 linear -> sRGB linear conversion matrix
# Derived from Display P3 -> XYZ -> sRGB
P3_TO_SRGB = np.array([
    [ 1.2249, -0.2249,  0.0000],
    [-0.0420,  1.0421,  0.0000],
    [-0.0196, -0.0786,  1.0982]
], dtype=np.float32)

# sRGB linear -> Display P3 linear (inverse)
SRGB_TO_P3 = np.array([
    [0.8225, 0.1774, 0.0000],
    [0.0332, 0.9669, 0.0000],
    [0.0171, 0.0724, 0.9108],
], dtype=np.float32)


def p3_to_srgb_linear(rgb_p3):
    """Convert Display P3 linear RGB to sRGB linear RGB."""
    # rgb_p3: (H, W, 3) float
    h, w = rgb_p3.shape[:2]
    flat = rgb_p3.reshape(-1, 3)
    result = flat @ P3_TO_SRGB.T
    return result.reshape(h, w, 3)


def srgb_to_p3_linear(rgb_srgb):
    """Convert sRGB linear RGB to Display P3 linear RGB."""
    h, w = rgb_srgb.shape[:2]
    flat = rgb_srgb.reshape(-1, 3)
    result = flat @ SRGB_TO_P3.T
    return result.reshape(h, w, 3)

def hlg_to_linear(x):
    """HLG OETF inverse -> linear. x in [0, 1]. Returns linear in [0, 1] (relative)."""
    a = 0.17883277
    b = 1 - 4 * a
    c = 0.5 - a * np.log(4 * a)
    return np.where(x <= 0.5, x**2 / 3.0, (np.exp((x - c) / a) + b) / 12.0)


def linear_to_hlg(x):
    """Linear -> HLG OETF."""
    a = 0.17883277
    b = 1 - 4 * a
    c = 0.5 - a * np.log(4 * a)
    return np.where(x <= 1.0/12.0, np.sqrt(3.0 * x), a * np.log(12.0 * x - b) + c)


def srgb_to_linear(x):
    """sRGB -> linear (inverse gamma)."""
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    """Linear -> sRGB gamma."""
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0/2.4)) - 0.055)


# =============================================================================
# 3. Gainmap Computation
# =============================================================================

def compute_gainmap(hdr_10bit, sdr_8bit):
    """
    Compute gainmap = HDR_linear / SDR_linear.

    HDR: 10-bit HLG in Display P3 primaries -> linear P3
    SDR: 8-bit sRGB -> linear sRGB (BT.709 primaries)

    The gainmap is a per-channel ratio: gain = P3_linear / sRGB_linear.
    When we reconstruct: HDR(P3) = SDR_inpainted(sRGB) × gainmap.
    This naturally encodes the color space difference — no manual P3↔sRGB
    conversion is needed. The libultrahdr encoder handles the color space
    metadata internally.
    """
    sdr_norm = sdr_8bit / 255.0
    sdr_lin = srgb_to_linear(sdr_norm)

    hdr_norm = hdr_10bit / 1023.0
    hdr_lin = hlg_to_linear(hdr_norm)

    eps = 1e-6
    gainmap = hdr_lin / np.maximum(sdr_lin, eps)
    gainmap = np.clip(gainmap, 0.01, 100.0)

    return gainmap, sdr_lin, hdr_lin


# =============================================================================
# 4. NumPy-based Morphological Operations (no scipy needed)
# =============================================================================

def binary_dilation_np(mask, iterations=1):
    """Binary dilation using pure numpy + OpenCV fallback."""
    import cv2
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = mask.astype(np.uint8)
    result = cv2.dilate(mask_u8, kernel, iterations=iterations)
    return result.astype(bool)


def binary_erosion_np(mask, iterations=1):
    """Binary erosion using pure numpy + OpenCV fallback."""
    import cv2
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = mask.astype(np.uint8)
    result = cv2.erode(mask_u8, kernel, iterations=iterations)
    return result.astype(bool)


# =============================================================================
# 5. Gainmap Visualization (PIL-based, no matplotlib)
# =============================================================================

def visualize_gainmap(gainmap, sdr_8bit, output_path, comparison_path):
    """Create gainmap heatmap + per-channel visualization using PIL."""
    h, w = gainmap.shape[:2]

    # Luminance gainmap
    lum_gain = 0.2126 * gainmap[:,:,0] + 0.7152 * gainmap[:,:,1] + 0.0722 * gainmap[:,:,2]
    lum_gain_log = np.log2(np.clip(lum_gain, 0.01, 100))

    # Display scale
    scale = min(800 / max(h, w), 1.0)
    dh, dw = int(h * scale), int(w * scale)

    def to_heatmap(data, w, h):
        """Convert floating data to 3-channel RGB heatmap."""
        vmin, vmax = np.percentile(data, 1), np.percentile(data, 99)
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        # Jet-like colormap in pure numpy
        r = np.clip(np.where(norm < 0.5, 0.0, (norm - 0.5) * 2.0), 0, 1)
        g = np.clip(np.where(norm < 0.5, norm * 2.0, 2.0 - norm * 2.0), 0, 1)
        b = np.clip(np.where(norm < 0.5, 1.0 - norm * 2.0, 0.0), 0, 1)
        rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
        return np.array(Image.fromarray(rgb).resize((w, h), Image.LANCZOS))

    # Build heatmap images
    heat_lum = to_heatmap(lum_gain_log, dw, dh)

    # Per-channel heatmaps
    ch_heatmaps = []
    for ch in range(3):
        ch_log = np.log2(np.clip(gainmap[:,:,ch], 0.01, 100))
        ch_heatmaps.append(to_heatmap(ch_log, min(dw // 2, 400), min(dh // 2, 300)))

    # SDR preview
    sdr_rgb = np.clip(sdr_8bit[:,:,:3], 0, 255).astype(np.uint8)
    sdr_preview = np.array(Image.fromarray(sdr_rgb).resize((dw, dh), Image.LANCZOS))

    # Build composite canvas
    margin = 10
    text_h = 24
    total_w = dw * 2 + margin
    total_h = dh + max(dh // 2, 300) + text_h * 2 + margin * 3

    canvas = Image.new('RGB', (total_w, total_h), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)

    # Row 1: SDR preview | Gainmap heatmap
    draw.text((margin, margin), "SDR Preview (AI model input)", fill=(0, 0, 0))
    canvas.paste(Image.fromarray(sdr_preview), (margin, margin + text_h))

    draw.text((dw + margin * 2, margin), "Gainmap (log2, luminance)", fill=(0, 0, 0))
    canvas.paste(Image.fromarray(heat_lum), (dw + margin * 2, margin + text_h))

    # Row 2: Per-channel gainmaps
    ch_start_y = dh + text_h + margin * 2
    draw.text((margin, ch_start_y - text_h), "Per-channel Gainmap (log2 scale)", fill=(0, 0, 0))
    ch_w = min(dw // 2, 400)
    ch_h = min(dh // 2, 300)
    for ch in range(3):
        x = margin + ch * (ch_w + margin)
        canvas.paste(Image.fromarray(ch_heatmaps[ch]), (x, ch_start_y))
        draw.text((x, ch_start_y - text_h + ch_start_y - margin),
                  f"{['R','G','B'][ch]}", fill=(0, 0, 0))

    canvas.save(output_path)
    print(f"Gainmap visualization saved to {output_path}")

    # Also save a comparison: original SDR vs SDR with gainmap overlay
    comp = Image.new('RGB', (dw * 2 + margin, dh + text_h + margin), (240, 240, 240))
    draw_comp = ImageDraw.Draw(comp)
    draw_comp.text((margin, margin), "SDR Original", fill=(0, 0, 0))
    comp.paste(Image.fromarray(sdr_preview), (margin, margin + text_h))
    draw_comp.text((dw + margin * 2, margin), "Gainmap Overlay (green=high gain)", fill=(0, 0, 0))
    # Create gainmap overlay: blend heatmap on SDR
    overlay = (Image.fromarray(sdr_preview).convert('RGBA'))
    heat_alpha = Image.fromarray(heat_lum).convert('L')
    heat_alpha = heat_alpha.point(lambda x: x // 2)  # 50% opacity
    overlay_rgba = Image.fromarray(heat_lum).convert('RGBA')
    overlay_rgba.putalpha(heat_alpha)
    blended = Image.fromarray(sdr_preview).convert('RGBA')
    blended.paste(overlay_rgba, (0, 0), overlay_rgba)
    comp.paste(blended.convert('RGB'), (dw + margin * 2, margin + text_h))
    comp.save(comparison_path)
    print(f"Gainmap comparison saved to {comparison_path}")

    return lum_gain, lum_gain_log


def compute_statistics(gainmap, output_path):
    """Compute and save gainmap statistics."""
    h, w = gainmap.shape[:2]
    lines = []
    lines.append("=" * 60)
    lines.append("Gainmap Statistics")
    lines.append("=" * 60)
    lines.append(f"Image size: {w} x {h}")
    lines.append(f"Total pixels: {w * h:,}")
    lines.append("")

    # Per-channel
    for ch, name in enumerate(['R', 'G', 'B']):
        data = gainmap[:,:,ch].ravel()
        data_f = data[(data > 0.01) & (data < 100)]
        lines.append(f"{name} channel:")
        lines.append(f"  Mean:   {data_f.mean():.6f}")
        lines.append(f"  Median: {np.median(data_f):.6f}")
        lines.append(f"  Std:    {data_f.std():.6f}")
        lines.append(f"  Min:    {data_f.min():.6f}")
        lines.append(f"  Max:    {data_f.max():.6f}")
        for p in [50, 75, 90, 95, 99]:
            lines.append(f"  P{p}:    {np.percentile(data_f, p):.6f}")
        lines.append("")

    # Edge analysis
    lum_gain = 0.2126 * gainmap[:,:,0] + 0.7152 * gainmap[:,:,1] + 0.0722 * gainmap[:,:,2]
    gy, gx = np.gradient(lum_gain)
    grad_mag = np.sqrt(gx**2 + gy**2)

    lines.append("Gradient (edge) analysis:")
    lines.append(f"  Mean gradient:   {grad_mag.mean():.6f}")
    lines.append(f"  Max gradient:    {grad_mag.max():.6f}")
    lines.append(f"  P90 gradient:    {np.percentile(grad_mag, 90):.6f}")
    lines.append(f"  P95 gradient:    {np.percentile(grad_mag, 95):.6f}")
    lines.append(f"  P99 gradient:    {np.percentile(grad_mag, 99):.6f}")
    lines.append("")

    # Smoothness assessment
    # Low gradient = smooth = good for Poisson inpainting
    smooth_ratio = (grad_mag < np.percentile(grad_mag, 90)).mean()
    lines.append(f"Smoothness assessment:")
    lines.append(f"  Pixels with gradient < P90: {smooth_ratio*100:.1f}%")
    lines.append(f"  High-frequency edge ratio:  {(grad_mag > np.percentile(grad_mag, 99)).mean()*100:.2f}%")
    lines.append("")

    # Poisson inpainting feasibility
    lines.append("Poisson inpainting feasibility:")
    if np.percentile(grad_mag, 99) < 0.5:
        lines.append("  ✓ Gainmap is very smooth — Poisson inpainting should work well")
    elif np.percentile(grad_mag, 99) < 2.0:
        lines.append("  ⚠ Gainmap has moderate edges — Poisson may produce visible transitions")
        lines.append("    Consider edge-aware (Navier-Stokes) inpainting for sharp transitions")
    else:
        lines.append("  ⚠ Gainmap has significant high-frequency content")
        lines.append("    Poisson inpainting alone may not be sufficient for all regions")

    text = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(text)
    print(text)
    print(f"\nStatistics saved to {output_path}")


# =============================================================================
# 6. Poisson Inpainting (Iterative Jacobi Method, pure NumPy)
# =============================================================================

def poisson_inpaint(image, mask, max_iters=5000, tolerance=1e-6):
    """
    Poisson (Laplace) inpainting using iterative Jacobi method.

    Solves Δu = 0 in the mask region with Dirichlet boundary conditions.

    The gainmap is inherently low-frequency, making Laplace inpainting the
    theoretically correct approach. The smooth interpolation from boundary
    values naturally extends the gainmap's smooth gradients.

    Args:
        image: (H, W) float64, known values outside mask
        mask: (H, W) bool, True where inpainting is needed
        max_iters: maximum iterations
        tolerance: convergence tolerance

    Returns:
        inpainted: (H, W) float32
    """
    result = image.copy().astype(np.float64)
    mask = mask.astype(bool)
    h, w = image.shape

    # Pre-compute interior mask indices for efficiency
    # Only iterate over mask pixels that aren't on the image boundary
    interior = mask.copy()
    interior[0, :] = False
    interior[-1, :] = False
    interior[:, 0] = False
    interior[:, -1] = False

    prev = result.copy()

    for iteration in range(max_iters):
        # Jacobi: u_new[i,j] = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1]) / 4
        up = np.roll(result, 1, axis=0)
        down = np.roll(result, -1, axis=0)
        left = np.roll(result, 1, axis=1)
        right = np.roll(result, -1, axis=1)

        new_vals = (up + down + left + right) / 4.0

        # Fix wrap-around at edges
        new_vals[0, :] = result[0, :]
        new_vals[-1, :] = result[-1, :]
        new_vals[:, 0] = result[:, 0]
        new_vals[:, -1] = result[:, -1]

        # Only update masked pixels
        result = np.where(interior, new_vals, result)

        # Convergence check every 100 iterations
        if iteration % 100 == 0:
            diff = np.abs(result[interior] - prev[interior]).max()
            if diff < tolerance:
                print(f"    Poisson converged at iteration {iteration}, max diff = {diff:.2e}")
                break
            prev = result.copy()

    if iteration >= max_iters - 1:
        diff = np.abs(result[interior] - prev[interior]).max()
        print(f"    Poisson reached max iterations ({max_iters}), max diff = {diff:.2e}")

    return result.astype(np.float32)


def edge_aware_inpaint(image, mask, max_iters=500, tolerance=1e-6):
    """
    Edge-aware inpainting using Perona-Malik anisotropic diffusion.

    For gainmap regions WITH rich luminance variation.
    div(g(|∇u|) · ∇u) = 0, where g(s) = 1/(1 + (s/K)²)

    The diffusion coefficient decreases near edges, preserving sharp transitions
    while still providing smooth interpolation in flat regions.
    """
    result = image.copy().astype(np.float64)
    mask = mask.astype(bool)
    h, w = image.shape

    # Edge threshold from known region
    gy, gx = np.gradient(image)
    grad_mag = np.sqrt(gx**2 + gy**2)
    K = np.percentile(grad_mag[~mask], 75)
    if K < 1e-6:
        K = 0.1
    print(f"    Edge-aware K (contrast threshold): {K:.6f}")

    prev = result.copy()

    for iteration in range(max_iters):
        gy, gx = np.gradient(result)
        grad_mag_cur = np.sqrt(gx**2 + gy**2)
        g = 1.0 / (1.0 + (grad_mag_cur / K)**2)  # diffusion coefficient

        gx_flow = g * gx
        gy_flow = g * gy

        # Divergence via central differences
        dgx_dx = np.zeros_like(gx_flow)
        dgx_dx[:, 1:-1] = (gx_flow[:, 2:] - gx_flow[:, :-2]) / 2.0
        dgx_dx[:, 0] = gx_flow[:, 1] - gx_flow[:, 0]
        dgx_dx[:, -1] = gx_flow[:, -1] - gx_flow[:, -2]

        dgy_dy = np.zeros_like(gy_flow)
        dgy_dy[1:-1, :] = (gy_flow[2:, :] - gy_flow[:-2, :]) / 2.0
        dgy_dy[0, :] = gy_flow[1, :] - gy_flow[0, :]
        dgy_dy[-1, :] = gy_flow[-1, :] - gy_flow[-2, :]

        laplacian = dgx_dx + dgy_dy
        dt = 0.1
        result = result + dt * laplacian * mask
        result = np.where(mask, result, image)

        if iteration % 100 == 0:
            diff = np.abs(result[mask] - prev[mask]).max()
            if diff < tolerance:
                print(f"    Edge-aware converged at iteration {iteration}, max diff = {diff:.2e}")
                break
            prev = result.copy()

    if iteration >= max_iters - 1:
        diff = np.abs(result[mask] - prev[mask]).max()
        print(f"    Edge-aware reached max iterations ({max_iters}), max diff = {diff:.2e}")

    return result.astype(np.float32)


# =============================================================================
# 7. OpenCV-based Inpainting (for SDR comparison)
# =============================================================================

def opencv_inpaint(image, mask, method='telea', radius=5):
    """OpenCV inpainting for SDR base image (uint8)."""
    import cv2
    img_min, img_max = image.min(), image.max()
    if img_max <= img_min:
        img_max = img_min + 1
    img_uint8 = np.clip((image - img_min) / (img_max - img_min) * 255, 0, 255).astype(np.uint8)
    mask_uint8 = mask.astype(np.uint8) * 255

    if method == 'telea':
        result = cv2.inpaint(img_uint8, mask_uint8, radius, cv2.INPAINT_TELEA)
    else:
        result = cv2.inpaint(img_uint8, mask_uint8, radius, cv2.INPAINT_NS)

    return result.astype(np.float32) / 255.0 * (img_max - img_min) + img_min


# =============================================================================
# 8. Mask Creation
# =============================================================================

def create_test_mask(h, w, center_y=None, center_x=None, radius=None):
    """Create a circular mask simulating an erased region."""
    center_y = center_y or h // 2
    center_x = center_x or w // 2
    radius = radius or min(h, w) // 6

    yy, xx = np.ogrid[:h, :w]
    mask = (yy - center_y)**2 + (xx - center_x)**2 <= radius**2
    return mask


# =============================================================================
# 9. Full Pipeline
# =============================================================================

def run_pipeline():
    """Run the complete HDR inpainting pipeline."""
    print("=" * 60)
    print("HDR AI Inpainting Pipeline — Gainmap Analysis & Inpainting")
    print("=" * 60)

    # [1] Load data
    print("\n[1/9] Loading data...")
    # HDR: decoded from Ultra HDR JPEG (10-bit HLG, Display P3)
    hdr_10bit = load_rgba1010102(HDR_RAW, WIDTH, HEIGHT)
    print(f"  HDR: {hdr_10bit.shape}, range [{hdr_10bit.min():.0f}, {hdr_10bit.max():.0f}]")

    # SDR: use the original JPEG (building.jpg) as reference, NOT the decoded UHDR SDR.
    # The UHDR decode pipeline modifies SDR colors significantly (mean diff ~88 vs original),
    # so we use the original JPEG which has the colors the user expects.
    sdr_jpg = np.array(Image.open(SDR_ORIGINAL_JPG)).astype(np.float32)
    # Pad to RGBA
    sdr_8bit = np.zeros((HEIGHT, WIDTH, 4), dtype=np.float32)
    sdr_8bit[:,:,:3] = sdr_jpg
    sdr_8bit[:,:,3] = 255
    print(f"  SDR (original JPEG): {sdr_8bit.shape}, range [{sdr_8bit[:,:,:3].min():.0f}, {sdr_8bit[:,:,:3].max():.0f}]")

    # [2] Compute gainmap
    print("\n[2/9] Computing gainmap (HDR_linear / SDR_linear)...")
    gainmap, sdr_lin, hdr_lin = compute_gainmap(hdr_10bit, sdr_8bit[:,:,:3])
    print(f"  Gainmap range: [{gainmap.min():.4f}, {gainmap.max():.4f}]")

    # [3] Statistics
    print("\n[3/9] Computing gainmap statistics...")
    compute_statistics(gainmap, GAINMAP_HIST_TXT)

    # [4] Visualize
    print("\n[4/9] Visualizing gainmap...")
    lum_gain, lum_gain_log = visualize_gainmap(gainmap, sdr_8bit, GAINMAP_PNG, COMPARISON_PNG)

    # [5] Create test mask
    print("\n[5/9] Creating test mask...")
    # Position mask in a region with both flat and textured areas
    mask = create_test_mask(HEIGHT, WIDTH,
                           center_y=HEIGHT // 2 - 100,
                           center_x=WIDTH // 2 + 200,
                           radius=400)
    print(f"  Mask pixels: {mask.sum()} ({100*mask.sum()/(HEIGHT*WIDTH):.2f}% of image)")

    # Save mask
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save(MASK_PNG)

    # Save SDR with mask overlay
    sdr_viz = np.clip(sdr_8bit[:,:,:3], 0, 255).astype(np.uint8).copy()
    sdr_viz[mask] = [255, 0, 0]  # Red overlay on mask region
    Image.fromarray(sdr_viz).save(SDR_OVERLAY_PNG)
    print(f"  Mask + overlay saved to {MASK_PNG}, {SDR_OVERLAY_PNG}")

    # [6] SDR inpainting (simulate AI model output)
    print("\n[6/9] SDR base inpainting (OpenCV stand-in for AI model)...")
    sdr_rgb = sdr_8bit[:,:,:3]
    sdr_inpainted = np.zeros_like(sdr_rgb)
    for ch in range(3):
        sdr_inpainted[:,:,ch] = opencv_inpaint(sdr_rgb[:,:,ch], mask, method='telea', radius=10)
    save_rgba8888(sdr_inpainted, SDR_INPAINTED_RAW, WIDTH, HEIGHT)
    print(f"  SDR inpainted saved to {SDR_INPAINTED_RAW}")

    # [7] Gainmap analysis (for reference, not used in encoding)
    print("\n[7/9] Gainmap analysis (for documentation)...")
    gainmap_poisson = np.zeros_like(gainmap)
    for ch in range(3):
        print(f"    Channel {['R','G','B'][ch]}:")
        gainmap_poisson[:,:,ch] = poisson_inpaint(gainmap[:,:,ch], mask)

    # Visualize gainmap comparison
    scale = min(800 / max(HEIGHT, WIDTH), 1.0)
    dh, dw = int(HEIGHT * scale), int(WIDTH * scale)

    def to_heatmap_rgb(data):
        vmin, vmax = np.percentile(data, 1), np.percentile(data, 99)
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        r = np.clip(np.where(norm < 0.5, 0.0, (norm - 0.5) * 2.0), 0, 1)
        g = np.clip(np.where(norm < 0.5, norm * 2.0, 2.0 - norm * 2.0), 0, 1)
        b = np.clip(np.where(norm < 0.5, 1.0 - norm * 2.0, 0.0), 0, 1)
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    orig_lum = 0.2126 * gainmap[:,:,0] + 0.7152 * gainmap[:,:,1] + 0.0722 * gainmap[:,:,2]
    poisson_lum = 0.2126 * gainmap_poisson[:,:,0] + 0.7152 * gainmap_poisson[:,:,1] + 0.0722 * gainmap_poisson[:,:,2]

    orig_log = np.log2(np.clip(orig_lum, 0.01, 100))
    poisson_log = np.log2(np.clip(poisson_lum, 0.01, 100))

    orig_hm = Image.fromarray(to_heatmap_rgb(orig_log)).resize((dw, dh), Image.LANCZOS)
    poisson_hm = Image.fromarray(to_heatmap_rgb(poisson_log)).resize((dw, dh), Image.LANCZOS)

    margin = 5
    total_w = dw * 2 + margin
    total_h = dh + 30
    canvas = Image.new('RGB', (total_w, total_h), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 5), "Original Gainmap", fill=(0, 0, 0))
    canvas.paste(orig_hm, (margin, 25))
    draw.text((dw + margin * 2, 5), "Poisson Inpainted", fill=(0, 0, 0))
    canvas.paste(poisson_hm, (dw + margin * 2, 25))
    mask_small = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((dw, dh), Image.LANCZOS))
    poisson_arr = np.array(poisson_hm)
    poisson_arr[mask_small > 128] = [255, 255, 255]
    canvas.paste(Image.fromarray(poisson_arr), (dw + margin * 2, 25))
    canvas.save(INPAINTED_GAINMAP_PNG)
    print(f"  Inpainted gainmap comparison saved to {INPAINTED_GAINMAP_PNG}")

    # [8] Encode to Ultra HDR JPEG — multiple approaches
    # Key insight from code analysis (jpegr.cpp):
    #   - generateGainMap() handles gamut conversion internally when HDR and SDR
    #     have different color gamuts (line 607-637). With -C 1 -c 0 it converts
    #     HDR from P3→BT.709 before computing gainmap.
    #   - The encoder always converts SDR to P3 in YUV space before JPEG encoding
    #     (line 277), then writes ICC with the original gamut from -c.
    #   - Previous bug: -C 0 -c 0 forced both to BT.709, skipping gamut conversion.
    #     But the HDR raw is in P3, so gainmap was computed from mismatched gamuts.
    #   - Fix: Use correct color spaces. Defaults (-C 1 -c 0) = HDR P3, SDR BT.709.
    #
    # Approach A: building.jpg SDR (BT.709) + HDR (P3), defaults
    # Approach B: building.jpg SDR (BT.709) + HDR (P3), explicit -C 1 -c 0
    # Approach C: UHDR's decoded SDR (P3) + HDR (P3), properly paired
    print("\n[8/9] Encoding to Ultra HDR JPEG...")

    results = {}

    # --- Approach A: building.jpg SDR (BT.709) + HDR (P3), defaults ---
    uhdr_result_a = OUTPUT_DIR / "result_uhdr_a.jpg"
    print("\n  --- Approach A: building.jpg SDR (BT.709) + HDR (P3), defaults ---")
    cmd_a = [
        str(ULTRAHDR_APP), "-m", "0",
        "-p", str(HDR_RAW),             # HDR (P3, HLG)
        "-y", str(SDR_INPAINTED_RAW),   # Inpainted SDR (BT.709, from building.jpg)
        "-w", str(WIDTH), "-h", str(HEIGHT),
        "-a", "5", "-b", "3",           # RGBA1010102 + RGBA8888
        "-t", "1",                       # HLG
        # -C 1 (P3) and -c 0 (BT.709) are defaults — omit
        "-q", "95", "-Q", "95",
        "-z", str(uhdr_result_a)
    ]
    print(f"  Running: {' '.join(cmd_a)}")
    result_a = subprocess.run(cmd_a, capture_output=True, text=True, cwd=str(WORK_DIR))
    if result_a.returncode == 0:
        size_mb = uhdr_result_a.stat().st_size / (1024*1024)
        print(f"  ✓ Approach A saved to {uhdr_result_a} ({size_mb:.1f} MB)")
        results['A'] = uhdr_result_a
    else:
        print(f"  ✗ Approach A failed: {result_a.stderr}")

    # --- Approach B: Explicit -C 1 -c 0 ---
    uhdr_result_b = OUTPUT_DIR / "result_uhdr_b.jpg"
    print("\n  --- Approach B: explicit -C 1 -c 0 ---")
    cmd_b = [
        str(ULTRAHDR_APP), "-m", "0",
        "-p", str(HDR_RAW),
        "-y", str(SDR_INPAINTED_RAW),
        "-w", str(WIDTH), "-h", str(HEIGHT),
        "-a", "5", "-b", "3",
        "-t", "1", "-C", "1", "-c", "0",   # Explicit: HDR=P3, SDR=BT.709
        "-q", "95", "-Q", "95",
        "-z", str(uhdr_result_b)
    ]
    print(f"  Running: {' '.join(cmd_b)}")
    result_b = subprocess.run(cmd_b, capture_output=True, text=True, cwd=str(WORK_DIR))
    if result_b.returncode == 0:
        size_mb = uhdr_result_b.stat().st_size / (1024*1024)
        print(f"  ✓ Approach B saved to {uhdr_result_b} ({size_mb:.1f} MB)")
        results['B'] = uhdr_result_b
    else:
        print(f"  ✗ Approach B failed: {result_b.stderr}")

    # --- Approach C: UHDR's decoded SDR (P3) + HDR (P3), properly paired ---
    uhdr_result_c = OUTPUT_DIR / "result_uhdr_c.jpg"
    sdr_decoded = load_rgba8888(SDR_RAW, WIDTH, HEIGHT)
    sdr_decoded_rgb = sdr_decoded[:,:,:3]
    # Inpaint the UHDR's decoded SDR in the mask region
    sdr_uhdr_inpainted = sdr_decoded_rgb.copy()
    for ch in range(3):
        sdr_uhdr_inpainted[:,:,ch] = opencv_inpaint(
            sdr_decoded_rgb[:,:,ch], mask, method='telea', radius=10)
    sdr_uhdr_inpainted_raw = OUTPUT_DIR / "sdr_uhdr_inpainted.raw"
    save_rgba8888(sdr_uhdr_inpainted, sdr_uhdr_inpainted_raw, WIDTH, HEIGHT)

    print("\n  --- Approach C: UHDR decoded SDR (P3) + HDR (P3), properly paired ---")
    cmd_c = [
        str(ULTRAHDR_APP), "-m", "0",
        "-p", str(HDR_RAW),
        "-y", str(sdr_uhdr_inpainted_raw),
        "-w", str(WIDTH), "-h", str(HEIGHT),
        "-a", "5", "-b", "3",
        "-t", "1", "-C", "1", "-c", "1",   # Both P3, properly paired
        "-q", "95", "-Q", "95",
        "-z", str(uhdr_result_c)
    ]
    print(f"  Running: {' '.join(cmd_c)}")
    result_c = subprocess.run(cmd_c, capture_output=True, text=True, cwd=str(WORK_DIR))
    if result_c.returncode == 0:
        size_mb = uhdr_result_c.stat().st_size / (1024*1024)
        print(f"  ✓ Approach C saved to {uhdr_result_c} ({size_mb:.1f} MB)")
        results['C'] = uhdr_result_c
    else:
        print(f"  ✗ Approach C failed: {result_c.stderr}")

    # Set main result to approach A for backward compatibility
    if 'A' in results:
        UHDR_RESULT = uhdr_result_a
    elif 'B' in results:
        UHDR_RESULT = uhdr_result_b
    elif 'C' in results:
        UHDR_RESULT = uhdr_result_c

    # Quality analysis
    print("\n" + "=" * 60)
    print("Inpainting Quality Analysis (Masked Region)")
    print("=" * 60)

    for ch, name in enumerate(['R', 'G', 'B']):
        orig = gainmap[:,:,ch][mask]
        poisson_inp = gainmap_poisson[:,:,ch][mask]

        mae_poisson = np.abs(orig - poisson_inp).mean()

        # Boundary continuity: compare outer ring vs inner ring of mask
        boundary_outer = binary_dilation_np(mask) & ~mask
        boundary_inner = mask & ~binary_erosion_np(mask)
        if boundary_outer.sum() > 0 and boundary_inner.sum() > 0:
            outer_poisson = gainmap_poisson[:,:,ch][boundary_outer]
            inner_poisson = gainmap_poisson[:,:,ch][boundary_inner]
            bc_poisson = np.abs(outer_poisson.mean() - inner_poisson.mean())
        else:
            bc_poisson = 0

        print(f"\n  {name} channel:")
        print(f"    Poisson MAE:       {mae_poisson:.6f}")
        print(f"    Poisson boundary:  {bc_poisson:.6f}")
        print(f"    Original range:    [{orig.min():.6f}, {orig.max():.6f}]")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"\nEncoding approaches tested:")
    print(f"  A: building.jpg SDR (BT.709) + HDR (P3), defaults — best SDR colors")
    print(f"  B: building.jpg SDR + HDR, explicit -C 1 -c 0 — same as A, explicit")
    print(f"  C: UHDR decoded SDR (P3) + HDR (P3), -C 1 -c 1 — best HDR fidelity")
    print(f"\nResults:")
    for label, path in results.items():
        size_mb = path.stat().st_size / (1024*1024)
        print(f"  [{label}] {path.name} ({size_mb:.1f} MB)")
    print(f"\nKey fix: Previous -C 0 -c 0 told encoder HDR was BT.709, but it's P3.")
    print(f"  Correct gamut params let encoder handle P3→BT.709 conversion properly.")
    print(f"\nOutput files in {OUTPUT_DIR}/:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024*1024)
            print(f"  {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    run_pipeline()