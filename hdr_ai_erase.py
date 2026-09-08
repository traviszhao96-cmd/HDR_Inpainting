#!/usr/bin/env python3
"""
HDR AI Erasure (消除) pipeline — pluggable content-inpainting backend.

Goal: erase an object from an Ultra HDR JPEG while keeping the HDR response
consistent. We do NOT let the encoder recompute the gainmap from the original
HDR (which still contains the object) — that leaks the object's HDR into the
erased region. Instead we rebuild the erased-region HDR as
    SDR_inpainted_linear * gainmap_inpainted
so the encoder computes an object-free gainmap there.

THE SWAP POINT is the content inpainter: `inpaint_sdr(sdr, mask)`. Two backends:
  - OpenCVInpainter  (Telea)  : offline stand-in, no cost.
  - OpenAIInpainter  (GPT Image /v1/images/edits): real generative fill.
The gainmap step is always done locally with Poisson (it's low-frequency).

CLI:
  # full pipeline, OpenCV stand-in
  python hdr_ai_erase.py --inpainter opencv --mask mask.png

  # only the SDR content step via OpenAI -> save erased SDR (test the model first)
  python hdr_ai_erase.py --content-only --inpainter openai \
      --sdr-in sdr.png --mask mask.png --out erased_sdr.png

  # full pipeline using an already-generated erased SDR (e.g. from OpenAI above)
  python hdr_ai_erase.py --sdr-erased erased_sdr.png --mask mask.png

Dependencies: numpy, PIL, cv2, requests (for OpenAI). Uses build/ultrahdr_app.
The bundled test HDR is Display P3/HLG; the SDR base image is BT.709/sRGB.
"""

import argparse
import base64
import io
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

WORK_DIR = Path("/Users/travis.zhao/ultrahdr")
OUTPUT_DIR = WORK_DIR / "output"
ULTRAHDR_APP = WORK_DIR / "build" / "ultrahdr_app"

WIDTH = 2464
HEIGHT = 3280

UHDR_INPUT = OUTPUT_DIR / "clean_input_uhdr.jpg"
HDR_ORIGINAL_RAW = WORK_DIR / "libultrahdr/tests/photo/building_hlgRGBA1010102_output.raw"
SDR_ORIGINAL_JPG = WORK_DIR / "libultrahdr/tests/photo/building.jpg"

SDR_RAW = OUTPUT_DIR / "_erase_sdr.raw"
HDR_RAW = OUTPUT_DIR / "_erase_hdr.raw"
HDR_REBUILT_RAW = OUTPUT_DIR / "_erase_hdr_rebuilt.raw"
SDR_INP_RAW = OUTPUT_DIR / "_erase_sdr_inp.raw"
RESULT_UHDR = OUTPUT_DIR / "erase_result_uhdr.jpg"

# OpenAI image-edit options
OPENAI_EDIT_URL = "https://api.openai.com/v1/images/edits"
OPENAI_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2")
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_WORK_SIZE = 1024          # longest side downscaled for the API call
OPENAI_CONTEXT_MIN = 128         # minimum context around the mask, in source pixels
# GPT Image mask: alpha=0 (transparent) => EDIT region; alpha=255 => preserve.
# If a model produces wrong edits, try flipping this to 255.
OPENAI_EDIT_ALPHA = 0


# =============================================================================
# 0. Color-space primitives (HLG <-> linear, sRGB <-> linear)
# =============================================================================

def hlg_to_linear(x):
    a = 0.17883277
    b = 1 - 4 * a
    c = 0.5 - a * np.log(4 * a)
    return np.where(x <= 0.5, x ** 2 / 3.0, (np.exp((x - c) / a) + b) / 12.0)


def linear_to_hlg(x):
    a = 0.17883277
    b = 1 - 4 * a
    c = 0.5 - a * np.log(4 * a)
    x = np.clip(x, 0, None)
    with np.errstate(invalid="ignore"):
        return np.where(x <= 1.0 / 12.0, np.sqrt(3.0 * x), a * np.log(12.0 * x - b) + c)


def srgb_to_linear(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


# =============================================================================
# 1. Raw I/O
# =============================================================================

def load_rgba1010102(path, width, height):
    d = np.fromfile(path, dtype=np.uint32).reshape(height, width)
    r = (d & 0x3FF).astype(np.float32)
    g = ((d >> 10) & 0x3FF).astype(np.float32)
    b = ((d >> 20) & 0x3FF).astype(np.float32)
    return np.stack([r, g, b], axis=-1)


def save_rgba1010102(rgb10, path, width, height):
    r = np.clip(np.round(rgb10[:, :, 0]), 0, 1023).astype(np.uint32)
    g = np.clip(np.round(rgb10[:, :, 1]), 0, 1023).astype(np.uint32)
    b = np.clip(np.round(rgb10[:, :, 2]), 0, 1023).astype(np.uint32)
    a = np.full((height, width), 0x3, dtype=np.uint32)
    packed = r | (g << 10) | (b << 20) | (a << 30)
    packed.tofile(path)


def load_rgba8888(path, width, height):
    d = np.fromfile(path, dtype=np.uint8).reshape(height, width, 4)
    return d[:, :, :3].astype(np.float32)


def save_rgba8888(rgb8, path, width, height):
    r = np.clip(np.round(rgb8[:, :, 0]), 0, 255).astype(np.uint8)
    g = np.clip(np.round(rgb8[:, :, 1]), 0, 255).astype(np.uint8)
    b = np.clip(np.round(rgb8[:, :, 2]), 0, 255).astype(np.uint8)
    a = np.full((height, width), 255, dtype=np.uint8)
    np.stack([r, g, b, a], axis=-1).ravel().tofile(path)


# =============================================================================
# 2. Gainmap computation
# =============================================================================

def compute_gainmap(hdr_10bit, sdr_8bit):
    sdr_lin = srgb_to_linear(sdr_8bit / 255.0)
    hdr_lin = hlg_to_linear(hdr_10bit / 1023.0)
    eps = 1e-6
    gain = hdr_lin / np.maximum(sdr_lin, eps)
    return np.clip(gain, 0.01, 100.0), sdr_lin, hdr_lin


def compute_luminance_gainmap(hdr_10bit, sdr_8bit):
    """Compute one BT.709 linear-light gain value per pixel.

    Both inputs to the edit stage come from the Ultra HDR decoder in the
    BT.709 base gamut. A scalar luminance gain preserves the inpainted SDR
    chromaticity and matches the single-channel map requested at encode time.
    """
    sdr_lin = srgb_to_linear(sdr_8bit / 255.0)
    hdr_lin = hlg_to_linear(hdr_10bit / 1023.0)
    luma_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    sdr_luma = np.sum(sdr_lin * luma_weights, axis=2)
    hdr_luma = np.sum(hdr_lin * luma_weights, axis=2)
    gain = hdr_luma / np.maximum(sdr_luma, 1e-6)
    return np.clip(gain, 0.01, 100.0), sdr_lin, hdr_lin


# =============================================================================
# 3. Poisson (Laplacian) inpainting — iterative Jacobi, bounding-box optimized
# =============================================================================

def poisson_inpaint(image, mask, max_iters=5000, tolerance=1e-6):
    image = image.astype(np.float64)
    mask = mask.astype(bool)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return image.astype(np.float32)
    y0, y1 = max(0, ys.min() - 1), min(image.shape[0], ys.max() + 2)
    x0, x1 = max(0, xs.min() - 1), min(image.shape[1], xs.max() + 2)

    sub = image[y0:y1, x0:x1].copy()
    msub = mask[y0:y1, x0:x1]
    # Update every masked pixel, including masks that touch an image edge.
    # At image edges the average uses only available in-bounds neighbours
    # (a zero-normal-gradient boundary). The previous implementation froze
    # masked edge pixels, leaking the original HDR object along the border.
    interior = msub.copy()
    if (~msub).sum() > 0:
        sub[msub] = sub[~msub].mean()
    else:
        raise ValueError("Poisson inpainting requires known pixels around the mask")

    prev = sub.copy()
    for it in range(max_iters):
        neighbor_sum = np.zeros_like(sub)
        neighbor_count = np.zeros(sub.shape, dtype=np.uint8)
        neighbor_sum[1:, :] += sub[:-1, :]
        neighbor_count[1:, :] += 1
        neighbor_sum[:-1, :] += sub[1:, :]
        neighbor_count[:-1, :] += 1
        neighbor_sum[:, 1:] += sub[:, :-1]
        neighbor_count[:, 1:] += 1
        neighbor_sum[:, :-1] += sub[:, 1:]
        neighbor_count[:, :-1] += 1
        new_vals = neighbor_sum / neighbor_count
        sub = np.where(interior, new_vals, sub)
        if it % 100 == 0:
            diff = np.abs(sub[interior] - prev[interior]).max()
            if diff < tolerance:
                break
            prev = sub.copy()

    result = image.copy()
    result[y0:y1, x0:x1] = sub
    return result.astype(np.float32)


# =============================================================================
# 4. Content inpainting — THE SWAP POINT. Backends below.
# =============================================================================

def inpaint_sdr(sdr, mask, backend="opencv", prompt=None, out_png=None, model=None):
    """Inpaint SDR inside mask. `sdr` float32 HxWx3 (0-255), `mask` bool HxW.
    Returns float32 HxWx3. `prompt`/`out_png` only used by the OpenAI backend."""
    if backend == "opencv":
        return OpenCVInpainter()(sdr, mask)
    if backend == "openai":
        return OpenAIInpainter(model=model)(sdr, mask, prompt=prompt, out_png=out_png)
    raise ValueError(f"unknown backend {backend}")


class OpenCVInpainter:
    """Offline stand-in (Telea). No cost; smooth fill, does not hallucinate."""

    def __call__(self, sdr, mask):
        import cv2
        sdr_u8 = np.clip(np.round(sdr), 0, 255).astype(np.uint8)
        mask_u8 = (mask.astype(np.uint8) * 255)
        out = sdr_u8.copy()
        for ch in range(3):
            out[:, :, ch] = cv2.inpaint(sdr_u8[:, :, ch], mask_u8, 10, cv2.INPAINT_TELEA)
        return out.astype(np.float32)


class OpenAIInpainter:
    """Real generative fill via a GPT Image model and /v1/images/edits."""

    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.environ.get(OPENAI_API_KEY_ENV)
        self.model = model or OPENAI_MODEL
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def __call__(self, sdr, mask, prompt=None, out_png=None):
        import requests
        prompt = prompt or ("Remove the object inside the marked region and fill it "
                            "with the surrounding natural background, keeping lighting, "
                            "perspective and detail consistent with the rest of the image.")
        source = np.clip(np.round(sdr), 0, 255).astype(np.uint8)
        if mask.shape != source.shape[:2]:
            raise ValueError(f"mask shape {mask.shape} does not match image shape {source.shape[:2]}")
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            raise ValueError("mask is empty")

        # Send a contextual crop instead of shrinking the entire source image.
        # This keeps the erased object large enough for the model to reconstruct
        # surrounding detail while avoiding unrelated full-frame regeneration.
        mask_w = int(xs.max() - xs.min() + 1)
        mask_h = int(ys.max() - ys.min() + 1)
        padding = max(OPENAI_CONTEXT_MIN, max(mask_w, mask_h))
        x0 = max(0, int(xs.min()) - padding)
        y0 = max(0, int(ys.min()) - padding)
        x1 = min(source.shape[1], int(xs.max()) + padding + 1)
        y1 = min(source.shape[0], int(ys.max()) + padding + 1)
        crop_source = source[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        im = Image.fromarray(crop_source)

        # Resize SDR + mask to the working size (API input constraints).
        crop_w, crop_h = im.size
        scale = OPENAI_WORK_SIZE / max(crop_w, crop_h)
        if scale < 1.0:
            im = im.resize((round(crop_w * scale), round(crop_h * scale)), Image.LANCZOS)
        work_size = im.size
        print(f"  OpenAI ROI: ({x0},{y0})-({x1},{y1}), API input: {work_size}")

        # Build mask as RGBA: alpha = OPENAI_EDIT_ALPHA where we want to edit (mask True),
        # alpha = 255 elsewhere (preserve). RGB set to white.
        m = np.zeros((work_size[1], work_size[0], 4), dtype=np.uint8)
        mask_img = Image.fromarray((crop_mask.astype(np.uint8) * 255))
        mask_img = mask_img.resize(work_size, Image.LANCZOS)
        m[:, :, 0] = 255
        m[:, :, 1] = 255
        m[:, :, 2] = 255
        m[:, :, 3] = np.where(np.array(mask_img) > 128,
                              OPENAI_EDIT_ALPHA, 255).astype(np.uint8)
        mask_png = Image.fromarray(m, "RGBA")

        def to_png_bytes(img):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        files = {
            "model": (None, self.model),
            "prompt": (None, prompt),
            "image": ("image.png", to_png_bytes(im), "image/png"),
            "mask": ("mask.png", to_png_bytes(mask_png), "image/png"),
            "n": (None, "1"),
            "quality": (None, "medium"),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        r = requests.post(OPENAI_EDIT_URL, headers=headers, files=files, timeout=180)
        r.raise_for_status()
        data = r.json()
        b64 = data["data"][0]["b64_json"]
        out = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

        # The edit model may alter pixels outside the requested region and the API
        # result may be lower resolution. Restore all unmasked pixels exactly from
        # the original full-resolution SDR image.
        generated_crop = np.array(
            out.resize((crop_w, crop_h), Image.LANCZOS), dtype=np.uint8
        )
        composited_crop = np.where(crop_mask[:, :, None], generated_crop, crop_source)
        out = source.copy()
        out[y0:y1, x0:x1] = composited_crop
        if out_png:
            Path(out_png).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(out).save(out_png)
        return out.astype(np.float32)


# =============================================================================
# 5. Mask helpers (test helper — final product uses interactive/SAM mask)
# =============================================================================

def draw_lamp_mask(width, height):
    yy, xx = np.ogrid[:height, :width]
    head = ((xx - 1500) / 95.0) ** 2 + ((yy - 105) / 80.0) ** 2 <= 1.0
    p0 = np.array([1570.0, 15.0])
    p1 = np.array([1325.0, 302.0])
    d = p1 - p0
    L = np.hypot(*d)
    t = ((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) / (L * L)
    t = np.clip(t, 0, 1)
    px = p0[0] + t * d[0]
    py = p0[1] + t * d[1]
    pole = np.hypot(xx - px, yy - py) <= 16.0
    return head | pole


def load_or_draw_mask(width, height, mask_file=None):
    if mask_file:
        mask_path = Path(mask_file)
        if not mask_path.is_file():
            raise FileNotFoundError(f"mask not found: {mask_path}")
        m = Image.open(mask_path).convert("L").resize((width, height), Image.NEAREST)
        return np.array(m) > 128
    return draw_lamp_mask(width, height)


# =============================================================================
# 6. Encode / decode via libultrahdr
# =============================================================================

def decode_uhdr(uhdr_path, sdr_raw, hdr_raw):
    subprocess.run([str(ULTRAHDR_APP), "-m", "1", "-j", str(uhdr_path),
                    "-o", "3", "-O", "3", "-z", str(sdr_raw)],
                   check=True, capture_output=True, cwd=str(WORK_DIR))
    subprocess.run([str(ULTRAHDR_APP), "-m", "1", "-j", str(uhdr_path),
                    "-o", "1", "-O", "5", "-z", str(hdr_raw)],
                   check=True, capture_output=True, cwd=str(WORK_DIR))


def encode_uhdr(hdr_raw, sdr_raw, out_path, *, hdr_gamut, multi_channel_gainmap=False):
    """Encode HLG HDR + sRGB SDR whose gamut labels describe the raw pixels.

    The original HDR test RAW is Display P3, but HDR decoded from our Ultra HDR
    base is BT.709. Reusing the P3 label for the decoded BT.709 pixels produces
    channel-dependent gain outliers and visible false colour in smooth skies.
    """
    subprocess.run([str(ULTRAHDR_APP), "-m", "0",
                    "-p", str(hdr_raw), "-y", str(sdr_raw),
                    "-w", str(WIDTH), "-h", str(HEIGHT),
                    "-a", "5", "-b", "3", "-t", "1",
                    "-C", str(hdr_gamut), "-c", "0",
                    "-M", "1" if multi_channel_gainmap else "0",
                    "-q", "100", "-Q", "100", "-z", str(out_path)],
                   check=True, capture_output=True, cwd=str(WORK_DIR))


def build_clean_input_uhdr():
    hdr = np.fromfile(HDR_ORIGINAL_RAW, dtype=np.uint32)
    hdr.tofile(HDR_RAW)
    bj = np.array(Image.open(SDR_ORIGINAL_JPG)).astype(np.float32)
    save_rgba8888(bj, SDR_INP_RAW, WIDTH, HEIGHT)
    # This source RAW is explicitly Display P3 / HLG.
    encode_uhdr(HDR_RAW, SDR_INP_RAW, UHDR_INPUT, hdr_gamut=1)
    print(f"Built clean input UHDR -> {UHDR_INPUT}")


# =============================================================================
# 7. Content-only step (for testing OpenAI/generative fill in isolation)
# =============================================================================

def content_only(sdr_png, mask_png, out_png, backend, prompt, model=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sdr = np.array(Image.open(sdr_png).convert("RGB")).astype(np.float32)
    mask = load_or_draw_mask(sdr.shape[1], sdr.shape[0], mask_png)
    print(f"Content-only: sdr {sdr.shape}, mask pixels {mask.sum()}, backend={backend}")
    sdr_inp = inpaint_sdr(sdr, mask, backend=backend, prompt=prompt,
                          out_png=out_png, model=model)
    # Always write the result to disk (OpenCV backend ignores out_png itself).
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.clip(sdr_inp, 0, 255).astype(np.uint8)).save(out_png)
        print(f"Saved erased SDR -> {out_png}")
    else:
        print("(--out not given; erased SDR returned in-memory)")
    return sdr_inp


# =============================================================================
# 8. Full pipeline
# =============================================================================

def run(backend="opencv", mask_file=None, prompt=None, sdr_erased=None, model=None):
    print("=" * 60)
    print("HDR AI Erasure (消除) Pipeline")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not ULTRAHDR_APP.is_file():
        raise FileNotFoundError(f"ultrahdr_app not found: {ULTRAHDR_APP}")

    if not UHDR_INPUT.exists():
        print("\n[0] Building clean input UHDR ...")
        build_clean_input_uhdr()

    print("\n[1] Decoding input UHDR -> SDR + HDR ...")
    decode_uhdr(UHDR_INPUT, SDR_RAW, HDR_RAW)
    sdr = load_rgba8888(SDR_RAW, WIDTH, HEIGHT)
    hdr = load_rgba1010102(HDR_RAW, WIDTH, HEIGHT)

    print("\n[2] Computing single-channel BT.709 luminance gainmap ...")
    gainmap, _, _ = compute_luminance_gainmap(hdr, sdr)

    mask = load_or_draw_mask(WIDTH, HEIGHT, mask_file)
    print(f"  Mask pixels: {mask.sum()} ({100 * mask.sum() / (HEIGHT * WIDTH):.2f}%)")

    if sdr_erased is not None:
        # Use a pre-generated erased SDR (e.g. from OpenAI content-only step).
        print(f"\n[3] Using pre-generated erased SDR -> {sdr_erased}")
        sdr_inp = np.array(Image.open(sdr_erased).convert("RGB").resize((WIDTH, HEIGHT))).astype(np.float32)
    else:
        print(f"\n[3] Inpainting SDR via backend={backend} ...")
        sdr_inp = inpaint_sdr(sdr, mask, backend=backend, prompt=prompt,
                              out_png=OUTPUT_DIR / "erase_sdr_inp.png", model=model)
    save_rgba8888(sdr_inp, SDR_INP_RAW, WIDTH, HEIGHT)

    print("\n[4] Inpainting single-channel gainmap (Poisson) ...")
    gain_inp = poisson_inpaint(gainmap, mask)

    print("\n[5] Rebuilding erased-region HDR = SDR_inp_lin * gainmap_inp ...")
    sdr_inp_lin = srgb_to_linear(sdr_inp / 255.0)
    hdr_erase_lin = np.clip(sdr_inp_lin * gain_inp[:, :, None], 0.0, 1.0)
    hdr_erase_10 = linear_to_hlg(hdr_erase_lin) * 1023.0
    hdr_rebuilt = hdr.copy()
    hdr_rebuilt[mask] = np.clip(hdr_erase_10, 0, 1023)[mask]
    save_rgba1010102(hdr_rebuilt, HDR_REBUILT_RAW, WIDTH, HEIGHT)

    print("\n[6] Encoding Ultra HDR ...")
    # libultrahdr decodes this input's HDR rendition into the BT.709 base
    # gamut. HDR_REBUILT_RAW therefore contains BT.709 pixels, not the P3
    # pixels of HDR_ORIGINAL_RAW. Keep the label consistent and use the
    # requested single-channel gain map for a stable luminance-only boost.
    encode_uhdr(HDR_REBUILT_RAW, SDR_INP_RAW, RESULT_UHDR,
                hdr_gamut=0, multi_channel_gainmap=False)
    print(f"  Saved {RESULT_UHDR} ({RESULT_UHDR.stat().st_size/1e6:.1f} MB)")

    print("\n[7] Decoding result for verification ...")
    res_sdr = OUTPUT_DIR / "_erase_res_sdr.raw"
    res_hdr = OUTPUT_DIR / "_erase_res_hdr.raw"
    decode_uhdr(RESULT_UHDR, res_sdr, res_hdr)
    sdr_r = load_rgba8888(res_sdr, WIDTH, HEIGHT)
    hdr_r = load_rgba1010102(res_hdr, WIDTH, HEIGHT)
    print("  Result HDR channel std:",
          [round(float(hdr_r[:, :, c].std()), 1) for c in range(3)])
    print("  Result SDR channel std:",
          [round(float(sdr_r[:, :, c].std()), 1) for c in range(3)])

    def norm8(x):
        return (np.clip(x / 1023.0 if x.max() > 255 else x / 255.0, 0, 1) * 255).astype(np.uint8)

    before = norm8(hdr)
    after = norm8(hdr_r)
    comp = np.concatenate([before, after], axis=1)
    comp_img = Image.fromarray(comp)
    comp_img.thumbnail((1600, 1600))
    comp_img.save(OUTPUT_DIR / "erase_before_after_hdr.png")
    print(f"  Before/After HDR -> {OUTPUT_DIR/'erase_before_after_hdr.png'}")
    print("\nDone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="HDR AI Erasure (pluggable content backend)")
    p.add_argument("--inpainter", choices=["opencv", "openai"], default="opencv")
    p.add_argument("--mask", type=str, default=None, help="mask PNG (white=erase); default = drawn lamp mask")
    p.add_argument("--prompt", type=str, default=None, help="prompt for the OpenAI backend")
    p.add_argument("--model", type=str, default=OPENAI_MODEL,
                   help=f"OpenAI image model (default: {OPENAI_MODEL})")
    p.add_argument("--sdr-erased", type=str, default=None, help="use an existing erased SDR PNG instead of inpainting now")
    p.add_argument("--content-only", action="store_true", help="only run the SDR content step")
    p.add_argument("--sdr-in", type=str, default=None, help="SDR PNG for --content-only")
    p.add_argument("--out", type=str, default=None, help="output erased SDR PNG for --content-only")
    args = p.parse_args()

    if args.content_only:
        if not args.sdr_in:
            p.error("--content-only requires --sdr-in")
        if not args.out:
            p.error("--content-only requires --out")
        content_only(args.sdr_in, args.mask, args.out, args.inpainter,
                     args.prompt, model=args.model)
    else:
        run(backend=args.inpainter, mask_file=args.mask, prompt=args.prompt,
            sdr_erased=args.sdr_erased, model=args.model)
