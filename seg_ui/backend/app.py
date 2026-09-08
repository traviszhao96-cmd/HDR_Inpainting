"""Local SAM 2.1 Tiny service for interactive image segmentation."""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from transformers import Sam2Model, Sam2Processor

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
from hdr_ai_erase import (  # noqa: E402
    compute_luminance_gainmap,
    linear_to_hlg,
    load_rgba1010102,
    load_rgba8888,
    poisson_inpaint,
    save_rgba1010102,
    save_rgba8888,
    srgb_to_linear,
)

MODEL_ID = os.environ.get("SEG_MODEL_ID", "facebook/sam2.1-hiera-tiny")
OPENAI_EDIT_URL = "https://api.openai.com/v1/images/edits"
OPENAI_MODEL = "gpt-image-2"
OPENAI_WORK_SIZE = 1024
ULTRAHDR_APP = ROOT_DIR / "build/ultrahdr_app"
DEFAULT_UHDR = ROOT_DIR / "output/clean_input_uhdr.jpg"
RESULTS_DIR = ROOT_DIR / "seg_ui/results"
_model: Sam2Model | None = None
_processor: Sam2Processor | None = None
_load_lock = threading.Lock()


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = select_device()


def get_model() -> tuple[Sam2Model, Sam2Processor]:
    global _model, _processor
    if _model is None or _processor is None:
        with _load_lock:
            if _model is None or _processor is None:
                _processor = Sam2Processor.from_pretrained(MODEL_ID)
                _model = Sam2Model.from_pretrained(MODEL_ID).to(DEVICE)
                _model.eval()
    return _model, _processor


app = FastAPI(title="HDR Mask Lab", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE.upper(),
        "model": MODEL_ID,
        "model_loaded": _model is not None,
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
    }


@app.get("/sample-uhdr")
def sample_uhdr():
    if not DEFAULT_UHDR.is_file():
        raise HTTPException(status_code=404, detail="默认 Ultra HDR 示例不存在")
    return FileResponse(DEFAULT_UHDR, media_type="image/jpeg", filename="clean_input_uhdr.jpg")


@app.get("/results/{job_id}/{filename}")
def result_file(job_id: str, filename: str):
    if not job_id.replace("-", "").isalnum() or Path(filename).name != filename:
        raise HTTPException(status_code=404, detail="结果不存在")
    path = RESULTS_DIR / job_id / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="结果不存在")
    media_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return FileResponse(path, media_type=media_type, filename=filename)


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def decode_uhdr(source_path: Path, width: int, height: int, job_dir: Path):
    sdr_raw = job_dir / "decoded_sdr.raw"
    hdr_raw = job_dir / "decoded_hdr.raw"
    commands = [
        [str(ULTRAHDR_APP), "-m", "1", "-j", str(source_path), "-o", "3", "-O", "3", "-z", str(sdr_raw)],
        [str(ULTRAHDR_APP), "-m", "1", "-j", str(source_path), "-o", "1", "-O", "5", "-z", str(hdr_raw)],
    ]
    try:
        for command in commands:
            subprocess.run(command, check=True, capture_output=True, cwd=str(ROOT_DIR))
    except (OSError, subprocess.CalledProcessError):
        return None
    expected = width * height * 4
    if sdr_raw.stat().st_size != expected or hdr_raw.stat().st_size != expected:
        return None
    return load_rgba8888(sdr_raw, width, height), load_rgba1010102(hdr_raw, width, height)


def openai_downscaled_edit(
    source: np.ndarray,
    mask: np.ndarray,
    *,
    api_key: str,
    prompt: str,
    model: str,
    cloud_input_path: Path,
):
    ys, xs = np.nonzero(mask)
    if not ys.size:
        raise HTTPException(status_code=400, detail="蒙版是空的，请重新圈选")

    mask_width = int(xs.max() - xs.min() + 1)
    mask_height = int(ys.max() - ys.min() + 1)
    padding = max(96, max(mask_width, mask_height))
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(source.shape[1], int(xs.max()) + padding + 1)
    y1 = min(source.shape[0], int(ys.max()) + padding + 1)
    crop_source = source[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]

    crop_height, crop_width = crop_source.shape[:2]
    scale = min(1.0, OPENAI_WORK_SIZE / max(crop_width, crop_height))
    work_width = max(1, round(crop_width * scale))
    work_height = max(1, round(crop_height * scale))
    work_image = Image.fromarray(crop_source).resize((work_width, work_height), Image.Resampling.LANCZOS)
    work_mask = Image.fromarray(crop_mask.astype(np.uint8) * 255).resize(
        (work_width, work_height), Image.Resampling.LANCZOS
    )
    work_image.save(cloud_input_path)

    mask_rgba = np.full((work_height, work_width, 4), 255, dtype=np.uint8)
    mask_rgba[:, :, 3] = np.where(np.asarray(work_mask) > 128, 0, 255).astype(np.uint8)
    api_mask = Image.fromarray(mask_rgba, "RGBA")
    aspect = work_width / work_height
    output_size = "1536x1024" if aspect > 1.2 else "1024x1536" if aspect < 0.83 else "1024x1024"

    files = {
        "model": (None, model),
        "prompt": (None, prompt),
        "image": ("image.png", png_bytes(work_image), "image/png"),
        "mask": ("mask.png", png_bytes(api_mask), "image/png"),
        "n": (None, "1"),
        "quality": (None, "low"),
        "size": (None, output_size),
        "output_format": (None, "png"),
    }
    try:
        response = requests.post(
            OPENAI_EDIT_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files=files,
            timeout=240,
        )
        response.raise_for_status()
        payload = response.json()
        edited = Image.open(io.BytesIO(base64.b64decode(payload["data"][0]["b64_json"]))).convert("RGB")
    except requests.HTTPError as error:
        detail = "OpenAI 图像编辑请求失败"
        try:
            api_error = error.response.json().get("error", {}).get("message")
            if api_error:
                detail = api_error
        except Exception:
            pass
        raise HTTPException(status_code=502, detail=detail) from error
    except (requests.RequestException, KeyError, ValueError, OSError) as error:
        raise HTTPException(status_code=502, detail="无法完成 OpenAI 图像编辑") from error

    generated = np.asarray(edited.resize((crop_width, crop_height), Image.Resampling.LANCZOS), dtype=np.uint8)
    composited_crop = np.where(crop_mask[:, :, None], generated, crop_source)
    result = source.copy()
    result[y0:y1, x0:x1] = composited_crop
    return result, {
        "roi": [x0, y0, x1, y1],
        "source_size": [crop_width, crop_height],
        "cloud_input_size": [work_width, work_height],
        "cloud_output_size": output_size,
    }


def opencv_erase(source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Local, free content fallback: OpenCV Telea inpainting inside the mask.
    Used so the whole Ultra HDR path can be validated without an OpenAI key."""
    import cv2
    out = source.copy()
    mask_u8 = (mask.astype(np.uint8) * 255)
    for ch in range(3):
        out[:, :, ch] = cv2.inpaint(source[:, :, ch], mask_u8, 10, cv2.INPAINT_TELEA)
    return out


def rebuild_uhdr(
    hdr: np.ndarray,
    original_sdr: np.ndarray,
    erased_sdr: np.ndarray,
    mask: np.ndarray,
    width: int,
    height: int,
    job_dir: Path,
) -> Path:
    gainmap, _, _ = compute_luminance_gainmap(hdr, original_sdr)
    gainmap_erased = poisson_inpaint(gainmap, mask)
    sdr_linear = srgb_to_linear(erased_sdr.astype(np.float32) / 255.0)
    hdr_linear = np.clip(sdr_linear * gainmap_erased[:, :, None], 0.0, 1.0)
    rebuilt = hdr.copy()
    rebuilt[mask] = np.clip(linear_to_hlg(hdr_linear) * 1023.0, 0, 1023)[mask]

    hdr_raw = job_dir / "rebuilt_hdr.raw"
    sdr_raw = job_dir / "erased_sdr.raw"
    output = job_dir / "erased_uhdr.jpg"
    save_rgba1010102(rebuilt, hdr_raw, width, height)
    save_rgba8888(erased_sdr, sdr_raw, width, height)
    command = [
        str(ULTRAHDR_APP), "-m", "0", "-p", str(hdr_raw), "-y", str(sdr_raw),
        "-w", str(width), "-h", str(height), "-a", "5", "-b", "3", "-t", "1",
        "-C", "0", "-c", "0", "-M", "0", "-q", "100", "-Q", "100", "-z", str(output),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, cwd=str(ROOT_DIR))
    except (OSError, subprocess.CalledProcessError) as error:
        raise HTTPException(status_code=500, detail="SDR 已消除，但 Ultra HDR 重建失败") from error
    return output


@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    lasso: str = Form("[]"),
    expand: int = Form(0),
):
    started = time.perf_counter()
    try:
        lasso_points = json.loads(lasso)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail="圈选坐标格式错误") from error
    if not isinstance(lasso_points, list) or len(lasso_points) < 3:
        raise HTTPException(status_code=400, detail="请完整地圈住一个对象")
    if not 0 <= expand <= 64:
        raise HTTPException(status_code=400, detail="蒙版扩展必须在 0–64 px")

    try:
        source = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception as error:
        raise HTTPException(status_code=400, detail="无法读取图片") from error

    try:
        polygon = np.asarray(lasso_points, dtype=np.float32)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or not np.isfinite(polygon).all():
            raise ValueError
        polygon[:, 0] = np.clip(polygon[:, 0], 0, source.width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, source.height - 1)
        polygon_i32 = np.rint(polygon).astype(np.int32)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail="圈选坐标无效") from error

    lasso_mask = np.zeros((source.height, source.width), dtype=np.uint8)
    cv2.fillPoly(lasso_mask, [polygon_i32], 255)
    if cv2.countNonZero(lasso_mask) < 16:
        raise HTTPException(status_code=400, detail="圈选区域太小")

    # Turn the freehand loop into the two prompts SAM understands best: an
    # interior point and a tight extent. The loop itself is not used as a mask,
    # otherwise the model merely copies the hand-drawn shape.
    moments = cv2.moments(polygon_i32)
    seed_x = int(moments["m10"] / moments["m00"])
    seed_y = int(moments["m01"] / moments["m00"])
    if not lasso_mask[seed_y, seed_x]:
        distance = cv2.distanceTransform(lasso_mask, cv2.DIST_L2, 5)
        seed_y, seed_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    # Probe several positions inside the loop in one decoder pass. A loop around
    # a thin object can have background at its geometric center; multi-point
    # probing lets SAM find the intended foreground without extra user clicks.
    xs = np.linspace(polygon_i32[:, 0].min(), polygon_i32[:, 0].max(), 7)[1:-1]
    ys = np.linspace(polygon_i32[:, 1].min(), polygon_i32[:, 1].max(), 7)[1:-1]
    seeds = [(seed_x, seed_y)]
    seeds.extend(
        (int(x), int(y))
        for y in ys
        for x in xs
        if lasso_mask[int(y), int(x)]
    )
    seeds = list(dict.fromkeys(seeds))

    midline_points = []
    left, right = int(polygon_i32[:, 0].min()), int(polygon_i32[:, 0].max())
    for fraction in (0.25, 0.5, 0.75):
        x = int(left + (right - left) * fraction)
        if lasso_mask[seed_y, x]:
            midline_points.append((x, seed_y))
    positive_points = list(dict.fromkeys([(seed_x, seed_y), *midline_points]))
    boundary_indices = np.linspace(0, len(polygon_i32) - 1, min(8, len(polygon_i32)), dtype=int)
    negative_points = [tuple(map(int, polygon_i32[index])) for index in boundary_indices]
    guided_points = [*positive_points, *negative_points]
    guided_labels = [1] * len(positive_points) + [0] * len(negative_points)
    prompt_length = len(guided_points)
    prompt_points = [guided_points]
    prompt_labels = [guided_labels]
    prompt_seeds = [(seed_x, seed_y)]
    for x, y in seeds:
        prompt_points.append([(x, y)] * prompt_length)
        prompt_labels.append([1] + [-1] * (prompt_length - 1))
        prompt_seeds.append((x, y))

    model, processor = get_model()
    inputs = processor(
        images=source,
        input_points=[[[[float(x), float(y)] for x, y in points] for points in prompt_points]],
        input_labels=[[labels for labels in prompt_labels]],
        return_tensors="pt",
    )
    inputs = {name: value.to(DEVICE) if hasattr(value, "to") else value for name, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    low_res = outputs.pred_masks[0].detach().float().cpu().numpy() > 0
    scores = outputs.iou_scores[0].detach().float().cpu()
    # Choose the model candidate that best agrees with the user's lasso while
    # penalizing masks that spill over it. This avoids the "whole image is green"
    # failure mode seen with a loose bounding box.
    clip_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    allowed = cv2.dilate(lasso_mask, clip_kernel, iterations=1) > 0
    low_height, low_width = low_res.shape[-2:]
    low_lasso = cv2.resize(lasso_mask, (low_width, low_height), interpolation=cv2.INTER_NEAREST) > 0
    low_allowed = cv2.dilate(low_lasso.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    ranked: list[tuple[float, int, int]] = []
    for prompt_index in range(low_res.shape[0]):
        for candidate_index in range(low_res.shape[1]):
            raw = low_res[prompt_index, candidate_index]
            intersection = np.count_nonzero(raw & low_lasso)
            spill = np.count_nonzero(raw & ~low_allowed)
            agreement = intersection / max(1, np.count_nonzero(low_lasso))
            spill_ratio = spill / max(1, np.count_nonzero(raw))
            fills_loop_penalty = max(0.0, agreement - 0.75) * 2.0
            model_score = float(scores[prompt_index, candidate_index])
            merit = model_score - 2.0 * spill_ratio - fills_loop_penalty
            ranked.append((merit, prompt_index, candidate_index))
    _, best_prompt, best_index = max(ranked, key=lambda item: item[0])
    seed_x, seed_y = prompt_seeds[best_prompt]
    selected_logits = outputs.pred_masks[:, best_prompt : best_prompt + 1, best_index : best_index + 1].cpu()
    best_raw = processor.post_process_masks(selected_logits, inputs["original_sizes"].cpu())[0][0, 0].numpy().astype(bool)
    mask = (best_raw & allowed).astype(np.uint8) * 255

    # Keep a single object: select the connected component that contains the
    # interior seed, falling back to the largest component inside the lasso.
    component_count, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count > 1:
        selected = int(components[seed_y, seed_x])
        if selected == 0:
            selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(components == selected, 255, 0).astype(np.uint8)

    if expand:
        diameter = expand * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        mask = cv2.dilate(mask, kernel, iterations=1)

    buffer = io.BytesIO()
    Image.fromarray(mask).save(buffer, format="PNG", optimize=True)
    return {
        "mask": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "score": round(float(scores[best_prompt, best_index]), 4),
        "device": DEVICE.upper(),
        "model": MODEL_ID,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "width": source.width,
        "height": source.height,
    }


@app.post("/erase")
async def erase(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(""),
    model: str = Form("opencv"),
    api_key: str = Form(""),
):
    started = time.perf_counter()
    # model: "opencv" (local, free, no key) or an OpenAI image model (e.g. gpt-image-2)
    if model not in {"opencv", "gpt-image-1", "gpt-image-2", OPENAI_MODEL}:
        raise HTTPException(status_code=400, detail=f"未知内容引擎 {model}")

    image_bytes = await image.read()
    mask_bytes = await mask.read()
    try:
        base_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask_image = Image.open(io.BytesIO(mask_bytes)).convert("L").resize(base_image.size, Image.Resampling.NEAREST)
    except Exception as error:
        raise HTTPException(status_code=400, detail="无法读取原图或蒙版") from error

    width, height = base_image.size
    erase_mask = np.asarray(mask_image) > 128
    if not erase_mask.any():
        raise HTTPException(status_code=400, detail="蒙版是空的，请重新圈选")

    job_id = uuid.uuid4().hex
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    suffix = Path(image.filename or "input.jpg").suffix.lower()
    source_path = job_dir / ("input" + (suffix if suffix in {".jpg", ".jpeg"} else ".png"))
    source_path.write_bytes(image_bytes)
    (job_dir / "mask.png").write_bytes(mask_bytes)

    decoded = decode_uhdr(source_path, width, height, job_dir) if ULTRAHDR_APP.is_file() else None
    if decoded is None:
        original_sdr = np.asarray(base_image, dtype=np.uint8)
        hdr = None
    else:
        original_sdr, hdr = decoded
        original_sdr = np.clip(np.round(original_sdr), 0, 255).astype(np.uint8)

    if model == "opencv":
        # Local, free content fill — validate the whole Ultra HDR path without OpenAI.
        erased_sdr = opencv_erase(original_sdr, erase_mask)
        Image.fromarray(original_sdr).save(job_dir / "cloud_input.png")
        downscale = {
            "backend": "opencv",
            "source_size": [width, height],
            "cloud_input_size": [width, height],
            "cloud_output_size": "local",
        }
    else:
        key = api_key.strip() or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise HTTPException(status_code=400, detail="请填写 OpenAI API Key；或改用本地 OpenCV(免费)引擎")
        edit_prompt = prompt.strip() or (
            "Remove the entire object inside the transparent masked region. Reconstruct only the natural "
            "background that should be behind it, matching perspective, texture, lighting, color, and image detail. "
            "Do not add any new objects and do not modify anything outside the mask."
        )
        erased_sdr, downscale = openai_downscaled_edit(
            original_sdr,
            erase_mask,
            api_key=key,
            prompt=edit_prompt,
            model=model,
            cloud_input_path=job_dir / "cloud_input.png",
        )
    preview_path = job_dir / "erased_sdr.png"
    Image.fromarray(erased_sdr).save(preview_path)

    if hdr is not None:
        result_path = rebuild_uhdr(
            hdr,
            original_sdr.astype(np.float32),
            erased_sdr.astype(np.float32),
            erase_mask,
            width,
            height,
            job_dir,
        )
        result_kind = "Ultra HDR"
    else:
        result_path = preview_path
        result_kind = "SDR"

    base_url = f"/results/{job_id}"
    return {
        "ok": True,
        "job_id": job_id,
        "kind": result_kind,
        "preview_url": f"{base_url}/erased_sdr.png",
        "download_url": f"{base_url}/{result_path.name}",
        "cloud_input_url": f"{base_url}/cloud_input.png",
        "downscale": downscale,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "model": model,
        "gainmap_channels": 1 if hdr is not None else None,
    }
