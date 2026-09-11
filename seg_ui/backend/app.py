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
from fastapi.responses import FileResponse, Response
from PIL import Image
from transformers import Sam2Model, Sam2Processor
from .mask_ops import refine_mask

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
COMFYUI_URL = os.environ.get("COMFYUI_URL", "").rstrip("/")
COMFYUI_LAMA_MODEL = os.environ.get("COMFYUI_LAMA_MODEL", "big-lama.pt")
COMFYUI_CHECKPOINT = os.environ.get("COMFYUI_CHECKPOINT", "sd-v1-5-inpainting.safetensors")
COMFYUI_ENGINE = os.environ.get("COMFYUI_ENGINE", "sd15").lower()
COMFYUI_WORK_SIZE = int(os.environ.get("COMFYUI_WORK_SIZE", "768"))
COMFYUI_FLUX2_UNET = os.environ.get("COMFYUI_FLUX2_UNET", "flux-2-klein-4b-fp8.safetensors")
COMFYUI_FLUX2_CLIP = os.environ.get("COMFYUI_FLUX2_CLIP", "qwen_3_4b.safetensors")
COMFYUI_FLUX2_VAE = os.environ.get("COMFYUI_FLUX2_VAE", "flux2-vae.safetensors")
QWEN_VISION_API_KEY = os.environ.get("QWEN_VISION_API_KEY", "").strip()
QWEN_VISION_BASE_URL = os.environ.get("QWEN_VISION_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
QWEN_VISION_MODEL = os.environ.get("QWEN_VISION_MODEL", "qwen3-vl-plus")
QWEN_VISION_MAX_PIXELS = int(os.environ.get("QWEN_VISION_MAX_PIXELS", "1600000"))
QWEN_VISION_TIMEOUT = int(os.environ.get("QWEN_VISION_TIMEOUT", "90"))
ULTRAHDR_APP = ROOT_DIR / "build/ultrahdr_app"
DEFAULT_UHDR = ROOT_DIR / "output/clean_input_uhdr.jpg"
TEST_IMAGES_DIR = ROOT_DIR / "seg_ui/test_images"
TEST_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
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
        "comfyui_configured": bool(COMFYUI_URL),
        "qwen_vision_configured": bool(QWEN_VISION_API_KEY),
        "qwen_vision_model": QWEN_VISION_MODEL,
    }


def available_test_images() -> list[Path]:
    return sorted(
        (
            path
            for path in TEST_IMAGES_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in TEST_IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )


@app.get("/test-images")
def test_images():
    return {
        "items": [
            {"name": path.name, "url": f"/test-images/{path.name}/preview"}
            for path in available_test_images()
        ]
    }


@app.get("/test-images/{filename}")
def test_image(filename: str):
    if Path(filename).name != filename or Path(filename).suffix.lower() not in TEST_IMAGE_SUFFIXES:
        raise HTTPException(status_code=404, detail="测试图片不存在")
    path = TEST_IMAGES_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="测试图片不存在")
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/webp" if path.suffix.lower() == ".webp" else "image/jpeg"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/test-images/{filename}/preview")
def test_image_preview(filename: str):
    if Path(filename).name != filename or Path(filename).suffix.lower() not in TEST_IMAGE_SUFFIXES:
        raise HTTPException(status_code=404, detail="测试图片不存在")
    path = TEST_IMAGES_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="测试图片不存在")
    try:
        preview = Image.open(path).convert("RGB")
        buffer = io.BytesIO()
        preview.save(buffer, format="JPEG", quality=90, optimize=True)
    except OSError as error:
        raise HTTPException(status_code=400, detail="无法生成测试图预览") from error
    return Response(buffer.getvalue(), media_type="image/jpeg")


async def source_upload_bytes(image: UploadFile | None, test_image_name: str) -> tuple[bytes, str]:
    if test_image_name:
        if Path(test_image_name).name != test_image_name or Path(test_image_name).suffix.lower() not in TEST_IMAGE_SUFFIXES:
            raise HTTPException(status_code=400, detail="测试图片名称无效")
        path = TEST_IMAGES_DIR / test_image_name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="测试图片不存在")
        return path.read_bytes(), path.name
    if image is None:
        raise HTTPException(status_code=400, detail="请提供图片")
    return await image.read(), image.filename or "input.jpg"


@app.get("/sample-uhdr")
def sample_uhdr():
    samples = available_test_images()
    path = samples[0] if samples else DEFAULT_UHDR
    if not path.is_file():
        raise HTTPException(status_code=404, detail="默认 Ultra HDR 示例不存在")
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/webp" if path.suffix.lower() == ".webp" else "image/jpeg"
    return FileResponse(path, media_type=media_type, filename=path.name)


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
    composited_crop = composite_generated_crop(crop_source, generated, crop_mask)
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


def composite_generated_crop(source: np.ndarray, generated: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend generated pixels under the mask with a narrow soft edge."""
    mask_u8 = mask.astype(np.uint8) * 255
    feathered = cv2.GaussianBlur(mask_u8, (0, 0), sigmaX=2.5, sigmaY=2.5).astype(np.float32) / 255.0
    alpha = feathered[:, :, None]
    return np.clip(
        source.astype(np.float32) * (1.0 - alpha) + generated.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def comfyui_downscaled_edit(
    source: np.ndarray,
    mask: np.ndarray,
    *,
    prompt: str,
    cloud_input_path: Path,
    engine_override: str | None = None,
):
    """Run a masked local inpainting workflow on a remote ComfyUI server."""
    if not COMFYUI_URL:
        raise HTTPException(status_code=400, detail="后台未设置 COMFYUI_URL")
    comfy_engine = (engine_override or COMFYUI_ENGINE).lower()

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
    scale = min(1.0, COMFYUI_WORK_SIZE / max(crop_width, crop_height))
    work_width = max(64, (round(crop_width * scale) // 8) * 8)
    work_height = max(64, (round(crop_height * scale) // 8) * 8)
    work_image = Image.fromarray(crop_source).resize((work_width, work_height), Image.Resampling.LANCZOS)
    work_mask = Image.fromarray(crop_mask.astype(np.uint8) * 255).resize(
        (work_width, work_height), Image.Resampling.NEAREST
    )
    work_image.save(cloud_input_path)

    request_id = uuid.uuid4().hex
    image_name = f"hdr_erase_{request_id}.png"
    mask_name = f"hdr_erase_{request_id}_mask.png"
    session = requests.Session()

    def upload(name: str, image: Image.Image):
        response = session.post(
            f"{COMFYUI_URL}/upload/image",
            files={"image": (name, png_bytes(image), "image/png")},
            data={"type": "input", "overwrite": "true"},
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("name", name)

    try:
        uploaded_image = upload(image_name, work_image)
        uploaded_mask = upload(mask_name, work_mask)
        if comfy_engine == "lama":
            workflow = {
                "2": {"class_type": "LoadImage", "inputs": {"image": uploaded_image}},
                "3": {"class_type": "LoadImageMask", "inputs": {"image": uploaded_mask, "channel": "red"}},
                "8": {"class_type": "AUSBOSS_NODES_LaMaInpaint", "inputs": {
                    "image": ["2", 0], "mask": ["3", 0], "model": COMFYUI_LAMA_MODEL,
                }},
                "9": {"class_type": "SaveImage", "inputs": {
                    "filename_prefix": f"hdr_erase/{request_id}", "images": ["8", 0],
                }},
            }
            output_node = "9"
            engine_name = COMFYUI_LAMA_MODEL
        elif comfy_engine in {"flux2", "flux2-klein", "flux2_klein"}:
            # FLUX.2 Klein is a reference-image editor rather than a classic
            # masked inpaint model.  The generated crop is therefore composited
            # back only under the user's mask below, preserving all surroundings.
            positive_prompt = prompt.strip() or (
                "Remove the main foreground subject or object near the center of this crop. "
                "Reconstruct the background behind it using the visible surroundings as a guide, as if the target had never been there. "
                "Continue existing surfaces, edges, lines, textures, and color gradients naturally. "
                "Match the original perspective, lighting, colors, and level of detail. "
                "Preserve all other subjects and the surrounding scene. "
                "Do not replace the target with another object, invent new elements, or change the overall appearance of the photo."
            )
            workflow = {
                "1": {"class_type": "LoadImage", "inputs": {"image": uploaded_image}},
                "2": {"class_type": "ImageScaleToTotalPixels", "inputs": {
                    "image": ["1", 0], "upscale_method": "nearest-exact", "megapixels": 0.25,
                    "resolution_steps": 1,
                }},
                "3": {"class_type": "GetImageSize", "inputs": {"image": ["2", 0]}},
                "4": {"class_type": "UNETLoader", "inputs": {
                    "unet_name": COMFYUI_FLUX2_UNET, "weight_dtype": "default",
                }},
                "5": {"class_type": "CLIPLoader", "inputs": {
                    "clip_name": COMFYUI_FLUX2_CLIP, "type": "flux2", "device": "default",
                }},
                "6": {"class_type": "CLIPTextEncode", "inputs": {
                    "clip": ["5", 0], "text": positive_prompt,
                }},
                "7": {"class_type": "CLIPTextEncode", "inputs": {
                    "clip": ["5", 0], "text": "",
                }},
                "8": {"class_type": "VAELoader", "inputs": {"vae_name": COMFYUI_FLUX2_VAE}},
                "9": {"class_type": "VAEEncode", "inputs": {
                    "pixels": ["2", 0], "vae": ["8", 0],
                }},
                # ReferenceLatent is the native conditioning used by the
                # official ComfyUI Image Edit (Flux.2 Klein 4B) blueprint.
                "10": {"class_type": "ReferenceLatent", "inputs": {
                    "conditioning": ["7", 0], "latent": ["9", 0],
                }},
                "11": {"class_type": "ReferenceLatent", "inputs": {
                    "conditioning": ["6", 0], "latent": ["9", 0],
                }},
                "12": {"class_type": "CFGGuider", "inputs": {
                    "model": ["4", 0], "positive": ["11", 0], "negative": ["10", 0], "cfg": 5,
                }},
                "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
                "14": {"class_type": "Flux2Scheduler", "inputs": {
                    "steps": 4, "width": ["3", 0], "height": ["3", 1],
                }},
                "15": {"class_type": "EmptyFlux2LatentImage", "inputs": {
                    "width": ["3", 0], "height": ["3", 1], "batch_size": 1,
                }},
                "16": {"class_type": "RandomNoise", "inputs": {
                    "noise_seed": int(request_id[:16], 16),
                }},
                "17": {"class_type": "SamplerCustomAdvanced", "inputs": {
                    "noise": ["16", 0], "guider": ["12", 0], "sampler": ["13", 0],
                    "sigmas": ["14", 0], "latent_image": ["15", 0],
                }},
                "18": {"class_type": "VAEDecode", "inputs": {
                    "samples": ["17", 0], "vae": ["8", 0],
                }},
                "19": {"class_type": "SaveImage", "inputs": {
                    "filename_prefix": f"hdr_erase/{request_id}", "images": ["18", 0],
                }},
            }
            output_node = "19"
            engine_name = COMFYUI_FLUX2_UNET
        else:
            positive_prompt = prompt.strip() or (
                "empty natural background, seamless continuation of the surrounding scene, "
                "photorealistic, matching perspective, texture, lighting and color"
            )
            workflow = {
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": COMFYUI_CHECKPOINT}},
                "2": {"class_type": "LoadImage", "inputs": {"image": uploaded_image}},
                "3": {"class_type": "LoadImageMask", "inputs": {"image": uploaded_mask, "channel": "red"}},
                "4": {"class_type": "CLIPTextEncode", "inputs": {"text": positive_prompt, "clip": ["1", 1]}},
                "5": {"class_type": "CLIPTextEncode", "inputs": {
                    "text": "object, person, car, street lamp, pole, shadow, reflection, text, watermark, "
                            "blurry, artifacts, distorted, low quality",
                    "clip": ["1", 1],
                }},
                "6": {"class_type": "VAEEncodeForInpaint", "inputs": {
                    "pixels": ["2", 0], "vae": ["1", 2], "mask": ["3", 0], "grow_mask_by": 8,
                }},
                "7": {"class_type": "KSampler", "inputs": {
                    "model": ["1", 0], "seed": int(request_id[:16], 16), "steps": 28, "cfg": 7.0,
                    "sampler_name": "dpmpp_2m", "scheduler": "karras", "positive": ["4", 0],
                    "negative": ["5", 0], "latent_image": ["6", 0], "denoise": 1.0,
                }},
                "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
                "9": {"class_type": "SaveImage", "inputs": {
                    "filename_prefix": f"hdr_erase/{request_id}", "images": ["8", 0],
                }},
            }
            output_node = "9"
            engine_name = COMFYUI_CHECKPOINT
        queued = session.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow, "client_id": request_id},
            timeout=30,
        )
        queued.raise_for_status()
        prompt_id = queued.json()["prompt_id"]
        deadline = time.monotonic() + 240
        output_info = None
        while time.monotonic() < deadline:
            history_response = session.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=30)
            history_response.raise_for_status()
            history = history_response.json().get(prompt_id)
            if history:
                images = history.get("outputs", {}).get(output_node, {}).get("images", [])
                if images:
                    output_info = images[0]
                    break
                status = history.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError("ComfyUI 工作流执行失败")
            time.sleep(1)
        if output_info is None:
            raise TimeoutError("ComfyUI 生成超时")
        generated_response = session.get(
            f"{COMFYUI_URL}/view",
            params={
                "filename": output_info["filename"],
                "subfolder": output_info.get("subfolder", ""),
                "type": output_info.get("type", "output"),
            },
            timeout=60,
        )
        generated_response.raise_for_status()
        edited = Image.open(io.BytesIO(generated_response.content)).convert("RGB")
    except (requests.RequestException, KeyError, ValueError, OSError, RuntimeError, TimeoutError) as error:
        raise HTTPException(status_code=502, detail=f"ComfyUI 消除失败：{error}") from error

    generated = np.asarray(edited.resize((crop_width, crop_height), Image.Resampling.LANCZOS), dtype=np.uint8)
    composited_crop = composite_generated_crop(crop_source, generated, crop_mask)
    result = source.copy()
    result[y0:y1, x0:x1] = composited_crop
    return result, {
        "backend": "comfyui",
        "engine": engine_name,
        "roi": [x0, y0, x1, y1],
        "source_size": [crop_width, crop_height],
        "cloud_input_size": [work_width, work_height],
        "cloud_output_size": f"{work_width}x{work_height}",
    }


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
    # Shared log scale makes before/after brightness directly comparable.
    log_before = np.log2(np.maximum(gainmap, 1e-6))
    log_after = np.log2(np.maximum(gainmap_erased, 1e-6))
    low = float(min(log_before.min(), log_after.min()))
    high = float(max(log_before.max(), log_after.max()))
    for name, values in (("gainmap_before.png", log_before), ("gainmap_after.png", log_after)):
        preview = np.clip((values - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(preview).save(job_dir / name)
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
    image: UploadFile | None = File(None),
    lasso: str = Form("[]"),
    expand: int = Form(0),
    search_margin: int = Form(48),
    test_image: str = Form(""),
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
    if not 0 <= search_margin <= 256:
        raise HTTPException(status_code=400, detail="分割搜索边距必须在 0–256 px")

    try:
        image_bytes, _ = await source_upload_bytes(image, test_image)
        source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
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

    if search_margin:
        search_diameter = search_margin * 2 + 1
        search_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (search_diameter, search_diameter))
        search_mask = cv2.dilate(lasso_mask, search_kernel, iterations=1)
    else:
        search_mask = lasso_mask.copy()
    search_y, search_x = np.nonzero(search_mask)
    search_box = [
        int(search_x.min()),
        int(search_y.min()),
        int(search_x.max()),
        int(search_y.max()),
    ]

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
    box_left, box_top, box_right, box_bottom = search_box
    negative_points = [
        (box_left, box_top), ((box_left + box_right) // 2, box_top), (box_right, box_top),
        (box_left, (box_top + box_bottom) // 2), (box_right, (box_top + box_bottom) // 2),
        (box_left, box_bottom), ((box_left + box_right) // 2, box_bottom), (box_right, box_bottom),
    ]
    negative_points = [point for point in negative_points if not lasso_mask[point[1], point[0]]]
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
        input_boxes=[[search_box for _ in prompt_points]],
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
    allowed = search_mask > 0
    low_height, low_width = low_res.shape[-2:]
    low_lasso = cv2.resize(lasso_mask, (low_width, low_height), interpolation=cv2.INTER_NEAREST) > 0
    low_allowed = cv2.resize(search_mask, (low_width, low_height), interpolation=cv2.INTER_NEAREST) > 0
    low_lasso_area = max(1, np.count_nonzero(low_lasso))
    ranked: list[tuple[float, int, int]] = []
    for prompt_index in range(low_res.shape[0]):
        for candidate_index in range(low_res.shape[1]):
            raw = low_res[prompt_index, candidate_index]
            intersection = np.count_nonzero(raw & low_lasso)
            spill = np.count_nonzero(raw & ~low_allowed)
            spill_ratio = spill / max(1, np.count_nonzero(raw))
            overfill = max(0.0, np.count_nonzero(raw) / low_lasso_area - 1.5)
            misses_lasso = 1.0 if intersection == 0 else 0.0
            model_score = float(scores[prompt_index, candidate_index])
            merit = model_score - 2.5 * spill_ratio - 0.45 * min(overfill, 3.0) - misses_lasso
            ranked.append((merit, prompt_index, candidate_index))
    _, best_prompt, best_index = max(ranked, key=lambda item: item[0])
    seed_x, seed_y = prompt_seeds[best_prompt]
    selected_logits = outputs.pred_masks[:, best_prompt : best_prompt + 1, best_index : best_index + 1].cpu()
    best_raw = processor.post_process_masks(selected_logits, inputs["original_sizes"].cpu())[0][0, 0].numpy().astype(bool)
    mask = (best_raw & allowed).astype(np.uint8) * 255

    # One candidate and one connected subject per lasso.
    mask = refine_mask(mask, outward=expand, smooth=3, seed=(seed_x, seed_y))

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
        "search_margin": search_margin,
        "search_box": search_box,
    }


@app.post("/test-masks/save")
async def save_test_mask(
    mask: UploadFile = File(...), image: UploadFile | None = File(None),
    test_image: str = Form(""), prompt: str = Form(""),
    model: str = Form("comfyui-flux2"), mask_expand: int = Form(0),
):
    source_bytes, source_name = await source_upload_bytes(image, test_image)
    try:
        source = Image.open(io.BytesIO(source_bytes))
        saved_mask = Image.open(io.BytesIO(await mask.read())).convert('L')
        if saved_mask.size != source.size:
            raise HTTPException(400, '蒙版尺寸与原图不一致，请重新圈选')
        binary = (np.asarray(saved_mask) > 127).astype(np.uint8) * 255
        if not binary.any():
            raise HTTPException(400, '蒙版为空，无法保存')
    except (ValueError, OSError) as error:
        raise HTTPException(400, '无法读取图片或蒙版') from error
    # Immutable test-case snapshots: never replace an earlier hand-painted mask.
    case_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:12]
    directory = ROOT_DIR / 'seg_ui/test_masks' / case_id
    directory.mkdir(parents=True, exist_ok=False)
    suffix = Path(source_name).suffix.lower()
    source_file = 'source' + (suffix if suffix in TEST_IMAGE_SUFFIXES else '.img')
    (directory / source_file).write_bytes(source_bytes)
    Image.fromarray(binary).save(directory / 'mask.png')
    metadata = {
        'version': 1, 'source_name': source_name, 'test_image': test_image or None,
        'image': source_file, 'mask': 'mask.png', 'width': source.width, 'height': source.height,
        'mask_convention': 'white=erase, black=keep; before erase expansion',
        'model': model, 'prompt': prompt, 'mask_expand': max(0, min(mask_expand, 64)),
    }
    (directory / 'case.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'ok': True, 'path': str(directory), 'case_id': case_id}


@app.post("/mask/refine")
async def clean_mask(mask: UploadFile = File(...), inward: int = Form(0), outward: int = Form(2), smooth: int = Form(3)):
    try:
        source = np.asarray(Image.open(io.BytesIO(await mask.read())).convert('L'))
    except (ValueError, OSError) as error:
        raise HTTPException(400, '无法读取蒙版') from error
    result = refine_mask(source, inward, outward, smooth)
    if not result.any():
        raise HTTPException(400, '向内收缩过大，蒙版已为空，请减小参数')
    return Response(png_bytes(Image.fromarray(result)), media_type='image/png')


@app.post("/erase")
async def erase(
    image: UploadFile | None = File(None),
    mask: UploadFile = File(...),
    prompt: str = Form(""),
    model: str = Form("opencv"),
    api_key: str = Form(""),
    test_image: str = Form(""),
    mask_expand: int = Form(0),
):
    started = time.perf_counter()
    # model: local OpenCV, remote ComfyUI, or an OpenAI image model.
    if model not in {"opencv", "comfyui", "comfyui-flux2", "gpt-image-1", "gpt-image-2", OPENAI_MODEL}:
        raise HTTPException(status_code=400, detail=f"未知内容引擎 {model}")

    image_bytes, source_filename = await source_upload_bytes(image, test_image)
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
    mask_expand = max(0, min(int(mask_expand), 64))
    if mask_expand:
        diameter = mask_expand * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        erase_mask = cv2.dilate(erase_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    job_id = uuid.uuid4().hex
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    suffix = Path(source_filename).suffix.lower()
    source_path = job_dir / ("input" + (suffix if suffix in {".jpg", ".jpeg"} else ".png"))
    source_path.write_bytes(image_bytes)
    Image.fromarray(erase_mask.astype(np.uint8) * 255).save(job_dir / "mask.png")

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
    elif model in {"comfyui", "comfyui-flux2"}:
        erased_sdr, downscale = comfyui_downscaled_edit(
            original_sdr,
            erase_mask,
            prompt=prompt,
            cloud_input_path=job_dir / "cloud_input.png",
            engine_override="flux2" if model == "comfyui-flux2" else None,
        )
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
    Image.fromarray(original_sdr).save(job_dir / "original_sdr.png")
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
        "comparisons": {
            "sdr": [f"{base_url}/original_sdr.png", f"{base_url}/erased_sdr.png"],
            "gainmap": [f"{base_url}/gainmap_before.png", f"{base_url}/gainmap_after.png"] if hdr is not None else None,
            "hdr": [f"{base_url}/{source_path.name}", f"{base_url}/{result_path.name}"] if hdr is not None else None,
        },
        "downscale": downscale,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "model": model,
        "gainmap_channels": 1 if hdr is not None else None,
    }
