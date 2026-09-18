"""Isolated FLUX Klein masked-sampling experiment, NOT a trained Fill model.

Uses saved SDR/mask artifacts, never changes the UI's default workflow.
Records the exact graph, seed, raw decoded output and a zero-mask control.
"""
import argparse
import io
import json
import time
import uuid
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter
if __package__:
    from .sdr_prompt import DEFAULT_SDR_PROMPT, removal_prompt
else:
    from sdr_prompt import DEFAULT_SDR_PROMPT, removal_prompt


def graph(image, mask, prompt, seed, prefix, width, height):
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}
    return {
        "1": node("LoadImage", image=image),
        "2": node("LoadImageMask", image=mask, channel="red"),
        "4": node("UNETLoader", unet_name="flux-2-klein-4b-fp8.safetensors", weight_dtype="default"),
        "5": node("CLIPLoader", clip_name="qwen_3_4b.safetensors", type="flux2", device="default"),
        "6": node("CLIPTextEncode", clip=["5", 0], text=prompt),
        "7": node("CLIPTextEncode", clip=["5", 0], text=""),
        "8": node("VAELoader", vae_name="flux2-vae.safetensors"),
        "9": node("VAEEncode", pixels=["1", 0], vae=["8", 0]),
        "10": node("ReferenceLatent", conditioning=["7", 0], latent=["9", 0]),
        "11": node("ReferenceLatent", conditioning=["6", 0], latent=["9", 0]),
        "12": node("CFGGuider", model=["4", 0], positive=["11", 0], negative=["10", 0], cfg=1.0),
        "13": node("KSamplerSelect", sampler_name="euler"),
        "14": node("Flux2Scheduler", steps=4, width=width, height=height),
        # This node blanks the selected pixels before VAE encoding AND attaches
        # a noise_mask to the latent. It does not add trained mask conditioning.
        "15": node("VAEEncodeForInpaint", pixels=["1", 0], vae=["8", 0], mask=["2", 0], grow_mask_by=0),
        "16": node("RandomNoise", noise_seed=seed),
        "17": node("SamplerCustomAdvanced", noise=["16", 0], guider=["12", 0], sampler=["13", 0],
                   sigmas=["14", 0], latent_image=["15", 0]),
        "18": node("VAEDecode", samples=["17", 0], vae=["8", 0]),
        "19": node("SaveImage", filename_prefix=prefix, images=["18", 0]),
    }


def fill_graph(image, mask, prompt, seed, prefix, lora_strength=0.0):
    """Dedicated Fill conditioning: image + mask reach the diffusion model."""
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}
    workflow = {
        "1": node("LoadImage", image=image),
        "2": node("LoadImageMask", image=mask, channel="red"),
        "4": node("UnetLoaderGGUF", unet_name="flux1-fill-dev-Q4_K_S.gguf"),
        "5": node("DualCLIPLoader", clip_name1="t5xxl_fp8_e4m3fn.safetensors",
                  clip_name2="clip_l.safetensors", type="flux", device="default"),
        "6": node("CLIPTextEncode", clip=["5", 0], text=prompt),
        "7": node("CLIPTextEncode", clip=["5", 0], text=""),
        "8": node("VAELoader", vae_name="flux-vae-bf16.safetensors"),
        "9": node("FluxGuidance", conditioning=["6", 0], guidance=30.0),
        "10": node("InpaintModelConditioning", positive=["9", 0], negative=["7", 0],
                   vae=["8", 0], pixels=["1", 0], mask=["2", 0], noise_mask=True),
        "17": node("KSampler", model=["20", 0] if lora_strength else ["4", 0],
                   seed=seed, steps=20, cfg=1.0, sampler_name="euler", scheduler="normal",
                   positive=["10", 0], negative=["10", 1], latent_image=["10", 2], denoise=1.0),
        "18": node("VAEDecode", samples=["17", 0], vae=["8", 0]),
        "19": node("SaveImage", filename_prefix=prefix, images=["18", 0]),
    }
    if lora_strength:
        workflow["20"] = node("LoraLoaderModelOnly", model=["4", 0],
                              lora_name="removal_timestep_alpha-2-1740.safetensors",
                              strength_model=lora_strength)
    return workflow


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--result-dir", type=Path, required=True, help="Existing result with original_sdr.png and mask.png")
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--engine", choices=["klein", "fill"], default="klein")
    p.add_argument("--lora-only", action="store_true", help="Fill gallery smoke test without baseline")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--mask-grow", type=int, default=0, help="Experimental dilation radius in working pixels")
    p.add_argument("--prompt", default=DEFAULT_SDR_PROMPT)
    args = p.parse_args()
    args.prompt = removal_prompt(args.prompt)
    if not 0 <= args.mask_grow <= 64:
        p.error("--mask-grow must be between 0 and 64")
    url = args.url.rstrip("/")
    session = requests.Session()
    queue = session.get(url + "/queue", timeout=20)
    queue.raise_for_status()
    if queue.json().get("queue_running") or queue.json().get("queue_pending"):
        raise RuntimeError("ComfyUI is busy; no experiment submitted")
    source = Image.open(args.result_dir / "original_sdr.png").convert("RGB")
    mask = Image.open(args.result_dir / "mask.png").convert("L")
    if source.size != mask.size:
        raise ValueError("Image/mask dimensions differ")
    binary = np.asarray(mask) > 127
    ys, xs = np.nonzero(binary)
    if not len(xs):
        raise ValueError("Empty mask")
    padding = max(96, int(max(np.ptp(xs), np.ptp(ys))) + 1)
    box = (max(0, int(xs.min()) - padding), max(0, int(ys.min()) - padding),
           min(source.width, int(xs.max()) + padding + 1), min(source.height, int(ys.max()) + padding + 1))
    crop = source.crop(box)
    scale = min(1, 768 / max(crop.size))
    size = tuple(max(64, int(d * scale) // 16 * 16) for d in crop.size)
    crop = crop.resize(size, Image.Resampling.LANCZOS)
    mask = Image.fromarray(binary.astype(np.uint8) * 255).crop(box).resize(size, Image.Resampling.NEAREST)
    if args.mask_grow:
        mask = mask.filter(ImageFilter.MaxFilter(args.mask_grow * 2 + 1))
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-mask-probe-" + uuid.uuid4().hex[:6]
    out = Path(__file__).resolve().parents[1] / "test_runs" / run_id
    out.mkdir(parents=True)
    crop.save(out / "input.png")
    mask.save(out / "mask.png")
    manifest = {"experiment": ("FLUX Fill dedicated inpainting: baseline vs removal LoRA strength 1"
                               if args.engine == "fill" else "Klein generic masked latent sampling, not trained inpainting"),
                "source_result": str(args.result_dir.resolve()), "seed": args.seed, "roi": box,
                "size": size, "mask_grow_work_pixels": args.mask_grow, "prompt": args.prompt, "runs": []}
    print(f"OUTPUT {out}", flush=True)

    def upload(image, name):
        content = io.BytesIO()
        image.save(content, format="PNG")
        r = session.post(url + "/upload/image", files={"image": (name, content.getvalue(), "image/png")},
                         data={"type": "input", "overwrite": "false"}, timeout=60)
        r.raise_for_status()
        data = r.json()
        return "/".join(filter(None, [data.get("subfolder"), data["name"]]))

    image_name = upload(crop, run_id + ".png")
    outputs = []
    try:
        variants = ([("fill-baseline", mask), ("fill-removal-lora", mask)] if args.engine == "fill"
                    else [("selected-mask", mask), ("zero-mask-control", Image.new("L", size, 0))])
        if args.lora_only:
            if args.engine != "fill":
                raise ValueError("--lora-only requires --engine fill")
            variants = [("fill-removal-lora", mask)]
        for label, test_mask in variants:
            mask_name = upload(test_mask, run_id + "-" + label + ".png")
            prefix = "hdr_probe/" + run_id + "-" + label
            workflow = (fill_graph(image_name, mask_name, args.prompt, args.seed, prefix,
                                   1.0 if label == "fill-removal-lora" else 0.0)
                        if args.engine == "fill" else graph(image_name, mask_name, args.prompt, args.seed, prefix, *size))
            (out / (label + "-workflow.json")).write_text(json.dumps(workflow, indent=2))
            r = session.post(url + "/prompt", json={"prompt": workflow, "client_id": run_id}, timeout=30)
            if not r.ok:
                raise RuntimeError(f"Workflow rejected: {r.status_code} {r.text}")
            prompt_id = r.json()["prompt_id"]
            entry = {"label": label, "prompt_id": prompt_id}
            manifest["runs"].append(entry)
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"QUEUED {label} {prompt_id}", flush=True)
            start = time.monotonic()
            while time.monotonic() - start < args.timeout:
                r = session.get(url + "/history/" + prompt_id, timeout=30)
                r.raise_for_status()
                history = r.json().get(prompt_id)
                if history:
                    (out / (label + "-history.json")).write_text(json.dumps(history, indent=2))
                    if history.get("status", {}).get("status_str") == "error":
                        raise RuntimeError(f"Execution failed; inspect {label}-history.json")
                    images = history.get("outputs", {}).get("19", {}).get("images", [])
                    if images:
                        r = session.get(url + "/view", params=images[0], timeout=60)
                        r.raise_for_status()
                        raw = Image.open(io.BytesIO(r.content)).convert("RGB")
                        if raw.size != size:
                            raise ValueError(f"Unexpected output size {raw.size} != {size}")
                        raw.save(out / (label + "-raw.png"))
                        outputs.append(raw)
                        entry["elapsed_seconds"] = round(time.monotonic() - start, 2)
                        print(f"DONE {label} {entry['elapsed_seconds']}s", flush=True)
                        break
                time.sleep(2)
            else:
                raise TimeoutError(f"Prompt {prompt_id} still pending; not cancelled")
        montage = Image.new("RGB", (size[0] * (2 + len(outputs)), size[1] + 30), "#222222")
        labels = (["Input", "Mask", "Fill RAW", "Fill + removal LoRA RAW"] if args.engine == "fill"
                  else ["Input", "Mask", "Masked sampling RAW", "Zero-mask RAW"])
        if args.lora_only:
            labels = ["Input", "Mask", "Fill + removal LoRA RAW"]
        for i, (label, im) in enumerate(zip(labels, [crop, mask.convert("RGB"), *outputs])):
            montage.paste(im, (i * size[0], 30))
            ImageDraw.Draw(montage).text((i * size[0] + 8, 8), label, fill="white")
        montage.save(out / "comparison.png")
        if len(outputs) == 2:
            difference = np.abs(np.asarray(outputs[0], dtype=float) - np.asarray(outputs[1], dtype=float)).mean(axis=2)
            selected = np.asarray(mask) > 127
            manifest["raw_ab_difference_mae_0_255"] = {"inside": float(difference[selected].mean()), "outside": float(difference[~selected].mean())}
    except Exception as error:
        manifest["error"] = str(error)
        raise
    finally:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
