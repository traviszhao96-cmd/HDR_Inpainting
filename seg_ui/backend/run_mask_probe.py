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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--result-dir", type=Path, required=True, help="Existing result with original_sdr.png and mask.png")
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--mask-grow", type=int, default=0, help="Experimental dilation radius in working pixels")
    p.add_argument("--prompt", default="Remove both walking people in the foreground on the right. Reconstruct the stone floor and wall behind them. Preserve the seated people in the background, architecture, colored light and perspective.")
    args = p.parse_args()
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
    manifest = {"experiment": "Klein generic masked latent sampling, not trained inpainting",
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
        for label, test_mask in [("selected-mask", mask), ("zero-mask-control", Image.new("L", size, 0))]:
            mask_name = upload(test_mask, run_id + "-" + label + ".png")
            workflow = graph(image_name, mask_name, args.prompt, args.seed, "hdr_probe/" + run_id + "-" + label, *size)
            (out / (label + "-workflow.json")).write_text(json.dumps(workflow, indent=2))
            r = session.post(url + "/prompt", json={"prompt": workflow, "client_id": run_id}, timeout=30)
            if not r.ok:
                raise RuntimeError(f"Workflow rejected: {r.status_code} {r.text}")
            prompt_id = r.json()["prompt_id"]
            entry = {"label": label, "prompt_id": prompt_id}
            manifest["runs"].append(entry)
            print(f"QUEUED {label} {prompt_id}", flush=True)
            start = time.monotonic()
            while time.monotonic() - start < 300:
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
        montage = Image.new("RGB", (size[0] * 4, size[1] + 30), "#222222")
        for i, (label, im) in enumerate(zip(["Input", "Mask", "Masked sampling RAW", "Zero-mask RAW"], [crop, mask.convert("RGB"), *outputs])):
            montage.paste(im, (i * size[0], 30))
            ImageDraw.Draw(montage).text((i * size[0] + 8, 8), label, fill="white")
        montage.save(out / "comparison.png")
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
