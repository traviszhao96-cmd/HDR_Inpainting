"""Experimental gainmap synthesis with RGB-derived edges; not HDR-trained.

Compares no ControlNet against weak Canny control with an identical seed.
Generation uses a larger mask than the final composite; the outer ring is discarded.
"""
import argparse
import io
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw
if __package__:
    from .run_mask_probe import fill_graph
else:
    from run_mask_probe import fill_graph

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hdr_ai_erase import (compute_luminance_gainmap, load_rgba1010102,
                         linear_to_hlg, srgb_to_linear, save_rgba1010102, save_rgba8888,
                         poisson_inpaint)

GAIN_PROMPT = (
    "Fill the masked region with an empty, unoccupied background represented as a "
    "monochrome grayscale illumination gain map. Continue the surrounding background "
    "surfaces, their geometry and broad illumination gradients naturally through the "
    "entire masked area. The filled area contains background surfaces only. "
    "Do not generate or reconstruct any people, faces, heads, bodies, limbs, human "
    "silhouettes, portraits or human-shaped shadows inside the masked region. "
    "Do not extend nearby people into the masked area. No new objects, color or text. "
    "Match the surrounding grayscale brightness at the boundary."
)


def finish_gain(raw, original, mask, size, low, high, sigma, feather):
    """Single-channel low-frequency proposal with a Dirichlet boundary correction."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    gray = np.asarray(raw.convert("L"), dtype=np.float32) / 255
    proposal = cv2.resize(gray, size, interpolation=cv2.INTER_LINEAR) * (high-low) + low
    if sigma > 0:
        proposal = cv2.GaussianBlur(proposal, (0, 0), sigma)
    # Harmonic continuation fixes values on the surrounding real gainmap.
    # Solve both the background and proposal residual with the same operator.
    yy, xx = np.nonzero(mask)
    ids = np.full(mask.shape, -1, dtype=np.int32)
    ids[mask] = np.arange(len(yy))
    rows, cols, data = [], [], []
    rhs = np.zeros((len(yy), 2), dtype=np.float64)
    degree = np.zeros(len(yy))
    residual = original - proposal
    for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
        ny, nx = yy+dy, xx+dx
        valid = (ny>=0)&(ny<mask.shape[0])&(nx>=0)&(nx<mask.shape[1])
        i = np.flatnonzero(valid); ny, nx = ny[valid], nx[valid]
        degree[i] += 1
        j = ids[ny,nx]; inside = j >= 0
        rows.extend(i[inside]); cols.extend(j[inside]); data.extend([-1.0]*int(inside.sum()))
        rhs[i[~inside],0] += original[ny[~inside],nx[~inside]]
        rhs[i[~inside],1] += residual[ny[~inside],nx[~inside]]
    rows.extend(range(len(yy))); cols.extend(range(len(yy))); data.extend(degree)
    matrix = coo_matrix((data,(rows,cols)),shape=(len(yy),len(yy))).tocsc()
    solutions = spsolve(matrix, rhs)
    background = original.copy(); background[mask] = solutions[:,0]
    aligned = proposal.copy(); aligned[mask] += solutions[:,1]
    dist = cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
    alpha = np.clip((dist-1)/max(feather,1),0,1)
    alpha = alpha*alpha*(3-2*alpha)
    mixed = original.copy()
    mixed[mask] = np.clip((background*(1-alpha)+aligned*alpha)[mask],low,high)
    return mixed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rgb-run", type=Path, required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--baseline-run", type=Path, help="Reuse a completed no-control run with identical source and seed")
    p.add_argument("--reuse-run", type=Path, help="Reprocess both saved model outputs without network or generation")
    p.add_argument("--no-control-only", action="store_true", help="Only evaluate the no-ControlNet branch")
    p.add_argument("--removal-lora", action="store_true", help="Additional exploratory ablation, not HDR-trained")
    p.add_argument("--blend-radius", type=int, default=0,
                   help="Reserved: use 0 for the isolated expanded-generation experiment")
    p.add_argument("--generation-grow", type=int, default=10,
                   help="Dilate only the model mask, in working pixels")
    p.add_argument("--generation-max-side", type=int, default=576)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--smooth-sigma", type=float, default=0.0)
    p.add_argument("--transition", type=int, default=10)
    args = p.parse_args()
    if args.blend_radius != 0:
        p.error("Use --blend-radius 0: the generated outer ring must be discarded")
    if not 0 <= args.generation_grow <= 96:
        p.error("--generation-grow must be between 0 and 96")
    if not 128 <= args.generation_max_side <= 1024 or not 1 <= args.steps <= 50 or not 0 <= args.smooth_sigma <= 32 or not 1 <= args.transition <= 96:
        p.error("Invalid generation or smoothing parameters")
    session = requests.Session()
    url = args.url.rstrip("/")
    if not args.reuse_run:
        r = session.get(url + "/queue", timeout=20); r.raise_for_status()
        if r.json().get("queue_running") or r.json().get("queue_pending"):
            raise RuntimeError("ComfyUI busy; gainmap experiment not submitted")
    meta = json.loads((args.rgb_run / "manifest.json").read_text())
    source_dir = Path(meta["source_result"])
    source = np.array(Image.open(source_dir / "original_sdr.png").convert("RGB"))
    height, width = source.shape[:2]
    hdr = load_rgba1010102(source_dir / "decoded_hdr.raw", width, height)
    gain, _, _ = compute_luminance_gainmap(hdr, source.astype(np.float32))
    log_gain = np.log2(gain)
    low, high = float(log_gain.min()), float(log_gain.max())
    if high - low < 1e-6:
        raise ValueError("Constant gainmap: no meaningful synthesis test")
    x0, y0, x1, y1 = meta["roi"]
    size = tuple(meta["size"])
    ratio = min(1.0, args.generation_max_side/max(size))
    generation_size = tuple(max(32, int(round(v*ratio/32))*32) for v in size)
    mask_im = Image.open(args.rgb_run / "mask.png").convert("L")
    mask = np.array(mask_im) > 127
    if not mask.any() or mask.all():
        raise ValueError("Final mask must contain both editable and known pixels")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2*args.generation_grow+1, 2*args.generation_grow+1))
    generation_mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    generation_mask_im = Image.fromarray(generation_mask.astype(np.uint8)*255)
    rgb_in = np.array(Image.open(args.rgb_run / "input.png").convert("RGB"))
    rgb_gen = np.array(Image.open(args.rgb_run / "fill-removal-lora-raw.png").convert("RGB"))
    rgb_edit = np.where(mask[..., None], rgb_gen, rgb_in)
    guide = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(rgb_edit, cv2.COLOR_RGB2GRAY), (0, 0), 1.2), 80, 160)
    # One fixed log-gain scale for both branches; preserve full precision outside mask.
    crop_log = cv2.resize(log_gain[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA)
    # Do not expose the old subject's gain structure to the generator. The
    # model gets the true gain only outside the edit mask; inside it receives a
    # smooth boundary continuation, while the edited RGB supplies structure.
    # TELEA on the small floating log-gain scale produced alternating overshoot
    # that clipped into black/white checkerboards. Harmonic averaging keeps the
    # fill within the known boundary range without introducing texture.
    gain_input_log = poisson_inpaint(crop_log, generation_mask)
    if not np.isfinite(gain_input_log).all() or gain_input_log.min() < low-1e-5 or gain_input_log.max() > high+1e-5:
        raise ValueError("Prefill is non-finite or outside the original gain range")
    preview = np.rint(np.clip((gain_input_log - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-gainmap-probe-" + uuid.uuid4().hex[:6]
    out = ROOT / "seg_ui/test_runs" / run_id
    out.mkdir(parents=True)
    Image.fromarray(preview).save(out / "gainmap-input.png")
    Image.fromarray(np.rint(np.clip((crop_log-low)/(high-low),0,1)*255).astype(np.uint8)).save(out / "gainmap-original-crop.png")
    Image.fromarray(rgb_edit).save(out / "rgb-guide.png")
    Image.fromarray(guide).save(out / "edge-guide.png")
    mask_im.save(out / "mask.png")
    generation_mask_im.save(out / "generation-mask.png")
    manifest = {"rgb_run": str(args.rgb_run.resolve()), "gain_domain": "log2 linear luminance ratio",
                "range": [low, high], "seed": 20260914, "control_strength": 0.35,
                "blend_radius": args.blend_radius,
                "generation_grow": args.generation_grow,
                "generation_size": generation_size, "steps": args.steps,
                "prefill": "harmonic-log-gain-v1",
                "prompt": GAIN_PROMPT,
                "prefill_range": [float(gain_input_log.min()), float(gain_input_log.max())],
                "smooth_sigma": args.smooth_sigma, "transition": args.transition,
                "postprocess": "grayscale; optional Gaussian (see smooth_sigma); harmonic boundary correction; inward smooth transition",
                "composite_policy": "original-mask-only; generated outer ring discarded",
                "removal_lora_strength": 1.0 if args.removal_lora else 0.0,
                "limitations": ["ordinary RGB model, not HDR trained", "8-bit proxy for generative input",
                                "single scene and seed", "original scalar gain is rederived, not extracted auxiliary JPEG"],
                "runs": []}
    print("OUTPUT", out, flush=True)
    def upload(im, suffix):
        b = io.BytesIO(); im.save(b, format="PNG")
        r = session.post(url + "/upload/image", files={"image": (run_id + suffix + ".png", b.getvalue(), "image/png")},
                         data={"type": "input", "overwrite": "false"}, timeout=60)
        r.raise_for_status(); d = r.json()
        return "/".join(filter(None, [d.get("subfolder"), d["name"]]))
    small_input = Image.fromarray(preview).resize(generation_size, Image.Resampling.BOX)
    small_mask = generation_mask_im.resize(generation_size, Image.Resampling.NEAREST)
    small_input.save(out / "model-input.png"); small_mask.save(out / "model-mask.png")
    if not args.reuse_run:
        gname = upload(small_input.convert("RGB"), "gain")
        mname = upload(small_mask, "mask")
        ename = upload(Image.fromarray(guide).resize(generation_size).convert("RGB"), "edges")
    prompt = GAIN_PROMPT
    variants = []
    try:
        branches = [("no-control", 0)] if args.no_control_only else [("no-control", 0), ("edge-control", 0.35)]
        for label, strength in branches:
            reuse = args.reuse_run or (args.baseline_run if label == "no-control" else None)
            if reuse:
                previous = json.loads((reuse / "manifest.json").read_text())
                if previous.get("prompt") != prompt:
                    raise ValueError("Baseline prompt differs; regenerate to test the new prompt")
                if previous.get("prefill") != manifest["prefill"]:
                    raise ValueError("Baseline prefill differs")
                if previous.get("generation_grow", 0) != args.generation_grow or tuple(previous.get("generation_size",size)) != generation_size or previous.get("steps",20) != args.steps:
                    raise ValueError("Baseline generation mask differs")
                if previous["rgb_run"] != manifest["rgb_run"] or previous["range"] != manifest["range"] or previous["seed"] != manifest["seed"] or previous.get("removal_lora_strength",0) != manifest["removal_lora_strength"]:
                    raise ValueError("Baseline provenance differs")
                raw = Image.open(reuse / (label+"-raw.png")).convert("RGB")
                if raw.size != generation_size: raise ValueError("Baseline size differs")
                raw.save(out / (label+"-raw.png"))
                raw.convert("L").save(out / (label+"-grayscale.png"))
                mixed = finish_gain(raw,crop_log,mask,size,low,high,args.smooth_sigma,args.transition)
                variants.append((label, mixed))
                np.save(out / (label+"-log-gain.npy"), mixed)
                manifest["runs"].append({"label": label, "reused_from": str(reuse.resolve())})
                continue
            if strength:
                deadline = time.monotonic() + 600
                while True:
                    r = session.get(url + "/object_info/ControlNetLoader", timeout=20); r.raise_for_status()
                    available = r.json()["ControlNetLoader"]["input"]["required"]["control_net_name"][0]
                    if "flux-canny-controlnet-v3.safetensors" in available: break
                    if time.monotonic() > deadline: raise TimeoutError("ControlNet download not ready")
                    print("WAIT ControlNet weight not ready", flush=True)
                    time.sleep(15)
            w = fill_graph(gname, mname, prompt, 20260914, "hdr_gain_probe/" + run_id + label,
                           1.0 if args.removal_lora else 0.0)
            w["17"]["inputs"]["steps"] = args.steps
            if strength:
                w["30"] = {"class_type": "LoadImage", "inputs": {"image": ename}}
                w["31"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "flux-canny-controlnet-v3.safetensors"}}
                w["32"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
                    "positive": ["10", 0], "negative": ["10", 1], "control_net": ["31", 0],
                    "image": ["30", 0], "strength": strength, "start_percent": 0.0, "end_percent": 0.7}}
                w["17"]["inputs"].update(positive=["32", 0], negative=["32", 1])
            (out / (label + "-workflow.json")).write_text(json.dumps(w, indent=2))
            r = session.post(url + "/prompt", json={"prompt": w, "client_id": run_id}, timeout=30)
            if not r.ok: raise RuntimeError(r.text)
            pid = r.json()["prompt_id"]
            entry = {"label": label, "prompt_id": pid}; manifest["runs"].append(entry)
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print("QUEUED", label, pid, flush=True)
            start = time.monotonic()
            while time.monotonic() - start < 900:
                r = session.get(url + "/history/" + pid, timeout=30); r.raise_for_status()
                h = r.json().get(pid)
                if h:
                    (out / (label + "-history.json")).write_text(json.dumps(h, indent=2))
                    if h.get("status", {}).get("status_str") == "error": raise RuntimeError("Execution error: " + label)
                    images = h.get("outputs", {}).get("19", {}).get("images", [])
                    if images:
                        r = session.get(url + "/view", params=images[0], timeout=60); r.raise_for_status()
                        raw = Image.open(io.BytesIO(r.content)).convert("RGB")
                        if raw.size != generation_size: raise ValueError("Generated dimensions changed")
                        raw.save(out / (label + "-raw.png"))
                        entry["elapsed_seconds"] = round(time.monotonic() - start, 2)
                        break
                time.sleep(2)
            else: raise TimeoutError("Prompt still running: " + pid)
            values = np.array(raw).astype(np.float32)
            entry["raw_chroma_mean"] = float(np.ptp(values, axis=2).mean())
            raw.convert("L").save(out / (label+"-grayscale.png"))
            mixed = finish_gain(raw,crop_log,mask,size,low,high,args.smooth_sigma,args.transition)
            variants.append((label, mixed))
            np.save(out / (label + "-log-gain.npy"), mixed)
            print("DONE", label, entry["elapsed_seconds"], flush=True)
        baseline_start = time.monotonic()
        smooth_gain = np.log2(np.maximum(poisson_inpaint(2 ** crop_log, mask), 1e-6))
        manifest["smooth_baseline_seconds"] = round(time.monotonic()-baseline_start, 3)
        np.save(out / "poisson-log-gain.npy", smooth_gain)
        variants.insert(0, ("poisson", smooth_gain))
        for label, values in variants:
            if label != "poisson":
                assert np.array_equal(values[~mask], crop_log[~mask]), "Outside gain changed"
            Image.fromarray(np.rint(np.clip((values-low)/(high-low),0,1)*255).astype(np.uint8)).save(out / (label+"-gainmap-composited.png"))
        # Same SDR composite for all HDR candidates; only gainmap differs.
        full_mask = cv2.resize(mask.astype(np.uint8), (x1-x0, y1-y0), interpolation=cv2.INTER_NEAREST).astype(bool)
        edited = source.copy().astype(np.float32)
        resized_rgb = cv2.resize(rgb_gen, (x1-x0, y1-y0), interpolation=cv2.INTER_LANCZOS4)
        rgb_alpha = full_mask.astype(np.float32)[..., None]
        roi = edited[y0:y1, x0:x1]
        roi[:] = roi * (1.0 - rgb_alpha) + resized_rgb.astype(np.float32) * rgb_alpha
        edited[y0:y1, x0:x1] = roi
        edited = np.rint(np.clip(edited, 0, 255)).astype(np.uint8)
        assert np.array_equal(edited[y0:y1,x0:x1][~full_mask], source[y0:y1,x0:x1][~full_mask])
        save_rgba8888(edited, out / "sdr.raw", width, height)
        Image.fromarray(edited).save(out / "edited-sdr.png")
        montage = Image.new("RGB", (size[0]*(2+len(variants)), size[1]+30), "#222222")
        for i, (label, im) in enumerate([("RGB guidance", Image.fromarray(rgb_edit)), ("Edges", Image.fromarray(guide)),
                    *[(label, Image.fromarray(np.rint(np.clip((v-low)/(high-low),0,1)*255).astype(np.uint8))) for label,v in variants]]):
            montage.paste(im.convert("RGB"), (i*size[0],30)); ImageDraw.Draw(montage).text((i*size[0]+5,8),label,fill="white")
        montage.save(out / "comparison.png")
        for label, values in variants:
            resized_gain = 2 ** cv2.resize(values, (x1-x0,y1-y0), interpolation=cv2.INTER_LINEAR)
            linear = srgb_to_linear(edited[y0:y1,x0:x1].astype(np.float32)/255) * resized_gain[...,None]
            candidate = hdr.copy()
            candidate[y0:y1,x0:x1][full_mask] = (linear_to_hlg(np.clip(linear,0,1))*1023)[full_mask]
            save_rgba1010102(candidate, out / (label+"-hdr.raw"), width,height)
            subprocess.run([str(ROOT/"build/ultrahdr_app"), "-m","0","-p",str(out/(label+"-hdr.raw")),
                            "-y",str(out/"sdr.raw"),"-w",str(width),"-h",str(height),"-a","5","-b","3",
                            "-t","1","-C","0","-c","0","-M","0","-q","100","-Q","100",
                            "-z",str(out/(label+"-uhdr.jpg"))], check=True, capture_output=True)
    except Exception as e:
        manifest["error"] = str(e); raise
    finally:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
