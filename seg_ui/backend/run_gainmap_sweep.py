"""Church gainmap comparison: fixed SDR, prompt, seed, mask and postprocess."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from hdr_ai_erase import load_rgba1010102, hlg_to_linear


def linear_to_srgb(x):
    return np.where(x<=0.0031308,12.92*x,1.055*np.maximum(x,0)**(1/2.4)-0.055)


def main():
    out = ROOT / "seg_ui/test_runs" / (time.strftime("%Y%m%d-%H%M%S")+"-church-gainmap-sweep")
    out.mkdir()
    rgb = ROOT / "seg_ui/test_runs/20260914-180407-mask-probe-a16a8c"
    env = dict(os.environ)
    env["NO_PROXY"] = "win-hm9ig3vhnaa.tailfff622.ts.net,localhost,127.0.0.1"
    rows = []
    print("SWEEP",out,flush=True)
    for name, side, steps in [("reference",576,16),("fewer-steps",576,8),("smaller",384,8)]:
        start = time.monotonic()
        command = [sys.executable,str(ROOT/"seg_ui/backend/run_gainmap_probe.py"),
                   "--url","https://win-hm9ig3vhnaa.tailfff622.ts.net", "--rgb-run",str(rgb),
                   "--removal-lora","--no-control-only","--generation-max-side",str(side),
                   "--steps",str(steps),"--smooth-sigma","0","--generation-grow","10"]
        done = subprocess.run(command,env=env,text=True,capture_output=True)
        print(done.stdout,flush=True)
        (out/(name+".log")).write_text(done.stdout+done.stderr)
        if done.returncode:
            raise RuntimeError(f"Run {name} failed; inspect {out/name}.log. No automatic resubmission.")
        run = Path(next(line[7:] for line in done.stdout.splitlines() if line.startswith("OUTPUT ")))
        meta = json.loads((run/"manifest.json").read_text())
        rows.append({"name":name,"run":str(run),"size":meta["generation_size"],"steps":steps,
                     "generation_seconds":meta["runs"][0]["elapsed_seconds"],
                     "wall_seconds":round(time.monotonic()-start,2),
                     "baseline_seconds":meta["smooth_baseline_seconds"]})
        (out/"results.json").write_text(json.dumps(rows,indent=2))

    ref = Path(rows[0]["run"])
    rgbmeta=json.loads((rgb/"manifest.json").read_text())
    w,h=Image.open(ref/"edited-sdr.png").size
    x0,y0,x1,y1=rgbmeta["roi"]
    mask=np.array(Image.open(ref/"mask.png"))>127
    lo,hi=json.loads((ref/"manifest.json").read_text())["range"]
    candidates=[("Smooth interpolation",ref,"poisson")]+[(r["name"],Path(r["run"]),"no-control") for r in rows]
    panels=[]
    for name,run,label in candidates:
        gain=np.load(run/(label+"-log-gain.npy"))
        hdr=load_rgba1010102(run/(label+"-hdr.raw"),w,h)[y0:y1,x0:x1]
        linear=hlg_to_linear(hdr/1023.)
        # Shared fixed-exposure display mapping so gain differences are visible.
        exposed=linear*4
        preview=linear_to_srgb(exposed/(1+exposed))
        preview=Image.fromarray(np.rint(np.clip(preview,0,1)*255).astype("uint8")).resize((512,768))
        gray=Image.fromarray(np.rint(np.clip((gain-lo)/(hi-lo),0,1)*255).astype("uint8")).convert("RGB")
        gray=gray.resize((512,768))
        panel=Image.new("RGB",(512,1566),"#161616");d=ImageDraw.Draw(panel)
        d.text((8,8),name,fill="white");panel.paste(gray,(0,30));panel.paste(preview,(0,798))
        panels.append(panel)
        preview.save(out/(name.replace(" ","-")+"-hdr-display.png"))
        if label!="poisson":
            baseline=np.load(ref/"poisson-log-gain.npy")
            row=next(r for r in rows if r["name"]==name)
            row["gain_mae_stops_vs_smooth"]=float(np.abs(gain-baseline)[mask].mean())
    canvas=Image.new("RGB",(512*len(panels),1566))
    for i,panel in enumerate(panels):canvas.paste(panel,(i*512,0))
    canvas.save(out/"comparison.png")
    (out/"results.json").write_text(json.dumps(rows,indent=2))
    report=["# Church gainmap parameter comparison","",
            "Same saved SDR, mask, seed and prompt. Generation buffer 10 working pixels; no Gaussian blur; identical single-channel boundary correction.",
            "Smooth baseline is harmonic interpolation in linear gain (not a literal Gaussian blur of the old subject).",
            "", "|Variant|Model input|Steps|Generation + retrieval (s)|Whole run (s)|", "|---|---|---|---|---|"]
    for row in rows:report.append(f"|{row['name']}|{row['size']}|{row['steps']}|{row['generation_seconds']}|{row['wall_seconds']}|")
    report += ["",f"Smooth baseline calculation: {rows[0]['baseline_seconds']}s (excludes decoding/encoding).",
               "One run per setting; timings include network and model residency effects, not a benchmark.",
               "comparison.png: top = shared-scale gainmap, bottom = fixed exposure Reinhard + sRGB display preview. Bottom is not native HDR rendering.",
               "UHDR files share the same SDR base, so SDR-only viewers cannot demonstrate HDR gain differences.",""]
    for name,run,label in candidates:report.append(f"- {name}: [{label}-uhdr.jpg]({run/(label+'-uhdr.jpg')})")
    (out/"REPORT.md").write_text("\n".join(report))
    print("COMPLETE",out,flush=True)


if __name__=="__main__":main()
