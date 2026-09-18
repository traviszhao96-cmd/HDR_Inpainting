"""Replay the other five saved cases with a fixed generic prompt and seed."""
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
baseline = root / "test_runs/20260911-142508-8af598/results.json"
run = root / "test_runs" / (time.strftime("%Y%m%d-%H%M%S") + "-fill-gallery")
run.mkdir(parents=True)
results = []
for case in json.loads(baseline.read_text()):
    if case["result"]["job_id"] == "5a678dbc579c4aaeb7e5b672208c7067":
        continue
    print("START", case["source_name"], flush=True)
    cmd = [sys.executable, str(root / "backend/run_mask_probe.py"), "--engine", "fill", "--lora-only",
           "--url", "https://win-hm9ig3vhnaa.tailfff622.ts.net", "--result-dir",
           str(root / "results" / case["result"]["job_id"]), "--mask-grow", "16", "--prompt",
           "An unobstructed continuation of the surrounding background, matching its surfaces, textures, perspective, lighting and colors. No new foreground objects."]
    done = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(done.stdout, flush=True)
    dirs = [line[7:] for line in done.stdout.splitlines() if line.startswith("OUTPUT ")]
    results.append({"case": case["case"], "source_name": case["source_name"],
                    "output": dirs[-1] if dirs else None, "exit_code": done.returncode,
                    "log": done.stdout, "command": cmd})
    (run / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    if done.returncode:
        raise SystemExit("Stopped after failed case; inspect queue before resuming")
print("COMPLETE", run, flush=True)
