# Generative gainmap UI integration

- Default UI gainmap: FLUX.1 Fill Q4_K_S + removal LoRA v2, 8 steps, max side 576. Shape follows ROI (temple: 336x576).
- Scalar log-gain input/output; use existing finish_gain boundary correction. No extra generation mask dilation, no Gaussian image blur, no ControlNet or RGB guidance. This is experimental, not an HDR-trained model.
- SDR remains 20 steps by default; 12-step experimental option is exposed. Smooth gainmap remains an explicit alternative. No silent fallback from generative errors.
- Per-job timings.json and gain_generation/report.json separate preparation, upload, queue/inference, download and reconstruction. Workflow and raw model output are saved.

First integration test: results/7d2db076b5464db5b3e1f7007ad70222 (temple, SDR 12 steps). 145.98 seconds total: 0.77 prep, 94.21 SDR, 50.99 gain/encode. SDR upload 63.20, queue/inference 16.22, download 12.50. Gain upload 19.21, queue/inference 17.30, download 10.06. Network transfers dominated THIS run; this does not prove the cause of the earlier user-reported 120s run.

Upload optimization added after this test: lossless WebP source, PNG mask, concurrent uploads using separate sessions. Local pixel equality passed; temple input decreased from 600093 to 444272 bytes (26%). No resolution decrease. Network timings vary and a controlled speedup is not established.

Validation: Python compilation and 3 SDR composite tests pass. Frontend HTTP 200. Existing frontend lint failures remain (unknown fetch JSON types, img tags and accessibility role preferences); lint is not clean.

Second end-to-end run after upload changes: results/c847305dacc343788ef42b4e3628b6f3, 98.05s (SDR 12 steps + gain 8). Preparation 0.76s, SDR 66.91s, gain/encode 30.37s. SDR upload 35.96s, queue/inference 17.48s, download 11.15s; gain upload 5.22s, queue/inference 10.70s, download 10.28s. WebP input compatibility is verified by completed model execution. The observed improvement from 146 to 98s includes uncontrolled network/model-cache/seed variation, not solely compression. Remaining transfer time is about 63s; combined queue/inference about 28s. SDR seam and shadow artifacts remain visible; this integration does not fix them.
