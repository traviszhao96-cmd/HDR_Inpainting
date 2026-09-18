# Church gainmap speed/quality experiment

Artifacts: `seg_ui/test_runs/20260915-154007-church-gainmap-sweep/REPORT.md`
and `comparison.png` in the same directory. Exact runs and parameters are in
`results.json`. Shared SDR run: `20260914-180407-mask-probe-a16a8c`.

|Method|Resolution|Steps|Generation + retrieval|Whole run|
|---|---|---|---|---|
|Fill + removal LoRA|384 x 576|16|45.36s|59.54s|
|Fill + removal LoRA|384 x 576|8|18.28s|30.22s|
|Fill + removal LoRA|256 x 384|8|12.66s|25.68s|

Smooth harmonic interpolation calculation alone took 0.84–0.87s, excluding
shared decoding and HDR encoding. It fills missing gain values from the
boundary; it is not Gaussian blur over the original people. Comparing its
calculation time to whole generative runtime is not an end-to-end speed ratio.
Only one sample per setting; first-run model loading and network variability
may inflate the 16-step timing. Do not generalize the observed timing ratio.

Visual review: 384 x 576 / 8 steps is close to the 16-step reference in this
scene. 256 x 384 / 8 steps breaks up gain structure around the seated figures.
Generated gain continues some floor illumination but can invent human-like
gain detail; the smooth baseline is bland but avoids such detail. All share
the same SDR people and shadow artifacts. No candidate resolves those errors.

Suggested next test setting: 384 x 576 / 8 steps, 10px working-mask expansion,
no Gaussian post-blur, single-channel output and boundary correction. The
production UI and default generation settings have not been changed by this
sweep. Preview comparisons use identical fixed exposure mapping, not a native
HDR display. Actual UHDR files are linked in the report for HDR evaluation.
