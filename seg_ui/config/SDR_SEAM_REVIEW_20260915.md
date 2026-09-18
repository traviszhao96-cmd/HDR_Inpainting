# SDR seam first pass — 2026-09-15

Scope: SDR only. Gainmap reconstruction and encoding were not changed.

- ComfyUI generation mask grows by 10 **source** pixels; final composite mask is unchanged.
- Local bounded RGB residual correction is estimated from known pixels outside the final mask. Only correction is smoothed, not image texture.
- A 6-source-pixel inward smoothstep transition preserves all pixels outside the mask exactly.
- Raw generated crop and both masks now persist in each ComfyUI result directory.

Validation: `python -m unittest backend.test_sdr_composite` (3 tests passed), Python compile, git diff whitespace check. Backend restarted on 7860.

Temple case: `test_runs/20260915-172345-fill-smooth-pipeline/results.json`, job `32c8d7df178140568fc558e6eb145145`, 71.506 seconds end-to-end.

`sdr-seam-comparison.png` compares old and new composite functions on the SAME new raw generation. Visual improvement is modest, not a completed seam fix. Paving/shadow discontinuities remain. A background person's legs were also extended into the generated region despite the no-people prompt. The prior run uses a different random seed; it is not a controlled test of mask expansion alone.

At this crop's 448x768 working size, 10 source pixels become roughly 2.4 working pixels. The buffer is intentionally narrow per the user's preference, but may be insufficient for latent-scale context. Do not claim the buffer solves geometric discontinuities.

Next candidates: controlled same-seed buffer comparison and gradient-domain boundary matching on saved raw output; neither has been validated yet. Shadows outside the mask need an explicit selection change, not blanket feathering. Avoid claiming all artifacts are color-only problems.

## Follow-up: no-buffer test

Per user request, removed the 10px generation buffer from the live ComfyUI path and restarted the backend. Color correction and inward blending remain unchanged. The pre-existing saved mask expansion (8px) was not changed; only the additional generation buffer was removed.

Recovered seed `7481916845727631017` by exact pixel comparison of the previous saved raw generation against ComfyUI history. Reran the same image, mask, prompt, model, seed, working dimensions and sampler settings. Output: `test_runs/20260915-183605-sdr-no-buffer/`, 61.87 seconds SDR-only. Gainmap was not rerun.

The new generation mask equals the composite mask; outside-mask pixels are identical to the source. The comparison still shows a conspicuous person-shaped seam, with a more noticeable dark/color edge in places in the no-buffer output. This single test does NOT support expansion as the sole cause or removal of expansion as a fix. Keep no-buffer as requested; next isolate compositing versus raw generation instead of adding more treatments simultaneously.
