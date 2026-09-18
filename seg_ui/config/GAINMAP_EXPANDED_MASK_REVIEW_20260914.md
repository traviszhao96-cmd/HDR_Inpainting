# Expanded generation mask experiment

Run: `seg_ui/test_runs/20260914-155406-gainmap-probe-a3c5ad`

FLUX.1 Fill Q4_K_S + removal LoRA, seed 20260914, same church SDR source.
Both no-control and Canny-control branches completed (39.96s and 37.69s).

Generation mask is dilated by 32 working pixels (512 x 768 crop); final
composite uses the original mask. Original mask contains 45,165 pixels;
generation mask contains 76,798 pixels. The expanded area is discarded.
No gain histogram matching or feathering is applied in this experiment.

Visual finding: the narrow bright outline remains in raw model output at the
expanded boundary, but is largely excluded by the smaller final mask. A broad
tonal seam remains, most visibly on the floor. Expanding the generation mask
addresses the boundary artifact, but does not ensure equal gain values across
the final composite boundary. Canny still gives little visible benefit.

Validation: generation mask contains the original mask; floating log-gain
outside the final mask is unchanged at crop resolution (runtime assertion).
SDR and decoded HDR raw pixels outside the upscaled original mask have zero
maximum error against the source. JPEG base-image verification passed. This
does not assert identical pixels after lossy UHDR encoding or certify HDR
display quality from an SDR preview.

Artifacts: `generation-mask.png`, `mask.png`, `no-control-raw.png`,
`no-control-gainmap-composited.png`, `no-control-uhdr.jpg`; matching edge-control
artifacts are saved alongside them. This is an isolated experiment, not a UI
production workflow change.

## Low-resolution run with corrected input

Run: `seg_ui/test_runs/20260914-161145-gainmap-probe-04c727`.
Generation: 256 x 384, 12 steps, 10px dilation at the 512 x 768 working scale.
Postprocess: single-channel grayscale, Gaussian sigma 5 working pixels,
harmonic boundary residual correction and a 10px inward transition. Gainmap
JPEG quality is not reduced: low-frequency content is intentional, compression
artifacts are not. SDR generation remains the saved earlier result.

The prior TELEA float log-gain prefill caused overshoot and clipped black/white
checkerboard input. It is replaced with harmonic log-gain continuation and a
finite/range guard. The earlier low-resolution run `20260914-160707-gainmap-probe-8101d3`
still used TELEA; this run keeps its generation parameters but corrects prefill.
Raw output no longer visibly shows the former sharp outline. This supports an
input-preprocessing contribution; it is not a complete isolation of every
latent/model effect. Broad generated luminance blobs remain after smoothing.

No-control and edge-control took 22.01s and 21.96s, versus ~38–40s in the earlier
512 x 768 / 20-step expanded-mask run (not a controlled performance benchmark).
Both final gainmap previews are mode L with finite floating gains, outside-mask
gain equality assertions passed, and JPEG base verification passed.
