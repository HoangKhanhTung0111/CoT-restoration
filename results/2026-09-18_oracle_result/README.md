# Full-validation intervention-utility oracle audit

## Provenance

- Kaggle commit: `d3ee1766d2d6469a2f61f782dc7f37f850ce6f35`
- Model: A3-M, best-restoration checkpoint at epoch 14
- Split: the unchanged five-scene, 55-image validation split
- Inference: full frame, AMP, one GPU
- Raw lightweight export: `a3m_lightweight_results.zip`
- SHA-256: `1F54F9ED8C7D64F005E81CB779C9F31F5EB5C1B31D3ECB90629F53C9CB1A1567`

The oracle uses ground truth and is a diagnostic upper bound, not a deployable
method. The fixed residual strength is also selected on this evaluation split,
making it an optimistic simple control.

## Aggregate results

| Method | Mean PSNR | Mean SSIM | PSNR vs restored | Restored fraction | Images worse than input |
|---|---:|---:|---:|---:|---:|
| Restored output | 21.2280 | 0.726550 | -- | 1.000 | 3 |
| Best fixed beta (0.92) | 21.3374 | not recomputed | +0.1093 | 0.920 | 2 |
| Image oracle | 21.3475 | 0.728125 | +0.1194 | 0.945 | 0 |
| 32x32 oracle | 21.9145 | 0.727392 | +0.6864 | 0.877 | 0 |
| 8x8 oracle | 22.2407 | 0.727458 | +1.0127 | 0.841 | 0 |

The 32x32 oracle beats the optimistic fixed-beta control by 0.5771 dB. Its
mean SSIM is 0.000842 above the restored output, so the PSNR gain is not bought
with a mean SSIM decrease. The 8x8 oracle adds more PSNR but offers almost no
additional SSIM, and its finer hard boundaries will be harder to learn and
blend cleanly. The 32x32 operating resolution is therefore the primary target.

## Distribution of 32x32 headroom

Headroom over the restored output is positive for every degradation category,
but uneven: snow contributes +2.555 dB, haze +1.101 dB, haze-rain +0.915 dB,
whereas low contributes only +0.129 dB. Against fixed beta, the oracle remains
better in 9 of 11 categories; fixed beta is better for low (-0.190 dB) and
low-snow (-0.115 dB).

Grouped by source scene, 32x32 headroom over restoration is positive for all
five scenes: +2.019, +0.206, +0.463, +0.039, and +0.705 dB. Against fixed beta,
four scenes are positive and scene `00110` is negative by 0.360 dB. Thus the
aggregate result is not solely a single harmful image, but it is still based on
only five independent source scenes.

## Decision and meaning

The feasibility gate passes: there is material local intervention headroom
beyond global residual attenuation. This establishes that helpful and harmful
restoration changes coexist spatially in A3-M outputs. It does not establish
that the useful regions can be identified without ground truth.

The next experiment is a deployable 32x32 signed-gain predictor trained only on
training scenes, with calibration and evaluation separated by source scene. It
must beat global beta, image-level selection, and an uncalibrated gain head on
the quality-harm curve. Failure to capture a meaningful portion of oracle
headroom would reject the practical intervention-utility hypothesis even though
the oracle upper bound exists.
