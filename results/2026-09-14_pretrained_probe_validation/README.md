# Pretrained validation probe — 2026-09-14

This is the first selection-safe zero-shot probe. All four official NAFNet
checkpoints were evaluated on the same five scene-disjoint validation scenes
from `CDD-11_train` (seed 42), with 55 samples per preset. `CDD-11_test` was not
used.

| Preset | PSNR | SSIM | Delta PSNR vs input | Delta SSIM vs input |
| --- | ---: | ---: | ---: | ---: |
| Input identity | 15.4670 | 0.5629 | — | — |
| gopro32 | 8.2692 | 0.3002 | -7.1979 | -0.2627 |
| gopro64 | 9.2662 | 0.3278 | -6.2008 | -0.2351 |
| sidd32 | **15.1067** | **0.5853** | -0.3603 | **+0.0224** |
| sidd64 | 14.9193 | 0.5735 | -0.5478 | +0.0106 |

`sidd32` is the provisional initialization candidate: it is the best restored
model on both metrics, is much smaller than `sidd64`, and does not require the
GoPro checkpoints' FP32 fallback. This is an initialization probe, not a
fine-tuned CDD-11 baseline.
