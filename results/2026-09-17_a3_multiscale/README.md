# A3-M: multiscale degradation reasoning

Sources: `artifacts/kaggle_downloads/12h30_17_09_2026/`. Both ZIP files pass
CRC checks. The run used clean commit `899ac92`, the same seed-42 split and
schedule as A3-L, complete SIDD32 initialization (664/664 backbone tensors),
two Tesla T4 GPUs, DDP, AMP and full-frame validation. It completed 20 epochs
in 27.46 minutes. Peak training allocation was 1,797 MB per GPU.

## Reasoning result

| Checkpoint | Epoch | PSNR | SSIM | Micro-F1 | Macro-F1 | Exact match |
|---|---:|---:|---:|---:|---:|---:|
| Best PSNR | 14 | 21.22805 | 0.726550 | 0.7583 | 0.7430 | 20.0% |
| Best reasoning | 17 | 20.90378 | 0.730657 | 0.7767 | 0.7594 | 29.1% |

At the best-reasoning checkpoint:

| Label | F1 | AUROC | Predicted positive / actual positive |
|---|---:|---:|---:|
| low | 0.949 | 0.981 | 29 / 30 |
| haze | 0.929 | 0.997 | 26 / 30 |
| rain | 0.632 | 0.790 | 18 / 20 |
| snow | 0.528 | **0.592** | 33 / 20 |

A3-M removes the zero-F1 failure and exceeds the macro-F1 target. Rain is now
meaningfully ranked. Snow is not: its mean predicted probability is 0.534 on
positive samples and 0.510 on negatives, with heavily overlapping ranges.
Balanced BCE shifts many samples above 0.5, producing nonzero F1, but does not
create adequate snow discrimination. A3-M therefore still fails the reasoning
gate requiring every label AUROC to be approximately 0.70 or higher and
probabilities to reflect the actual degradation components.

## Restoration result

| Comparison at best PSNR | Delta PSNR | Delta SSIM | PSNR scene wins | Paired scene p (PSNR) |
|---|---:|---:|---:|---:|
| A3-M vs A0-L | -0.06784 dB | +0.004593 | 3/5 | 0.619 |
| A3-M vs A2-L | -0.02427 dB | +0.000443 | 4/5 | 0.885 |
| A3-M vs A3-L | +0.12569 dB | +0.002240 | 4/5 | 0.490 |

No comparison is statistically decisive on five scenes, and A3-M does not
meet the predeclared improvement target of approximately +0.2 dB over the
matched baseline with non-decreasing SSIM. A0-L remains the highest-PSNR run.

Relative to A0-L, A3-M is worse on pure rain (-0.656 dB), pure snow
(-0.575 dB), low-rain (-0.633 dB) and low-snow (-0.297 dB). It remains below
the degraded input on rain scene 00059 (-0.392 dB), snow scene 00059
(-4.224 dB), and snow scene 00127 (-1.952 dB). Snow SSIM is also below input
on scenes 00059 and 00127.

The panels agree with the metrics. Rain streaks remain visible. Large snow
particles become dark or blurred spots, and `low_haze_snow/00059` contains
strong speckle and color artifacts. The higher SSIM is partly consistent with
smoothing and does not demonstrate successful precipitation removal.

## Decision

A3-M fails the complete feasibility criteria: it partially improves the
classifier but does not reliably separate snow and does not improve restoration
over matched controls. Under the predeclared stopping rule, stop the current
one-pass bottleneck/skip-modulation degradation-reasoning branch. Do not run
A4/A5, tune thresholds or loss weights on this validation split, or expand this
configuration to multiple seeds/full CDD-11.

This result rejects the current NAFNet-CoT adapter design on this pilot setup;
it does not reject all reasoning-based restoration. Any future sequential or
mixture-of-experts design should be treated as a new hypothesis with new
controls rather than another rescue ablation.
