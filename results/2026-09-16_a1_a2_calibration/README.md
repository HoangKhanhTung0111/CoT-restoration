# A1/A2 five-epoch architecture calibration

Both runs used the SIDD32 initialization, seed 42, 256-pixel training crops,
full-frame validation/evaluation and DDP on two Tesla T4 GPUs. All auxiliary
loss weights were zero; A1 and A2 differed only by skip gating.

| Run | PSNR | SSIM | Difference from A0 |
|---|---:|---:|---:|
| A1 bottleneck adapter | 19.6957 | 0.67122 | +0.1982 dB / -0.00001 |
| A2 adapter + skip gates | **19.7729** | **0.67494** | **+0.2754 dB / +0.00371** |

A2 is the provisional architecture selected for A3. It improved PSNR over A0
on 34/55 images and improved all five scene-level PSNR means, but the five-scene
95% confidence interval still includes zero. This is promising screening
evidence, not a generalization claim.

Rain improved by 0.8285 dB relative to A0. Snow improved by 0.2176 dB relative
to A0 but remained 0.6680 dB below the degraded input. Low-rain regressed by
0.2197 dB relative to A0. These categories are the main A3 diagnostic targets.

The raw image ZIP and original lightweight ZIP remain local under
`artifacts/kaggle_downloads/3h30_16_09_2026/`. Checkpoints and rendered images
are intentionally excluded from Git.

The archived `ablation_summary.csv` is preserved verbatim and uses the legacy
`test.*` column prefix even though its own `test.split` field is `validation`.
The summarizer was corrected after this run to emit the neutral
`evaluation.*` prefix and avoid implying that the held-out test split was used.
