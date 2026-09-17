# A0-L/A2-L: matched twenty-epoch controls

Sources: `artifacts/kaggle_downloads/10h50_17_09_2026/`. Both ZIP files pass
CRC checks. The lightweight archive contains complete 20-epoch logs and 55
full-frame validation rows per run; the comparison archive contains one panel
for each of 11 degradation types per run.

Both runs used clean commit `780993c`, the same seed-42 scene split as A3-L,
the same SIDD32 initialization (664/664 backbone tensors), two Tesla T4 GPUs,
DDP, AMP, crop/batch settings, optimizer and cosine schedule. Both selected
their best PSNR checkpoint at epoch 14.

| Run | PSNR | SSIM | Time (min) | Parameters |
|---|---:|---:|---:|---:|
| A0-L baseline | **21.29589** | 0.721956 | 23.38 | 29,159,715 |
| A2-L adapter + skip gates | 21.25232 | **0.726107** | 24.18 | 29,406,823 |
| A3-L + degradation supervision | 21.10236 | 0.724310 | 25.26 | 29,406,823 |

Relative to A0-L, A2-L changes PSNR by -0.04357 dB and SSIM by +0.004151.
It wins 25/55 images in PSNR and 31/55 in SSIM. Scene-mean PSNR improves in
3/5 scenes, but a paired scene-level test is not significant (`p=0.689`;
Wilcoxon `p=1.0`). The SSIM result is also not significant (`p=0.268`;
Wilcoxon `p=0.3125`). Thus there is no evidence that the current bottleneck
adapter and skip gates improve restoration.

Precipitation is worse with A2-L: relative to A0-L, rain changes by
-0.60853 dB and snow by -1.15440 dB. On scene 00059, A2-L remains below the
degraded input for both rain and snow. The saved panels show nearly identical
A0-L/A2-L outputs: rain streaks remain visible, while large snow particles are
turned into dark or blurred spots rather than removed.

Relative to A2-L, A3-L changes PSNR by -0.14996 dB and SSIM by -0.001797. It
wins only 1/5 scene means in PSNR; neither metric is significant. Relative to
A0-L, A3-L is -0.19352 dB in PSNR and +0.002354 SSIM. Degradation supervision
therefore does not provide a restoration benefit under the matched schedule,
and A3-L still fails the reasoning gate because snow F1 is zero.

## Decision

Do not proceed to A4. Run the single pre-declared A3-M attempt, which adds a
multiscale descriptor from all encoder skips and train-split label-balanced
BCE while keeping the A3-L schedule and loss weight fixed. If A3-M still fails
to learn rain/snow or does not beat the controls, stop the present one-pass
degradation-reasoning branch. Do not tune further on these five scenes.
