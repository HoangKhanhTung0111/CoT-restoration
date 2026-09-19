# Matched A3-M versus A0-L error comparison

This comparison reuses the existing lightweight exports; it does not train or
evaluate another model. Sources are
`artifacts/kaggle_downloads/10h50_17_09_2026/controls_long_lightweight_results.zip`
and
`artifacts/kaggle_downloads/12h30_17_09_2026/a3m_lightweight_results.zip`.

The comparison is matched: both models ran for 20 epochs, selected their
best-PSNR checkpoint at epoch 14, used the same seed-42 split, SIDD32
initialization, optimization schedule, full-frame AMP evaluation, five
validation scenes, and 55 image/degradation pairs. All 55 keys and their input
metrics match exactly between exports. The held-out test split was not used.

## Aggregate result

| Model | PSNR | SSIM | PSNR-harmed images | Parameters | MACs at 256 |
|---|---:|---:|---:|---:|---:|
| A0-L | **21.29589** | 0.721956 | **2/55 (3.64%)** | 29,159,715 | 16.045 G |
| A3-M | 21.22805 | **0.726550** | 3/55 (5.45%) | 29,548,711 | 16.676 G |

A3-M changes mean PSNR by -0.06784 dB and SSIM by +0.004593. It wins 27/55
individual images in both metrics, but only 3/11 degradation-type mean PSNRs.
The existing five-scene paired comparison is not statistically decisive
(`p=0.619`). A3-M adds 388,996 parameters (+1.33%), 3.93% MACs, and its recorded
mean latency is 28.7% higher; latency includes warm-up outliers and is therefore
secondary to the MAC comparison.

Using output PSNR below input PSNR as the harm definition, both models harm
`rain/00059` and `snow/00059`; A3-M additionally harms `snow/00127`. For SSIM,
both harm the two snow images `00059` and `00127`. Counting an image harmed if
either metric falls below input gives 3/55 for both models, but A3-M is worse
under the primary PSNR definition.

## Degradation-level finding

| Degradation | A3-M - A0-L PSNR | A3-M - A0-L SSIM | A3-M PSNR wins | A0/A3-M PSNR harm |
|---|---:|---:|---:|---:|
| haze | +0.847 | +0.0058 | 4/5 | 0/0 |
| haze+rain | +0.780 | +0.0116 | 4/5 | 0/0 |
| haze+snow | +0.638 | +0.0132 | 4/5 | 0/0 |
| low | -0.291 | -0.0037 | 2/5 | 0/0 |
| low+haze | -0.140 | -0.0134 | 3/5 | 0/0 |
| low+haze+rain | -0.337 | -0.0068 | 2/5 | 0/0 |
| low+haze+snow | -0.081 | -0.0002 | 2/5 | 0/0 |
| low+rain | -0.633 | +0.0019 | 1/5 | 0/0 |
| low+snow | -0.297 | +0.0407 | 1/5 | 0/0 |
| rain | -0.656 | -0.0009 | 1/5 | 1/1 |
| snow | -0.575 | +0.0024 | 3/5 | 1/2 |

A3-M's gains are confined to the three haze-containing conditions without
low light. It loses PSNR on every low-light condition and on both pure
precipitation conditions. Averaged over the four single degradations, A3-M is
-0.16883 dB; over the seven composites it is -0.01013 dB. The large SSIM gain
for low+snow does not correspond to a PSNR gain and the saved panels show
smoothing/dark particle artifacts, so it is not evidence of successful snow
removal.

## Interpretation and decision

The multiscale descriptor and shared bottleneck/skip modulation do learn a
useful haze bias, but introduce negative transfer when low light or
precipitation is present. Better degradation classification therefore does not
translate into better restoration. In particular, the current shared one-pass
modulation cannot preserve A0-L's low-light/rain/snow behavior while applying
the haze improvement.

This closes the evidence review; no Kaggle run is needed. A new restoration
hypothesis should directly address cross-degradation interference, preserve a
strong shared baseline, and isolate atmospheric/illumination correction from
sparse precipitation removal. It should be treated as a new method with A0-L
as the primary matched control, not as another A3-M or beta-controller rescue.
