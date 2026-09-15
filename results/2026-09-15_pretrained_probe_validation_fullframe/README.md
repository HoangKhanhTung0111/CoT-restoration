# Four-pretrained full-frame validation probe

All four official NAFNet checkpoints loaded with exact backbone compatibility
and were evaluated on the same five-scene, 55-sample CDD-11-30 validation split.

| Preset | PSNR | SSIM | AMP fallback |
|---|---:|---:|---:|
| GoPro width 32 | 8.9515 | 0.2505 | yes |
| GoPro width 64 | 10.8437 | 0.3561 | yes |
| SIDD width 32 | **15.1186** | **0.5852** | no |
| SIDD width 64 | 15.1146 | 0.5832 | no |

SIDD32 is retained as the initialization: it slightly outperforms SIDD64 while
using roughly one quarter of its parameters, about half its latency, and less
than half its peak GPU memory. The GoPro checkpoints are both out of domain and
numerically unstable under FP16 for this probe.

`summary.json` retains aggregate and per-degradation metrics. `metrics.csv`
retains all image-level records. No generated images or checkpoint copies are
stored in Git.
