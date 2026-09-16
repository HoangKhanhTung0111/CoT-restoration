# A3 five-epoch degradation-supervision calibration

A3 matched the selected A2 configuration except that multi-label degradation
supervision was enabled with weight 0.05. It completed all five epochs on two
Tesla T4 GPUs in 6.51 minutes, used full-frame validation, and loaded all
664/664 compatible SIDD32 backbone tensors.

The best-PSNR checkpoint reached 19.6415 dB and 0.67961 SSIM. Relative to A2,
this is -0.1314 dB and +0.00467 SSIM. The five-scene paired comparison was not
statistically decisive (PSNR p=0.499; SSIM p=0.240), so A3 does not establish a
restoration improvement over A2.

Degradation BCE fell from 0.6830 to 0.5766 and micro-F1 reached 0.5917, showing
that the supervised branch learned. The result is incomplete: low and haze had
F1 scores of 0.7792 and 0.7692, while rain and snow both had F1=0 because neither
label was predicted positive at threshold 0.5. Snow restoration remained below
the degraded input by 0.5268 dB and 0.0221 SSIM.

The raw lightweight and comparison ZIP files are intentionally kept outside Git
under `artifacts/kaggle_downloads/A3/`. A3-L is the next controlled diagnostic:
it keeps the A3 model and losses unchanged, extends training to 20 epochs, and
selects separate best-PSNR and best-macro-F1 checkpoints.
