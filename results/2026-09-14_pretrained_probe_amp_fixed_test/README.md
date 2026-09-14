# GoPro pretrained probe after numerical fix — 2026-09-14

This diagnostic run evaluated the two GoPro checkpoints on all 55 held-out
CDD-11-30 test samples after adding the AMP finite-output guard.

- `gopro32`: 7.0357 dB PSNR / 0.1702 SSIM, 736.1 ms/image.
- `gopro64`: 8.7526 dB PSNR / 0.3033 SSIM, 1453.5 ms/image.
- Both presets produced non-finite AMP output and completed successfully after
  the automatic FP32 fallback.

These values verify numerical execution only. They must not be combined with
the earlier SIDD test values to select an initialization, because doing so
would leak information from the held-out test split. The replacement probe
uses the scene-disjoint validation split and also reports the unrestored input
baseline.
