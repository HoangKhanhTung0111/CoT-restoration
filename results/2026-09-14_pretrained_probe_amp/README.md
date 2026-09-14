# Pretrained probe — 2026-09-14

This run evaluated 55 CDD-11-30 test samples for each of the four official
NAFNet checkpoints with AMP requested.

- `sidd32`: 14.8406 dB / 0.5525 macro PSNR/SSIM.
- `sidd64`: 14.3606 dB / 0.5224 macro PSNR/SSIM.
- `gopro32` and `gopro64`: invalid (`NaN` for all samples).

The audit had already verified an exact 664/664 tensor match for all four
checkpoints. The all-NaN GoPro result is therefore recorded as a numerical AMP
failure, not as a restoration score. Do not use this run to rank all four
initializations. Re-run the two GoPro presets after the FP32-LayerNorm/fallback
fix and merge only their finite results into a later comparison.
