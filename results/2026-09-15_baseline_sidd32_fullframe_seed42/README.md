# Finalized A0 baseline: SIDD32, seed 42, five epochs

- Data: CDD-11-30 with the scene-disjoint 20/5/5 train/validation/test split
- Initialization: official `NAFNet-SIDD-width32.pth`; all 664 backbone tensors matched
- Training: five epochs, 256-pixel crops, AMP, DDP on two Tesla T4 GPUs
- Runtime: 4.81 minutes; peak allocated GPU memory was 1.64 GB per rank
- Full-frame validation: **19.4975 dB PSNR / 0.6712 SSIM**
- Degraded input: 15.4670 dB PSNR / 0.5629 SSIM
- Improvement: +4.0305 dB PSNR / +0.1083 SSIM

The final checkpoint improved PSNR on 51/55 validation samples and SSIM on
45/55. Ten degradation categories improved in aggregate; `snow` regressed by
0.8857 dB and remains an explicit failure case for the degradation-aware model.

This run trained with the earlier center-crop validation checkpoint selector.
The validation curve was monotonic and epoch five was selected, so the saved
full-frame result remains a valid A0 calibration reference. Subsequent runs use
full-frame validation during training as well as full-frame final evaluation.

The archived directory intentionally excludes checkpoints and rendered images.
Raw per-image measurements, training logs, memory logs, environment metadata,
and configuration are retained for later ablation analysis. Because validation
contains only five independent scene contents, this is a provisional small-set
baseline rather than a generalization claim.
