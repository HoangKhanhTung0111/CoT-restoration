# Baseline SIDD32 five-epoch DDP calibration

- Git commit: `439e8b9b69f292a80c117012881598cc511e689e`
- Model: baseline NAFNet SIDD-width32, 29.160M parameters
- Initialization: official `NAFNet-SIDD-width32.pth`, all 664 tensors matched
- Data: CDD-11-30, 20/5/5 scene-level train/validation/test split, seed 42
- Runtime: AMP and two Tesla T4 GPUs through DDP
- Batch: global 4, per-rank 2, global microbatch 2, per-rank microbatch 1
- Training: five epochs, 256 crop, two sampled patches per image

The run completed in 4.77 minutes. Center-crop validation PSNR increased
monotonically from 16.2076 dB after epoch 1 to 19.1288 dB after epoch 5.

The DDP memory fix succeeded. Combined process RSS rose from 3.17 GB at setup
to 4.11 GB after the first training pass, then stayed between 4.09 and 4.13 GB
through epoch 5. Peak allocated CUDA memory was 1.64 GB on each rank. This
replaces the prior `nn.DataParallel` run, which leaked about 11.5 GB of host RAM
per epoch.

The archived full-image evaluation reported 18.6330 dB and 0.6639 SSIM versus
15.4670 dB and 0.5629 SSIM for the degraded inputs. Those evaluation numbers
used uniform 256-pixel tile averaging and are retained as historical results,
not as the definitive baseline. Visual inspection found grid seams at tile
boundaries. A local A/B check on `rain/00059` improved from 23.34 dB with the
archived tiled path to 28.78 dB with full-frame inference. CDD-11 evaluation is
therefore changed to full-frame inference before the baseline is finalized.

Large checkpoints and rendered images are intentionally excluded from Git.
The original downloaded artifact is stored locally at
`artifacts/kaggle_downloads/my_folder.zip`.
