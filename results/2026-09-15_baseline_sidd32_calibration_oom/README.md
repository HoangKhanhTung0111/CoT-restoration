# Baseline SIDD32 calibration — interrupted by host-memory OOM

- Kaggle commit: `9f839b2955948d5b8b41898a32096da46a325ed8`
- Model: baseline NAFNet SIDD-width32, 29.160M parameters
- Data: CDD-11-30, 20/5/5 scene-level train/validation/test split, seed 42
- Runtime: AMP, two Tesla T4 GPUs via DataParallel
- Batch: logical 4, microbatch 2, 256 crop, two sampled patches per image

Epoch 1 completed in roughly 12.5 minutes and reached 16.8445 dB on the
center-crop validation pass. Epoch 2 finished its 110 training steps at 25.4
minutes, then the process received `SIGKILL: 9` while starting validation. The
notebook's subsequent full-resolution evaluation did not run.

This is an interrupted calibration, not a final baseline result. The 16.8445
dB center-crop value is not directly comparable with the earlier full-image
zero-shot probe. The follow-up patch disables pinned host memory and optimizer
serialization for calibration, performs explicit garbage collection, and logs
host/CUDA memory before and after train, validation, and checkpoint stages.
