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

A second diagnostic run at commit `4a2a84a` disabled pinned memory and optimizer
serialization and recorded the leak precisely. Process RSS was about 1.5 GB at
setup, 13.3 GB after epoch 1, and 24.8 GB after epoch 2, while CUDA allocation
was only about 335 MB at the measurement points. Validation and checkpointing
left RSS essentially unchanged. The roughly 11.5 GB growth per epoch therefore
occurred during training under `nn.DataParallel`, not in the dataset,
validation loop, or checkpoint writer. That run reached 18.1856 dB after epoch
2 and was then killed before epoch 3.

This is an interrupted calibration, not a final baseline result. The 16.8445
dB center-crop value is not directly comparable with the earlier full-image
zero-shot probe. The follow-up patch disables pinned host memory and optimizer
serialization for calibration, performs explicit garbage collection, and logs
host/CUDA memory before and after train, validation, and checkpoint stages. The
durable fix replaces `nn.DataParallel` with `torchrun` plus
`DistributedDataParallel`: two processes, one persistent model replica per T4.
