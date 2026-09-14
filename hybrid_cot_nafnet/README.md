# CoT-NAFNet

This directory contains a dependency-light, end-to-end CDD-11 pipeline for a
single Kaggle GPU. It does not modify or import the BasicSR copy under `NAFNet/`.

## Current design

- `model.py`: checkpoint-compatible NAFNet backbone plus `CoTNAFNet`.
- `modules/gated_cot_adapter.py`: bottleneck reasoning, four-label degradation
  prediction, and zero-initialized affine gates for every skip connection.
- `datasets/cdd11.py`: paired CDD-11 loader with a scene-level validation split.
- `train_kaggle.py`: AMP training, microbatch accumulation, checkpoint/resume,
  validation, and a wall-clock safety limit.
- `evaluate.py`: full CDD-11 evaluation using memory-safe tiled inference.

The current pipeline postpones paired content-consistency loss. That loss needs
a second degradation view of each scene and materially raises activation memory.

## Local smoke checks

```bash
python -m compileall -q hybrid_cot_nafnet
python -m hybrid_cot_nafnet.train_kaggle --help
python -m hybrid_cot_nafnet.evaluate --help
```

## Kaggle commands

Attach the existing Kaggle inputs and enable a GPU. This project uses the exact
paths recorded in `skills.md`:

```text
/kaggle/input/datasets/mintesnotfikir/cdd-11-30/
    CDD-11_train/
    CDD-11_test/
/kaggle/input/datasets/hoangkhanhtung/nafnetmodel/
    NAFNet-GoPro-width32.pth
    NAFNet-GoPro-width64.pth
    NAFNet-SIDD-width32.pth
    NAFNet-SIDD-width64.pth
```

Audit all pairs and all four pretrained checkpoints before training:

```bash
python -m hybrid_cot_nafnet.audit_kaggle
```

First run one epoch using the 17M GoPro-width32 initialization:

```bash
python -m hybrid_cot_nafnet.train_kaggle \
  --data-root /kaggle/input/datasets/mintesnotfikir/cdd-11-30 \
  --output-dir /kaggle/working/cot_nafnet_smoke \
  --preset gopro32 --pretrained auto --epochs 1 --max-minutes 8 \
  --crop-size 256 --batch-size 2 --microbatch-size 1 \
  --patches-per-image 1 --num-workers 2
```

Then start the research run:

```bash
python -m hybrid_cot_nafnet.train_kaggle \
  --data-root /kaggle/input/datasets/mintesnotfikir/cdd-11-30 \
  --output-dir /kaggle/working/cot_nafnet_output \
  --model hybrid --preset gopro32 --pretrained auto \
  --epochs 100 --max-minutes 0 \
  --crop-size 256 --batch-size 2 --microbatch-size 1 \
  --patches-per-image 2 --num-workers 2
```

Evaluate the best checkpoint. Only one tile is held on the GPU at a time:

```bash
python -m hybrid_cot_nafnet.evaluate \
  --checkpoint /kaggle/working/cot_nafnet_output/best.pt \
  --data-root /kaggle/input/datasets/mintesnotfikir/cdd-11-30 \
  --output-dir /kaggle/working/cot_nafnet_evaluation \
  --tile 256 --overlap 32 --num-workers 2
```

To resume an interrupted run, pass the last checkpoint and keep the same model
configuration:

```bash
python -m hybrid_cot_nafnet.train_kaggle \
  --data-root /kaggle/input/datasets/mintesnotfikir/cdd-11-30 \
  --output-dir /kaggle/working/cot_nafnet_output \
  --model hybrid --preset gopro32 --epochs 100 --max-minutes 0 \
  --resume /kaggle/working/cot_nafnet_output/last.pt
```

If a 256 crop still causes OOM because another notebook process holds GPU memory,
restart the Kaggle session. As a fallback, use `--crop-size 192`; both 192 and
256 are divisible by NAFNet's padding factor of 16.
