# CoT-NAFNet

This directory contains a dependency-light, end-to-end CDD-11 pipeline for a
single Kaggle GPU. It does not modify or import the BasicSR copy under `NAFNet/`.

## Current design

- `model.py`: checkpoint-compatible NAFNet backbone plus `CoTNAFNet`.
- `modules/gated_cot_adapter.py`: bottleneck reasoning, four-label degradation
  prediction, and zero-initialized affine gates for every skip connection.
- `datasets/cdd11.py`: paired CDD-11 loader with a scene-level validation split.
- `train_kaggle.py`: AMP training, same-scene paired views, content consistency,
  degradation supervision, adapter warm-up, checkpoint/resume, and experiment
  metadata.
- `evaluate.py`: full CDD-11 evaluation using memory-safe tiled inference.

The research path uses the same spatial crop and augmentation for two different
degradation views of one scene. Set `--content-weight 0` to disable this branch
for its ablation. Use `--no-skip-gates` for the bottleneck-only ablation.

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

Optionally measure the four pretrained models zero-shot on the scene-disjoint
CDD-11 validation split (the default and the only split used for selection):

```bash
python -m hybrid_cot_nafnet.probe_pretrained_cdd11
```

The report includes input PSNR/SSIM and output-minus-input deltas. Use
`--split test` only for a final locked evaluation, never to choose the
pretrained initialization or tune hyperparameters.

For a visual sanity check, save one labelled comparison per degradation type:

```bash
python -m hybrid_cot_nafnet.probe_pretrained_cdd11 \
  --presets sidd32 --split validation --save-comparisons
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
  --crop-size 256 --batch-size 4 --microbatch-size 2 --multi-gpu \
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

Aggregate completed baseline/hybrid runs into one comparison table:

```bash
python -m hybrid_cot_nafnet.summarize_experiments \
  --experiments-root /kaggle/working/experiments \
  --output /kaggle/working/experiments/ablation_summary.csv
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
