# CoT-Restoration

Lightweight, single-pass composite image restoration using a checkpoint-compatible
NAFNet backbone and a degradation-aware bottleneck/skip adapter.

## Kaggle inputs

The code uses the existing paths below without placeholders:

```text
/kaggle/input/datasets/mintesnotfikir/cdd-11-30
/kaggle/input/datasets/hoangkhanhtung/nafnetmodel
```

Import [`notebooks/kaggle_cot_nafnet.ipynb`](https://github.com/HoangKhanhTung0111/CoT-restoration/blob/main/notebooks/kaggle_cot_nafnet.ipynb)
directly from this GitHub repository; copying individual cells is unnecessary.
The notebook clones the matching commit into `/kaggle/working`, validates all
CDD-11 pairs, checks all four official pretrained checkpoints, and only starts
training after `RUN_TRAIN` is explicitly enabled.

## Current scope

The current implementation is a tested pipeline baseline, not yet a claim of
state-of-the-art CDD-11 performance. Every run records its configuration,
environment, dataset split, pretrained loading report, checkpoints, training
history, and evaluation metrics for later ablation.

After finalizing A0, import
[`notebooks/kaggle_ablation_a1_a2.ipynb`](https://github.com/HoangKhanhTung0111/CoT-restoration/blob/main/notebooks/kaggle_ablation_a1_a2.ipynb)
to run the controlled five-epoch A1/A2 comparison. Its first Run All is a dry
run; enable `RUN_ABLATIONS` only after checking the printed commands.

The fresh notebook defaults to `RUN_TRAIN = False`: **Run All** first performs a
read-only audit and saves `/kaggle/working/cot_nafnet_audit/audit.json`. Review
or share that file before enabling a long run.

An optional `RUN_PRETRAINED_PROBE` switch evaluates all four official NAFNet
checkpoints zero-shot on the five-scene validation split and saves a separate
JSON/CSV report. It also records the unrestored input metrics and metric deltas.
This validation measurement can inform initialization; it is not a trained
CDD-11 baseline. The held-out test split is reserved for final reporting.

The probe rejects non-finite outputs. If a checkpoint overflows under AMP, it
records the fallback and retries that preset in FP32 instead of writing `NaN`
scores. Training performs the same finite-output preflight before epoch one.
Set `RUN_QUALITATIVE_PROBE = True` to save one labelled
`Input | Restored | Ground truth` comparison for every degradation type using
the selected validation preset.

## Research training controls

The default `sidd32` hybrid run uses the validation-selected official pretrained model,
paired degradation views of the same crop, multi-label degradation supervision,
content consistency, content/degradation decorrelation, skip gating, and a
three-epoch adapter warm-up with the backbone frozen.

Training uses PyTorch `DistributedDataParallel` through `torchrun` when Kaggle
exposes two T4 GPUs. Each process owns one fixed model replica, avoiding the
host-memory leak observed when `DataParallel` rebuilt a SIDD32 replica every
forward under Kaggle PyTorch 2.10. The notebook's logical batch 4 and
microbatch 2 become batch 2 and microbatch 1 per GPU. CDD-11 evaluation uses
artifact-free full-frame inference on one GPU by default. Feathered tiled
inference remains available as a memory fallback.

Training crops remain 256 pixels, while validation uses complete images by
default (`--val-crop-size 0`). Consequently, `best.pt` is selected with the
same full-frame PSNR protocol used by the final validation evaluator.

Six ordered ablations are recorded in `configs/ablation_matrix.json`. After
multiple runs, collect their saved JSON metrics with:

```bash
python -m hybrid_cot_nafnet.summarize_experiments
```

This writes `/kaggle/working/experiments/ablation_summary.csv`.
