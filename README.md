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

Training automatically wraps the model with PyTorch `DataParallel` when Kaggle
exposes two T4 GPUs. The notebook uses logical batch 4 and microbatch 2; paired
views make each microbatch contain four images, normally split as two images per
GPU. Evaluation remains single-image tiled inference, where a second GPU offers
little benefit.

Six ordered ablations are recorded in `configs/ablation_matrix.json`. After
multiple runs, collect their saved JSON metrics with:

```bash
python -m hybrid_cot_nafnet.summarize_experiments
```

This writes `/kaggle/working/experiments/ablation_summary.csv`.
