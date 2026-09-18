# CoT-Restoration

Lightweight, single-pass composite image restoration using a checkpoint-compatible
NAFNet backbone and a degradation-aware bottleneck/skip adapter.

## Repository layout

```text
configs/             Reproducible experiment definitions
docs/                Paper and design notes
hybrid_cot_nafnet/   Training, evaluation, data and model code
notebooks/           GitHub-importable Kaggle notebooks and baseline reference
results/             Small verified metrics, logs and experiment manifests
artifacts/           Ignored local Kaggle downloads, packages and smoke outputs
NAFNet/, CoTIR/      Ignored upstream reference repositories
```

Raw ZIPs and generated images do not belong in the repository root or Git
history. See `artifacts/README.md` for the local-only layout.

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

A2 is the provisional architecture selected by that calibration. Import
[`notebooks/kaggle_ablation_a3.ipynb`](https://github.com/HoangKhanhTung0111/CoT-restoration/blob/main/notebooks/kaggle_ablation_a3.ipynb)
for the controlled A3 degradation-supervision run. It changes only the
degradation loss weight from 0 to 0.05 and exports labelled qualitative panels.

A3 learned low/haze but did not produce positive rain/snow predictions. Import
[`notebooks/kaggle_ablation_a3_long.ipynb`](https://github.com/HoangKhanhTung0111/CoT-restoration/blob/main/notebooks/kaggle_ablation_a3_long.ipynb)
for the controlled A3-L duration diagnostic. It repeats A3 from scratch for 20
epochs, logs per-label precision/recall/F1/AUROC/AP, and evaluates separate
best-PSNR and best-macro-F1 checkpoints without touching the test split.

A3-L completed 20 epochs but still failed to predict snow at threshold 0.5.
The next step is the matched 20-epoch A0-L/A2-L comparison in
[`notebooks/kaggle_ablation_controls_long.ipynb`](notebooks/kaggle_ablation_controls_long.ipynb),
using [`configs/calibration_controls_long.json`](configs/calibration_controls_long.json).
It runs baseline and adapter/skip-gate controls sequentially on two T4 GPUs,
with auxiliary losses disabled and only best-PSNR validation evaluation.
See the [A3-L report](results/2026-09-17_a3_long/README.md) for the evidence and
comparison protocol. A4 remains deferred pending successful reasoning.

The matched controls show no restoration benefit from A2-L or A3-L. The single
planned architecture rescue is A3-M: import
[`notebooks/kaggle_ablation_a3_multiscale.ipynb`](notebooks/kaggle_ablation_a3_multiscale.ipynb)
from GitHub. It fuses degradation evidence from all encoder scales and balances
the four-label BCE using only train-split frequencies. If A3-M still fails the
reasoning/restoration gates, stop this degradation-reasoning branch.

A3-M completed the planned attempt. It raised macro-F1 to 0.759 and removed
zero-F1 labels, but snow AUROC remained 0.592 and restoration stayed below the
matched A0-L baseline by 0.068 dB. The current one-pass degradation-reasoning
branch is therefore stopped; A4/A5 are not justified. See the
[A3-M report](results/2026-09-17_a3_multiscale/README.md).

A separate intervention-utility feasibility study is now capped at three
attempts. Attempt 1 recovered the lossless 20-epoch comparison panels and found
1.79--2.02 dB of 32x32 local-oracle headroom on the available source scene,
far above the corresponding fixed residual-strength controls. This is only an
exploratory result because all 11 degradation variants share scene `00059`.
The A3-M evaluation now computes the same oracle audit over the complete
55-image validation split. See the
[attempt-1 oracle report](results/2026-09-18_oracle_audit_sample11/README.md).

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
