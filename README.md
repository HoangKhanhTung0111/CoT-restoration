# CoT-Restoration

Lightweight, single-pass composite image restoration using a checkpoint-compatible
NAFNet backbone and a degradation-aware bottleneck/skip adapter.

## Kaggle inputs

The code uses the existing paths below without placeholders:

```text
/kaggle/input/datasets/mintesnotfikir/cdd-11-30
/kaggle/input/datasets/hoangkhanhtung/nafnetmodel
```

Import `notebooks/kaggle_cot_nafnet.ipynb` directly from this GitHub repository.
The notebook clones the matching commit into `/kaggle/working`, validates all
CDD-11 pairs, checks all four official pretrained checkpoints, and only starts
training after `RUN_TRAIN` is explicitly enabled.

## Current scope

The current implementation is a tested pipeline baseline, not yet a claim of
state-of-the-art CDD-11 performance. Every run records its configuration,
environment, dataset split, pretrained loading report, checkpoints, training
history, and evaluation metrics for later ablation.
