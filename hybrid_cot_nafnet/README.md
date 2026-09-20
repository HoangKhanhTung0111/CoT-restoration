# CoT-NAFNet

This directory contains a dependency-light, end-to-end CDD-11 pipeline for a
single Kaggle GPU. It does not modify or import the BasicSR copy under `NAFNet/`.

## Current design

- `model.py`: checkpoint-compatible NAFNet backbone plus `CoTNAFNet`.
- `modules/gated_cot_adapter.py`: bottleneck reasoning, four-label degradation
  prediction, and zero-initialized affine gates for every skip connection.
- `datasets/cdd11.py`: paired CDD-11 loader with a scene-level validation split.
- `datasets/order_controls.py`: deterministic synthetic low+haze loader for
  Fixed-A, Fixed-B, and Balanced formation-order controls.
- `train_kaggle.py`: AMP training, same-scene paired views, content consistency,
  degradation supervision, adapter warm-up, full-frame validation for checkpoint
  selection, checkpoint/resume, and experiment metadata.
- `evaluate.py`: full CDD-11 evaluation using full-frame inference by default,
  with feathered tiles as a memory fallback.
- `audit_degradation_order.py`: frozen-A0 validation audit that replays the same
  degradation realization across all formation-order permutations and reports
  raw plus input-severity-matched effects without loading CDD-11_test.
- `evaluate_order_controls.py`: joint raw-input evaluation of the three order
  controls with scene-clustered intervals and preregistered contrasts.
- `audit_order_control_data.py`: read-only pretraining gate for scene separation,
  equal update budgets, and paired A/B realization invariants.

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
python -m torch.distributed.run --standalone --nproc_per_node 2 \
  -m hybrid_cot_nafnet.train_kaggle \
  --data-root /kaggle/input/datasets/mintesnotfikir/cdd-11-30 \
  --output-dir /kaggle/working/cot_nafnet_smoke \
  --preset gopro32 --pretrained auto --epochs 1 --max-minutes 8 \
  --crop-size 256 --batch-size 2 --microbatch-size 1 \
  --patches-per-image 1 --num-workers 2
```

Then start the research run:

```bash
python -m torch.distributed.run --standalone --nproc_per_node 2 \
  -m hybrid_cot_nafnet.train_kaggle \
  --data-root /kaggle/input/datasets/mintesnotfikir/cdd-11-30 \
  --output-dir /kaggle/working/cot_nafnet_output \
  --model hybrid --preset gopro32 --pretrained auto \
  --epochs 100 --max-minutes 0 \
  --crop-size 256 --val-crop-size 0 \
  --batch-size 4 --microbatch-size 2 --multi-gpu \
  --patches-per-image 2 --num-workers 0 --no-pin-memory
```

For short calibration runs, add `--no-save-optimizer` to minimize host-memory
pressure. In that mode `last.pt` can restore model weights and epoch number,
but the optimizer/scheduler restart if the run is resumed.

Evaluate the best checkpoint on validation while developing. CDD-11 images fit
comfortably as full frames on a T4:

```bash
python -m hybrid_cot_nafnet.evaluate \
  --checkpoint /kaggle/working/cot_nafnet_output/best.pt \
  --data-root /kaggle/input/datasets/mintesnotfikir/cdd-11-30 \
  --output-dir /kaggle/working/cot_nafnet_evaluation \
  --split validation \
  --tile 0 --num-workers 2
```

Use `--split test` only after the experiment configuration is locked.

Aggregate completed baseline/hybrid runs into one comparison table:

```bash
python -m hybrid_cot_nafnet.summarize_experiments \
  --experiments-root /kaggle/working/experiments \
  --output /kaggle/working/experiments/ablation_summary.csv
```

The initial controlled A1/A2 calibration is recorded separately in
`configs/calibration_a1_a2.json`. Run both configurations, full-frame validation,
full-frame evaluation, and CSV aggregation with:

```bash
python -m hybrid_cot_nafnet.run_ablation \
  --config configs/calibration_a1_a2.json \
  --data-root /kaggle/input/datasets/mintesnotfikir/cdd-11-30 \
  --experiments-root /kaggle/working/experiments_a1_a2 \
  --nproc-per-node 2
```

This short comparison deliberately uses no backbone freeze and a backbone LR
scale of 1.0. That keeps the backbone schedule aligned with A0 and makes skip
gating the only difference between A1 and A2. Adapter-only warm-up remains a
separate optimization ablation for longer experiments.

After selecting A2 provisionally, run the isolated A3 degradation-supervision
calibration with:

```bash
python -m hybrid_cot_nafnet.run_ablation \
  --config configs/calibration_a3.json \
  --data-root /kaggle/input/datasets/mintesnotfikir/cdd-11-30 \
  --experiments-root /kaggle/working/experiments_a3 \
  --nproc-per-node 2
```

The A3 evaluator also saves one labelled `Input | Restored | Ground truth`
comparison per degradation category. Degradation F1 is meaningful for A3
because its multi-label BCE weight is nonzero; it was only diagnostic noise in
A1/A2.

The follow-up A3-L diagnostic keeps A3 unchanged and extends its schedule to 20
epochs. It also evaluates the PSNR-selected and macro-F1-selected checkpoints:

```bash
python -m hybrid_cot_nafnet.run_ablation \
  --config configs/calibration_a3_long.json \
  --data-root /kaggle/input/datasets/mintesnotfikir/cdd-11-30 \
  --experiments-root /kaggle/working/experiments_a3l \
  --nproc-per-node 2
```

`best.pt` is selected by validation PSNR and `best_reasoning.pt` by degradation
macro-F1. Training and final evaluation report per-label precision, recall, F1,
AUROC, and average precision so frequent low/haze labels cannot hide failed
rain/snow recognition.

A3-M is the single multiscale follow-up after the matched 20-epoch controls.
It uses encoder skip statistics for degradation recognition and computes BCE
positive weights from the training split:

```bash
python -m hybrid_cot_nafnet.run_ablation \
  --config configs/calibration_a3_multiscale.json \
  --data-root /kaggle/input/datasets/mintesnotfikir/cdd-11-30 \
  --experiments-root /kaggle/working/experiments_a3m \
  --nproc-per-node 2
```

Legacy checkpoints remain compatible because multiscale reasoning is disabled
unless the checkpoint/config explicitly enables it.

To resume an interrupted run, pass the last checkpoint and keep the same model
configuration:

```bash
python -m torch.distributed.run --standalone --nproc_per_node 2 \
  -m hybrid_cot_nafnet.train_kaggle \
  --data-root /kaggle/input/datasets/mintesnotfikir/cdd-11-30 \
  --output-dir /kaggle/working/cot_nafnet_output \
  --model hybrid --preset gopro32 --epochs 100 --max-minutes 0 \
  --resume /kaggle/working/cot_nafnet_output/last.pt
```

If a 256 training crop causes OOM because another notebook process holds GPU
memory, restart the Kaggle session. As a training fallback, use
`--crop-size 192`; both 192 and 256 are divisible by NAFNet's padding factor of
16. CDD-11 evaluation should use `--tile 0`; for larger images that do not fit,
use feathered inference such as `--tile 512 --overlap 64`.
