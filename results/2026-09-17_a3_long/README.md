# A3-L: twenty-epoch duration diagnostic

Verified source: `artifacts/kaggle_downloads/A3L/a3l_lightweight_results.zip`
(ZIP CRC check passed). Numbers below come from its run summary, dataset
manifest, and the two validation evaluation summaries. This report does not
represent a new training run or a fresh visual inspection of the image panels.

| Checkpoint | Epoch | PSNR (dB) | SSIM | Micro-F1 | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Best PSNR | 20 | 21.102363 | 0.724310 | 0.724138 | 0.551253 |
| Best reasoning | 14 | 21.072952 | 0.723883 | 0.732558 | 0.558052 |

Both evaluations contain 55 full-frame validation samples from scenes
`00059`, `00065`, `00110`, `00122`, and `00127`; seed is 42. Training completed
20 epochs in 25.263 minutes with DDP on two GPUs, logical batch 4 and
microbatch 2 (per rank: 2 and 1), AMP enabled, and no backbone freeze.

At the best-reasoning checkpoint, low/haze/rain/snow F1 is respectively
0.878788 / 0.983051 / 0.370370 / 0. Snow has no positive predictions at
threshold 0.5 and AUROC 0.604286. Thus the reasoning gate remains unmet.
Do not tune the threshold on these same validation samples and report it as
independent evidence of success.

The restoration gain over earlier five-epoch A0/A2 runs is confounded by the
longer training schedule. It does not yet establish a benefit from the adapter
or degradation supervision. Snow at best PSNR averages 25.667400 dB and
0.886888 SSIM, versus input 24.721440 dB and 0.888669 SSIM: mean PSNR improves
while SSIM slightly declines.

## Next controlled comparison

Run `configs/calibration_controls_long.json` using
`notebooks/kaggle_ablation_controls_long.ipynb`:

- A0-L: baseline NAFNet, 20 epochs, all auxiliary losses disabled.
- A2-L: bottleneck adapter and skip gates, 20 epochs, all auxiliary losses disabled.
- Compare each `best.pt` against A3-L `best.pt`, selected by validation PSNR.

Match SIDD32 initialization, seed 42, split, 256-pixel training crops,
two patches per image, batch/microbatch sizes, two-GPU DDP, AdamW settings,
20-epoch cosine schedule, FFT weight 0.05, and full-frame validation.
Keep the existing training implementation to preserve compatibility with A3-L.
Auxiliary classification scores from unsupervised A2-L are not evidence of
degradation recognition, and A0-L has no classification head.

Before interpreting results, confirm completed epochs, pretrained loading,
dataset manifests, AMP/DDP settings and matching `(scene_id, degradation_type)`
rows. Report paired PSNR/SSIM deltas per image, degradation type and scene;
the five scenes, not the 55 related images, are the independent units.

A0-L to A2-L measures the adapter/gates contribution; A2-L to A3-L measures
the contribution of degradation supervision. One seed on five validation
scenes is exploratory. Keep test evaluation reserved for the finalized model.
Do not start A4 while the reasoning gate fails. The next architecture diagnostic
is the single planned A3-M multiscale attempt; if it still fails to learn
rain/snow or improve restoration, stop the current degradation-reasoning branch.

## Preparation and local verification

The two control configurations pass a field-by-field comparison against A3-L;
differences are limited to run name, model, skip gates, degradation weight and
whether the reasoning checkpoint is evaluated. Both DDP command lines pass
the runner's dry run.

Both control branches also completed a one-epoch compact-model GPU smoke run
on synthetic images, including full-frame evaluation, 11 comparison panels per
run and identical data manifests. These smoke checks verify pipeline behavior;
they do not validate SIDD32 pretrained training or two-T4 DDP locally.
Outputs: `artifacts/local_smoke/long_controls_txdq_p23/`.

The production 20-epoch runs have **not started**. To run on Kaggle, import the
control notebook from its GitHub URL, select two T4 GPUs, and attach the existing
CDD-11-30 and nafnetmodel inputs. Run the audit/dry run, then set
`RUN_CONTROLS = True` and run from that cell onward. Download
`controls_long_lightweight_results.zip` and `controls_long_comparisons.zip`
after both runs complete. The notebook prints the exact Git commit used for
provenance.
