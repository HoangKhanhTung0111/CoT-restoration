# Local intervention-utility predictor pilot

## Provenance

- Kaggle commit: `1d58da895cfee04e379ff69eb613ef93161cc902`
- Frozen restorer: A3-M best-restoration checkpoint, stored epoch index 13
  (epoch 14 in the human-readable training report)
- Policy: 35,017-parameter CNN predicting signed normalized gain on 32x32 blocks
- Split: 15 predictor-training scenes, 5 source-disjoint calibration scenes,
  and the unchanged 5-scene/55-image validation split
- CDD-11 test: untouched
- Raw export: `gain_predictor_results.zip`
- SHA-256: `2F1CBF9B257F6C2BE80047C274F24705F831DAF624A7CF7BC4BD7CA64D15C456`

The archive entries were read successfully. The reported restored and 32x32
oracle metrics exactly reproduce the previous full-validation oracle audit,
providing a cross-run consistency check.

## Aggregate result

The decision threshold and the fixed residual coefficient were selected only
on the five calibration scenes. Their validation results are:

| Method | Mean PSNR | Mean SSIM | PSNR vs restored | Images worse than input |
|---|---:|---:|---:|---:|
| Input | 15.4670 | 0.562878 | -5.7610 | 0 |
| Frozen A3-M | 21.2280 | 0.726550 | -- | 3 |
| Calibrated fixed beta (0.99) | 21.2527 | 0.726854 | +0.0247 | 3 |
| Learned local policy | 21.2279 | 0.726549 | -0.0001 | 3 |
| Privileged 32x32 oracle | 21.9145 | 0.727392 | +0.6864 | 0 |

The learned policy is 0.0248 dB below fixed beta, captures -0.02% of the
available oracle headroom, and selects the restored result for 99.978% of
pixels. It therefore degenerates to the unmodified frozen restorer.

## Prespecified decision gates

| Gate | Requirement | Result | Pass? |
|---|---:|---:|:---:|
| Quality over fixed beta | at least +0.2 dB | -0.0248 dB | No |
| Mean SSIM | no drop vs fixed beta | -0.000305 | No |
| Harm prevention | fewer than 3 harmed images | 3 | No |
| Oracle utilization | at least 20% | -0.02% | No |
| Breadth | beat beta on at least 7/11 types | 3/11 | No |

All five gates fail. In particular, the policy fails to reject the three known
harmful cases, including `snow/00059` and `snow/00127`, where restoration is
worse than the degraded input.

## Diagnosis and meaning

The CNN did learn some ordering signal: block-level rank correlation between
predicted and true signed gain is 0.637. That ranking is not converted into a
useful decision rule. Calibration selects a very low threshold (-0.483), and
even on the calibration scenes the resulting policy reaches 23.39097 dB versus
23.39178 dB for fixed beta. Thus the failure is already visible before the
locked validation evaluation and is not merely an unlucky validation threshold.

The near-one restored fraction also makes the reported precision of 0.874
uninformative: recall is 0.9999 because nearly every block is accepted. The
training loss decreases from 0.477 to 0.342, so optimization ran, but the
learned score is not calibrated or selective enough to prevent restoration
harm across scenes.

## Decision

The deployable local-gain hypothesis, in this particular sampled-block CNN and
hard-threshold formulation, fails. The oracle result remains valid: useful and
harmful interventions coexist spatially, but this predictor cannot identify
them well enough without ground truth.

Under the prespecified stopping rule, do not rerun this unchanged method, tune
its threshold on validation, or spend additional trials changing only epochs,
width, or seed. Any continuation must be a materially new hypothesis and
protocol rather than a rescue of this configuration.
