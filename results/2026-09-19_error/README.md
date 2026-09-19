# Oracle-guided error decomposition and learnability audit

## Provenance

- Kaggle commit: `4d4f51c9c92f840a43f544b580a6936841c65a4f`
- Frozen models: A3-M epoch-index-13 checkpoint and the selected beta controller
- Splits: 15 controller-training scenes, 5 calibration scenes, 5 validation scenes
- Evaluation: full-frame outputs and all 32x32 blocks in every split
- Test split: untouched
- Raw export: `error_decomposition_results.zip`
- SHA-256: `F7C29050B303433C68208F8BC87561543EDBA6B48E459B9A87EA0AFB392F319B`

The archive is readable. Analytic block SSE and rendered-image SSE agree to a
maximum relative error of 1.6e-6, 2.5e-6, and 2.2e-6 on train, calibration, and
validation respectively. These are below the prespecified 1e-5 tolerance, so
the diagnosis is not attributable to block assembly or coefficient arithmetic.

## Controller learnability across scene splits

| Split | Images | Controller vs fixed beta | Oracle vs fixed beta | Recoverable SSE captured | Block-beta MAE | Block-beta Pearson |
|---|---:|---:|---:|---:|---:|---:|
| Controller train | 165 | -0.0126 dB | +0.7775 dB | 1.84% | 0.126 | 0.322 |
| Calibration | 55 | -0.1679 dB | +0.5381 dB | -33.79% | 0.128 | 0.242 |
| Validation | 55 | -0.0556 dB | +1.5132 dB | -21.98% | 0.194 | 0.129 |

The controller fails to improve full-image PSNR even on the scenes from which
its sampled training blocks were drawn. It captures only 1.84% of the
oracle-recoverable training SSE. Scene generalization degrades further, but it
is not the primary failure because useful train fit was never established.

## Error decomposition

On validation, clipped continuous beta removes 16.47% of fixed-beta SSE; 83.53%
remains after the oracle. Amplitude control is therefore meaningful—the
per-image macro PSNR gain is large—but it is not the dominant explanation of
total squared error. Much of the remaining error requires a different residual
direction or missing/restored content, not merely rescaling the current output.

The recoverable fraction varies strongly by degradation and scene. Snow is
highest at 33.67%, followed by haze at 26.76%; rain is only 6.53%. Scenes
`00110` and `00059` have 45.36% and 39.99% recoverable error, while `00122` has
only 4.78%. This confirms that the large oracle result is broad in sign but
highly uneven in magnitude.

## Unclipped beta regimes

Validation blocks divide into:

| Unclipped regime | Block fraction | Fraction of recoverable headroom |
|---|---:|---:|
| beta_raw <= 0 | 4.15% | 29.13% |
| 0 < beta_raw < 1 | 47.87% | 66.70% |
| beta_raw >= 1 | 47.98% | 4.17% |

Thus the many blocks whose unconstrained optimum exceeds one contribute little
to the available gain. Conversely, the rare blocks where the current residual
points in a non-helpful direction contribute disproportionately to restoration
harm. On controller-training scenes, beta_raw<=0 blocks are only 1.45% but
already account for 15.36% of headroom, indicating a difficult and imbalanced
tail even before the validation distribution shift.

## Diagnosis and decision

The prespecified primary diagnosis is **controller fit/objective failure**, not
missing context or scene generalization. The current sampled-block training
procedure has not shown that the 35k-parameter controller can exploit its own
training oracle. No new context module or end-to-end architecture is justified
from this result alone.

The next bounded check should stay entirely on controller-training data:

1. Reconstruct the exact 64 sampled blocks per image used during training and
   compare controller regret with fixed beta and oracle on those blocks.
2. Run a small memorization test on a fixed subset using all of its blocks.
3. If the model cannot fit that subset, stop this controller formulation. If it
   fits sampled blocks but not full training images, the identified problem is
   sampling/objective mismatch; only then is one headroom-aware sampling or
   weighting experiment justified.

Do not use validation for that check and do not open the test split. Controlled
information-source probes become relevant only after useful training fit is
demonstrated.
