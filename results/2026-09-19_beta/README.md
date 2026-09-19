# Continuous-beta oracle audit

## Provenance

- Kaggle commit: `d83a484ebc76d9a7f77552a1ee2292fc1ee816f4`
- Frozen restorer: A3-M best-restoration checkpoint, stored epoch index 13
- Split: unchanged five-scene/55-image validation split
- Block size: 32x32
- Fixed-beta control: 0.99, selected on the preceding calibration scenes
- Test split: untouched
- Raw export: `continuous_beta_oracle_results.zip`
- SHA-256: `50C657B0873BB5EEBBE4FA09CB36BD3F48DA707997E930CC8EC0A7F766809191`

All archive entries were readable. The input, A3-M, fixed-beta, and binary
oracle metrics exactly reproduce the preceding audits.

## Aggregate result

| Method | Mean PSNR | Mean SSIM | Images worse than input |
|---|---:|---:|---:|
| Input | 15.4670 | 0.562878 | 0 |
| A3-M, beta=1 | 21.2280 | 0.726550 | 3 |
| Fixed beta=0.99 | 21.2527 | 0.726854 | 3 |
| Binary image oracle | 21.3475 | 0.728125 | 0 |
| Binary 32x32 oracle | 21.9145 | 0.727392 | 0 |
| Continuous image oracle | 21.9077 | 0.729203 | 0 |
| Continuous 32x32 oracle | **22.7659** | **0.732693** | **0** |

The continuous block oracle is 0.8514 dB above the binary block oracle, 1.5132
dB above fixed beta, and 0.8582 dB above the continuous image oracle. Mean SSIM
also rises by 0.00530 over the binary block oracle, so the PSNR result is not
bought with a mean structural-similarity decrease.

## Breadth and beta distribution

The gain over the binary block oracle is positive for all 55 images. The median
per-image gain is 0.293 dB, 36/55 images gain at least 0.1 dB, and the range is
approximately 0.00002--4.999 dB. Scene `00110` contributes several large gains,
but all five scene aggregates are positive: +1.089, +0.142, +2.385, +0.224,
and +0.416 dB.

All 11 degradation categories improve. The single-degradation group gains
+0.802 dB and the composite group gains +0.880 dB. Thus the aggregate is not
limited to either simple or composite corruptions.

The analytic block betas are distributed as follows:

- beta <= 0.05: 5.22%
- 0.05 < beta < 0.95: **40.53%**
- beta >= 0.95: 54.25%

The large interior mass shows that intermediate restoration strength provides
genuine additional oracle capacity; this is not merely the previous binary
choice expressed on a continuous scale.

## Decision and meaning

All prespecified resource-allocation gates pass:

- at least +0.1 dB over the binary block oracle: **+0.851 dB**;
- positive on at least four of five scenes: **5/5**;
- positive for both single and composite groups: **yes**.

This establishes a strong and broad upper bound for spatially varying residual
strength in the current AiOIR setting. It does not show that the optimal beta is
predictable without ground truth. The earlier binary gain predictor already
demonstrated that oracle feasibility and deployability are separate questions.

The justified next step is one bounded controller pilot with A3-M frozen. The
controller should predict continuous block beta from input, restored output,
and residual, and train against reconstruction regret rather than thresholding
signed gain. Fixed beta remains the primary practical control. The test split
must remain closed until that controller passes validation gates.
