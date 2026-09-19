# Continuous-beta controller pilot

## Provenance

- Kaggle commit: `c08043eda7f825ed9a3c43be8cee6687f8306a3b`
- Frozen restorer: A3-M best-restoration checkpoint, stored epoch index 13
- Controller: 35,017 parameters, one beta per 32x32 block
- Loss: normalized reconstruction regret
- Split: 15 controller-training scenes, 5 source-disjoint calibration scenes,
  and the unchanged 5-scene/55-image validation split
- Selected controller epoch: 15 of 20
- Test split: untouched
- Raw export: `beta_controller_results.zip`
- SHA-256: `03937B5EE779DA574BF9877EFB9C4EB57B7B4D8F86EB0631C58F8B4B43A502D1`

All archive entries were readable. The frozen-restorer, fixed-beta, and oracle
metrics reproduce the preceding audits exactly.

## Aggregate result

| Method | Mean PSNR | Mean SSIM | Images worse than input |
|---|---:|---:|---:|
| Input | 15.4670 | 0.562878 | 0 |
| A3-M, beta=1 | 21.2280 | 0.726550 | 3 |
| Fixed beta=0.99 | **21.2527** | 0.726854 | 3 |
| Learned beta controller | 21.1971 | **0.726924** | **2** |
| Binary block oracle | 21.9145 | 0.727392 | 0 |
| Continuous block oracle | 22.7659 | 0.732693 | 0 |

The controller is 0.0556 dB below fixed beta and captures -3.67% of the
continuous-oracle headroom. It raises mean SSIM by only 0.000070 and reduces
the harmed-image count from three to two. This is a quality--risk trade-off,
not an overall quality improvement.

## Distribution

The controller beats fixed beta on 20/55 images and loses on 35/55. Grouped by
source scene, its PSNR differences are +0.407, -0.212, +0.071, -0.406, and
-0.137 dB. Grouped by degradation type, only snow is positive (+0.462 dB);
the other ten types range from -0.021 to -0.213 dB.

The controller rescues `rain/00059`: PSNR increases from 31.141 with fixed beta
to 32.844, above the 31.440 input. It also improves but does not rescue
`snow/00059` and `snow/00127`; these remain worse than their inputs. Thus the
reduction from three harmed images to two is real but narrow.

Predicted beta has mean 0.914, with 57.2% of blocks in the interior and 42.8%
near one; no block is near zero. Against the continuous-oracle audit, the
per-image mean-beta MAE is about 0.147 and its Pearson correlation is only
0.171. The controller therefore behaves mainly like broad residual attenuation
rather than recovering the oracle's spatial control pattern.

Training optimization did run: normalized regret falls from 0.109 to about
0.041. Calibration PSNR peaks at 23.2239 dB at epoch 15, still below the
previously measured fixed-beta calibration result of 23.3918 dB. The failure is
therefore already visible on calibration and is not solely an unlucky
validation aggregate.

## Prespecified decision

| Gate | Result | Pass? |
|---|---:|:---:|
| At least +0.2 dB over fixed beta | -0.0556 dB | No |
| No mean SSIM drop | +0.000070 | Yes |
| Fewer harmed images | 2 vs 3 | Yes |
| Capture at least 20% oracle headroom | -3.67% | No |
| Beat fixed beta on at least 7/11 types | 1/11 | No |

Three of five gates fail. The controller is not eligible for a test run.

## Meaning and stopping decision

The continuous oracle establishes substantial available capacity, but this
small local controller cannot infer the required beta pattern reliably from
input, restoration, and residual under the current data protocol. The result
does not disprove all adaptive-strength controllers, but it rejects this
sampled-block CNN and regret objective as the deployable realization.

Under the bounded-pilot rule, stop the local-intervention branch as the main
method direction. Do not tune its epoch, width, seed, or validation operating
point, and do not open the test split. The evidence is now better used to
analyze which content/degradation patterns make the frozen backbone over- or
under-restore, before proposing a materially different AiOIR mechanism.
