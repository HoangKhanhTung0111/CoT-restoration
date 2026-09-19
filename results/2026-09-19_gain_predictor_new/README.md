# Within-image gain-score ranking diagnostic

## Provenance

- Code path: corrected ranking notebook introduced by commit `767f5ee`
- Frozen models: A3-M restorer and the 35,017-parameter gain predictor
- Split: unchanged five-scene/55-image validation split
- Test split: untouched
- Block size: 32x32
- Matched random repetitions: 20 per image and rejection fraction
- Raw export: `gain_predictor_results.zip`
- SHA-256: `7158288AA9B0A1E4826206328670E0D0DBB34EE160E0C37E17EA40B2CDCE7C59`

The archive does not embed the Git commit itself, which is a provenance
limitation. All entries were readable, and the frozen-restorer, fixed-beta,
gain-policy, and oracle metrics reproduce the preceding run.

## Rejection curve

Blocks were ranked separately inside every image. The policy retains the input
for the lowest-scored fraction and uses the A3-M output everywhere else.
Random rejects exactly the same number of blocks inside the same image.

| Rejected | Policy PSNR | Random PSNR | vs random | vs full restoration | Harmed images |
|---:|---:|---:|---:|---:|---:|
| 0% | 21.2280 | 21.2280 | 0.0000 | 0.0000 | 3 |
| 1% | 21.2021 | 21.0449 | +0.1572 | -0.0260 | 3 |
| 2% | 21.1686 | 20.8781 | +0.2905 | -0.0594 | 3 |
| 5% | 21.0597 | 20.4573 | +0.6024 | -0.1683 | 3 |
| 10% | 20.8299 | 19.8865 | +0.9435 | -0.3981 | 3 |
| 20% | 20.2820 | 18.9781 | +1.3038 | -0.9461 | 3 |
| 30% | 19.6595 | 18.2800 | +1.3795 | -1.5685 | 3 |

Every nonzero policy is above the 95th percentile of its matched random
baseline, so the score contains useful relative ranking information. However,
every nonzero policy is worse than full restoration, decreases mean SSIM, and
leaves the same three images worse than their inputs. The validation-selected
policy is therefore 0% rejection.

The strongest practical baseline remains calibrated fixed beta 0.99 at 21.2527
dB and 0.726854 SSIM. The selected policy is 0.0247 dB below it. The local
oracle remains 0.5670 dB above the image oracle and 0.6618 dB above fixed beta.

## What the score learned

At 1% rejection, 111 of 440 rejected blocks are truly harmful, giving 25.2%
precision but only 2.05% recall over 5,405 harmful blocks. This is enrichment
over the approximate 12--13% harmful-block prevalence, but three quarters of the
rejected blocks are still beneficial. Removing those beneficial blocks costs
more PSNR than the correctly rejected harmful blocks recover.

The 1% policy improves only 11 of 55 images and degrades 44. It marginally
improves the three already harmful cases (`rain/00059`, `snow/00059`, and
`snow/00127`) by about 0.0016, 0.0007, and 0.0053 dB respectively, far too
little to make any of them better than its input.

## Prespecified decision

- Useful ranking versus matched random: **pass**.
- Useful versus fixed beta: **fail**.
- Eligible for one locked test run: **no**.

The correct interpretation is that the score ranks relative restoration
benefit, but binary rejection is not useful. This does not establish a
deployable selector and does not justify opening the test split. It also should
not be called negative transfer; the evidenced phenomenon is local restoration
harm on this A3-M validation setting.

The current binary selector branch stops here. A continuation would be a new,
explicit hypothesis—such as cost-sensitive multi-level intervention—not a
threshold or rejection-rate rescue of this model.
