# Controller training-fit audit

The audit used commit `a8b77e9` and only the 15 controller-training scenes.
Calibration, validation, and test were not loaded.

## Result

- On the exact 10,560 sampled training blocks, the frozen controller reduced
  normalized regret from 0.05354 to 0.04280, but captured only 1.48% of
  oracle-recoverable SSE.
- On all 129,030 training blocks, it captured only 1.84% of recoverable SSE.
- The unchanged architecture captured 97.75% of recoverable SSE when trained
  to memorize all 782 blocks of one high-headroom snow image, passing the
  prespecified 80% memorization gate.
- That memorizer mostly learned an image-wide near-zero action: mean beta was
  0.0257 and beta/oracle correlation was only 0.181 at its best-capture epoch.

The deployed controller therefore fails the 20% training-headroom gate even
though it improves its own normalized block-regret objective. Similar capture
on sampled and full training blocks argues against block sampling as the main
cause. The evidence instead supports objective/weighting misalignment: the
normalized per-block loss can improve while neglecting rare blocks with large
absolute impact on image PSNR.

## Decision

An information-source or larger-architecture probe is not authorized. The one
bounded rescue keeps both A3-M and the 35k-parameter controller unchanged, uses
all training blocks, and replaces the loss with equal-image
`log(SSE / fixed-SSE)`, which is equivalent to maximizing mean image PSNR.
Validation may be loaded only if the calibration-selected checkpoint captures
at least 20% of mean-PSNR oracle headroom on its own training scenes. The test
split remains closed.
