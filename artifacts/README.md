# Local artifacts

This directory contains large or raw files that are useful locally but must not
be committed to Git.

```text
artifacts/
├── kaggle_downloads/  # ZIPs, downloaded notebook copies, raw images and probes
├── packages/          # Generated or legacy source bundles
└── local_smoke/       # Synthetic data and outputs created by local smoke tests
```

Verified lightweight experiment records belong in `results/`. Reproducible
notebooks belong in `notebooks/`. Model checkpoints remain in Kaggle outputs or
an external model store.
