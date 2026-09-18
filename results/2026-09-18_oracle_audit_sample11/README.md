# Exploratory oracle audit on saved 20-epoch contact sheets

## Scope

This is attempt 1 of at most 3 for the intervention-utility hypothesis. The
20-epoch exports retained one lossless comparison sheet per degradation type,
so this audit covers 11 degradation variants of only source scene `00059`.
It is an exploratory signal, not a validation-set conclusion.

The oracle is allowed to inspect ground truth and select either the degraded
input or the restored output at image, 32x32-block, or 8x8-block resolution.
The fixed-beta control selects one global residual strength on these same 11
images and is therefore also optimistic.

## Mean PSNR results

| Model | Restored | Best fixed beta | Image oracle | 32x32 oracle | 8x8 oracle |
|---|---:|---:|---:|---:|---:|
| A0-L | 22.9999 | 23.0594 (+0.0595) | 23.4276 (+0.4278) | 24.7909 (+1.7910) | 25.3123 (+2.3124) |
| A2-L | 23.0963 | 23.1356 (+0.0393) | 23.8140 (+0.7177) | 25.0764 (+1.9801) | 25.6532 (+2.5569) |
| A3-M | 23.1404 | 23.3820 (+0.2416) | 23.5611 (+0.4206) | 25.1595 (+2.0190) | 25.7616 (+2.6212) |

Parentheses report headroom over the unmodified restored output. A3-M's best
fixed beta is 0.88 on this exploratory sample. The A3-M oracle keeps about
76.6% of pixels restored at 32x32 resolution and 74.6% at 8x8 resolution.

## Interpretation and gate decision

The large gap between local oracle headroom (2.02--2.62 dB for A3-M) and the
best global residual shrinkage control (+0.24 dB) supports the existence of
spatially heterogeneous helpful and harmful interventions on this scene. The
signal appears for all three model variants, so it is not unique to the failed
degradation-reasoning adapter.

Attempt 1 passes the exploratory gate, but cannot establish average headroom:
all 11 samples share one source scene. Attempt 2 must run the same audit on all
55 validation images inside Kaggle. If the 32x32 oracle does not retain at
least 0.2 dB mean headroom over both the restored output and best fixed-beta
control, the local intervention-utility direction stops without training a
gain predictor.

## Reproduction

```powershell
python -m hybrid_cot_nafnet.oracle_audit `
  --model a0l artifacts/kaggle_downloads/10h50_17_09_2026/comparisons_extracted/a0l_nafnet_baseline_sidd32_seed42_20ep `
  --model a2l artifacts/kaggle_downloads/10h50_17_09_2026/comparisons_extracted/a2l_bottleneck_skip_sidd32_seed42_20ep `
  --model a3m artifacts/kaggle_downloads/12h30_17_09_2026/comparisons_extracted/evaluation `
  --output-dir results/2026-09-18_oracle_audit_sample11
```

The Kaggle evaluation path now supports `--oracle-audit` and records the same
diagnostics directly for every evaluated image without exporting full contact
sheets.
