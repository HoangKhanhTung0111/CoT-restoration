"""Paired, preregistered decision summary for the S2b R0/R1 null pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .datasets.cdd11 import CDD11_TYPES


BOOTSTRAP_SEED = 20260924
BOOTSTRAP_DRAWS = 10_000
TARGET_B = ("b_low_strong_rain_haze", "b_low_strong_snow_haze")
TARGET_A = ("a_low_strong_rain_haze", "a_low_strong_snow_haze")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ("r0", "r1"):
        for probe in ("original", "generator_b", "generator_a"):
            parser.add_argument(
                f"--{arm}-{probe.replace('_', '-')}", action="append", required=True
            )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_rows(paths: str | Path | list[str]) -> dict[tuple[str, str], dict]:
    path_list = [paths] if isinstance(paths, (str, Path)) else paths
    rows = []
    for path in path_list:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    result = {(row["scene"], row["degradation_type"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"Duplicate scene/type rows in {path}")
    return result


def bootstrap_ci(values: np.ndarray) -> list[float]:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975], method="linear")]


def paired_table(r0: dict, r1: dict) -> dict:
    if set(r0) != set(r1):
        raise RuntimeError("R0/R1 evaluation keys differ")
    types = sorted({key[1] for key in r0})
    output = {}
    for degradation_type in types:
        keys = sorted(key for key in r0 if key[1] == degradation_type)
        psnr = np.array([float(r1[key]["psnr"]) - float(r0[key]["psnr"]) for key in keys])
        ssim = np.array([float(r1[key]["ssim"]) - float(r0[key]["ssim"]) for key in keys])
        inner = np.array([
            float(r1[key]["inner32_psnr"]) - float(r0[key]["inner32_psnr"])
            for key in keys
        ])
        harm0 = np.array([float(r0[key]["psnr"]) < float(r0[key]["input_psnr"]) for key in keys])
        harm1 = np.array([float(r1[key]["psnr"]) < float(r1[key]["input_psnr"]) for key in keys])
        output[degradation_type] = {
            "scene_count": len(keys),
            "mean_delta_psnr": float(psnr.mean()),
            "ci95_delta_psnr": bootstrap_ci(psnr),
            "positive_fraction": float((psnr > 0).mean()),
            "mean_delta_inner32_psnr": float(inner.mean()),
            "mean_delta_ssim": float(ssim.mean()),
            "harm_rate_r0": float(harm0.mean()),
            "harm_rate_r1": float(harm1.mean()),
            "harm_rate_delta": float(harm1.mean() - harm0.mean()),
        }
    return output


def pooled_scene_delta(r0: dict, r1: dict, target_types: tuple[str, ...]) -> np.ndarray:
    scenes = sorted({scene for scene, kind in r0 if kind in target_types})
    return np.array([
        np.mean([
            float(r1[(scene, kind)]["psnr"]) - float(r0[(scene, kind)]["psnr"])
            for kind in target_types
        ])
        for scene in scenes
    ])


def summarize(paths: dict[str, dict[str, str]]) -> dict:
    rows = {
        arm: {probe: read_rows(path) for probe, path in probes.items()}
        for arm, probes in paths.items()
    }
    tables = {
        probe: paired_table(rows["r0"][probe], rows["r1"][probe])
        for probe in ("original", "generator_b", "generator_a")
    }
    for arm in ("r0", "r1"):
        for probe in ("original", "generator_b", "generator_a"):
            probe_rows = rows[arm][probe]
            scenes = {scene for scene, _ in probe_rows}
            folds = {int(row["fold"]) for row in probe_rows.values()}
            if len(scenes) != 25 or folds != set(range(5)):
                raise RuntimeError(
                    f"{arm}/{probe} must contain 25 out-of-fold scenes across folds 0..4"
                )
    original_means = [tables["original"][name]["mean_delta_psnr"] for name in CDD11_TYPES]
    pooled_b = pooled_scene_delta(rows["r0"]["generator_b"], rows["r1"]["generator_b"], TARGET_B)
    pooled_a = pooled_scene_delta(rows["r0"]["generator_a"], rows["r1"]["generator_a"], TARGET_A)
    scene_folds = {
        scene: int(row["fold"])
        for (scene, kind), row in rows["r0"]["generator_b"].items()
        if kind == TARGET_B[0]
    }
    fold_b_means = [
        float(np.mean([value for scene, value in zip(sorted(scene_folds), pooled_b)
                       if scene_folds[scene] == fold]))
        for fold in range(5)
    ]
    criteria = {
        "b_each_target_mean_ge_0p2_and_ci_lower_gt_0": all(
            tables["generator_b"][name]["mean_delta_psnr"] >= 0.2
            and tables["generator_b"][name]["ci95_delta_psnr"][0] > 0
            for name in TARGET_B
        ),
        "b_pooled_positive_fraction_gt_0p60": bool((pooled_b > 0).mean() > 0.60),
        "b_positive_mean_in_at_least_4_of_5_folds": sum(value > 0 for value in fold_b_means) >= 4,
        "b_target_ssim_not_down_gt_0p002": all(
            tables["generator_b"][name]["mean_delta_ssim"] >= -0.002 for name in TARGET_B
        ),
        "b_target_harm_rate_not_up_gt_0p04": all(
            tables["generator_b"][name]["harm_rate_delta"] <= 0.04 for name in TARGET_B
        ),
        "a_cross_generator_both_target_means_positive": all(
            tables["generator_a"][name]["mean_delta_psnr"] > 0 for name in TARGET_A
        ),
        "original_macro_delta_ge_minus_0p05": float(np.mean(original_means)) >= -0.05,
        "no_original_type_mean_down_gt_0p1": min(original_means) >= -0.1,
    }
    if all(criteria.values()):
        decision = "COVERAGE_SUFFICIENT_NULL_RESULT"
    elif (
        criteria["b_each_target_mean_ge_0p2_and_ci_lower_gt_0"]
        and not criteria["a_cross_generator_both_target_means_positive"]
    ):
        decision = "COVERAGE_GAIN_GENERATOR_SPECIFIC"
    else:
        decision = "COVERAGE_NOT_SUFFICIENT"
    return {
        "protocol": "s2b-coverage-null-cv5-v2",
        "decision": decision,
        "candidate_method_versions_used": 0,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "draws": BOOTSTRAP_DRAWS,
            "unit": "scene",
            "paired": True,
            "out_of_fold_scene_count": 25,
            "fold_count": 5,
        },
        "criteria": criteria,
        "pooled": {
            "generator_b_target_mean_delta_psnr": float(pooled_b.mean()),
            "generator_b_target_ci95": bootstrap_ci(pooled_b),
            "generator_b_positive_fraction": float((pooled_b > 0).mean()),
            "generator_b_fold_mean_delta_psnr": fold_b_means,
            "generator_a_target_mean_delta_psnr": float(pooled_a.mean()),
            "generator_a_target_ci95": bootstrap_ci(pooled_a),
            "original_macro_delta_psnr": float(np.mean(original_means)),
        },
        "per_type": tables,
    }


def main() -> None:
    args = parse_args()
    paths = {
        arm: {
            probe: getattr(args, f"{arm}_{probe}")
            for probe in ("original", "generator_b", "generator_a")
        }
        for arm in ("r0", "r1")
    }
    result = summarize(paths)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
