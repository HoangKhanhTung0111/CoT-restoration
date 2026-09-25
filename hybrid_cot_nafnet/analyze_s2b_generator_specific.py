"""Descriptive post-hoc audit for the completed S2b generator-specific result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from .datasets.cdd11 import CDD11_TYPES
from .summarize_s2b_coverage import TARGET_A, TARGET_B


ARMS = {
    "r0": "s2b_r0_reconstruction",
    "r1": "s2b_r1_coverage",
}
PROBES = ("original", "generator_b", "generator_a")
EXPECTED_TYPES = {
    "original": tuple(CDD11_TYPES),
    "generator_b": (
        "b_low_weak",
        "b_low_strong",
        "b_rain_haze",
        "b_snow_haze",
        "b_low_weak_rain_haze",
        "b_low_strong_rain_haze",
        "b_low_weak_snow_haze",
        "b_low_strong_snow_haze",
    ),
    "generator_a": (
        "a_low_weak",
        "a_low_strong",
        "a_rain_haze",
        "a_snow_haze",
        "a_low_weak_rain_haze",
        "a_low_strong_rain_haze",
        "a_low_weak_snow_haze",
        "a_low_strong_snow_haze",
    ),
}
FAMILIES = {
    "generator_b": {
        "low_only": ("b_low_weak", "b_low_strong"),
        "weather_only": ("b_rain_haze", "b_snow_haze"),
        "low_weather_composite": (
            "b_low_weak_rain_haze",
            "b_low_strong_rain_haze",
            "b_low_weak_snow_haze",
            "b_low_strong_snow_haze",
        ),
    },
    "generator_a": {
        "low_only": ("a_low_weak", "a_low_strong"),
        "weather_only": ("a_rain_haze", "a_snow_haze"),
        "low_weather_composite": (
            "a_low_weak_rain_haze",
            "a_low_strong_rain_haze",
            "a_low_weak_snow_haze",
            "a_low_strong_snow_haze",
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(archive: zipfile.ZipFile, name: str) -> dict:
    try:
        payload = archive.read(name)
    except KeyError as exc:
        raise RuntimeError(f"Missing archive member: {name}") from exc
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {name}")
    return value


def _read_csv(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    try:
        payload = archive.read(name)
    except KeyError as exc:
        raise RuntimeError(f"Missing archive member: {name}") from exc
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))


def load_metrics(archive: zipfile.ZipFile) -> dict[str, dict[str, dict[tuple, dict]]]:
    result: dict[str, dict[str, dict[tuple, dict]]] = {
        arm: {probe: {} for probe in PROBES} for arm in ARMS
    }
    for fold in range(5):
        for arm, run_name in ARMS.items():
            for probe in PROBES:
                name = (
                    f"experiments/fold_{fold}/{run_name}/"
                    f"evaluation_{probe}/metrics.csv"
                )
                for row in _read_csv(archive, name):
                    row_fold = int(row["fold"])
                    if row_fold != fold:
                        raise RuntimeError(f"Fold mismatch in {name}: {row_fold}")
                    key = (row_fold, row["scene"], row["degradation_type"])
                    if key in result[arm][probe]:
                        raise RuntimeError(f"Duplicate metric key in {arm}/{probe}: {key}")
                    result[arm][probe][key] = row
    return result


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def correlation(left: list[float], right: list[float]) -> dict[str, float]:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Correlation inputs must have the same length >= 2")

    def pearson(a: list[float], b: list[float]) -> float:
        value = float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])
        if not math.isfinite(value):
            raise RuntimeError("Correlation is not finite")
        return value

    return {
        "pearson": pearson(left, right),
        "spearman": pearson(average_ranks(left), average_ranks(right)),
    }


def quadrant_counts(left: list[float], right: list[float]) -> dict[str, int]:
    counts = {
        "left_positive_right_positive": 0,
        "left_positive_right_nonpositive": 0,
        "left_nonpositive_right_positive": 0,
        "left_nonpositive_right_nonpositive": 0,
    }
    for left_value, right_value in zip(left, right):
        left_label = "positive" if left_value > 0 else "nonpositive"
        right_label = "positive" if right_value > 0 else "nonpositive"
        counts[f"left_{left_label}_right_{right_label}"] += 1
    return counts


def analyze(archive_path: Path) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise RuntimeError(f"Archive CRC failed at {corrupt_member}")
        run = _read_json(archive, "run.json")
        preregistered = _read_json(archive, "s2b_coverage_summary.json")
        metrics = load_metrics(archive)

    if run.get("status") != "COMPLETE" or run.get("errors"):
        raise RuntimeError("S2b archive is not complete")
    if preregistered.get("decision") != "COVERAGE_GAIN_GENERATOR_SPECIFIC":
        raise RuntimeError("Unexpected preregistered S2b decision")

    deltas: dict[str, dict[tuple, dict[str, float]]] = {}
    maximum_input_difference = 0.0
    for probe in PROBES:
        r0 = metrics["r0"][probe]
        r1 = metrics["r1"][probe]
        if set(r0) != set(r1):
            raise RuntimeError(f"R0/R1 keys differ for {probe}")
        expected_types = set(EXPECTED_TYPES[probe])
        actual_types = {key[2] for key in r0}
        if actual_types != expected_types:
            raise RuntimeError(f"Unexpected degradation types for {probe}")
        if len(r0) != 25 * len(expected_types):
            raise RuntimeError(f"Unexpected metric row count for {probe}: {len(r0)}")
        deltas[probe] = {}
        for key in sorted(r0):
            for field in ("input_psnr", "input_ssim", "inner32_input_psnr"):
                difference = abs(float(r1[key][field]) - float(r0[key][field]))
                maximum_input_difference = max(maximum_input_difference, difference)
            deltas[probe][key] = {
                "psnr": float(r1[key]["psnr"]) - float(r0[key]["psnr"]),
                "ssim": float(r1[key]["ssim"]) - float(r0[key]["ssim"]),
                "inner32_psnr": (
                    float(r1[key]["inner32_psnr"])
                    - float(r0[key]["inner32_psnr"])
                ),
            }
    if maximum_input_difference > 1e-12:
        raise RuntimeError(
            f"R0/R1 input metrics differ; max absolute difference={maximum_input_difference}"
        )

    scene_folds: dict[str, set[int]] = defaultdict(set)
    for probe_rows in deltas.values():
        for fold, scene, _ in probe_rows:
            scene_folds[scene].add(fold)
    if len(scene_folds) != 25 or any(len(folds) != 1 for folds in scene_folds.values()):
        raise RuntimeError("Expected 25 scenes, each assigned to exactly one fold")

    def scene_mean(probe: str, scene: str, types: tuple[str, ...]) -> float:
        fold = next(iter(scene_folds[scene]))
        return float(np.mean([deltas[probe][(fold, scene, kind)]["psnr"] for kind in types]))

    scene_rows = []
    for scene in sorted(scene_folds):
        fold = next(iter(scene_folds[scene]))
        row = {
            "scene": scene,
            "fold": fold,
            "generator_b_target_delta_psnr": scene_mean(
                "generator_b", scene, TARGET_B
            ),
            "generator_a_target_delta_psnr": scene_mean(
                "generator_a", scene, TARGET_A
            ),
            "original_macro_delta_psnr": scene_mean(
                "original", scene, tuple(CDD11_TYPES)
            ),
            "generator_b_all_delta_psnr": scene_mean(
                "generator_b", scene, EXPECTED_TYPES["generator_b"]
            ),
            "generator_a_all_delta_psnr": scene_mean(
                "generator_a", scene, EXPECTED_TYPES["generator_a"]
            ),
        }
        row["target_generalization_gap_b_minus_a"] = (
            row["generator_b_target_delta_psnr"]
            - row["generator_a_target_delta_psnr"]
        )
        scene_rows.append(row)

    b_target = [row["generator_b_target_delta_psnr"] for row in scene_rows]
    a_target = [row["generator_a_target_delta_psnr"] for row in scene_rows]
    original_macro = [row["original_macro_delta_psnr"] for row in scene_rows]

    type_means = {}
    for probe in PROBES:
        type_means[probe] = {}
        for kind in EXPECTED_TYPES[probe]:
            values = [
                value["psnr"]
                for key, value in deltas[probe].items()
                if key[2] == kind
            ]
            type_means[probe][kind] = float(np.mean(values))

    family_means = {}
    for probe, families in FAMILIES.items():
        family_means[probe] = {
            family: float(np.mean([type_means[probe][kind] for kind in types]))
            for family, types in families.items()
        }

    fold_means = []
    for fold in range(5):
        fold_rows = [row for row in scene_rows if row["fold"] == fold]
        fold_means.append(
            {
                "fold": fold,
                "scene_count": len(fold_rows),
                "generator_b_target_delta_psnr": float(
                    np.mean([row["generator_b_target_delta_psnr"] for row in fold_rows])
                ),
                "generator_a_target_delta_psnr": float(
                    np.mean([row["generator_a_target_delta_psnr"] for row in fold_rows])
                ),
                "original_macro_delta_psnr": float(
                    np.mean([row["original_macro_delta_psnr"] for row in fold_rows])
                ),
            }
        )

    pooled = preregistered["pooled"]
    checks = {
        "archive_crc_passed": True,
        "run_complete": True,
        "r0_r1_input_metrics_exact_match": maximum_input_difference == 0.0,
        "scene_count": len(scene_rows),
        "metric_rows_total": sum(
            len(metrics[arm][probe]) for arm in ARMS for probe in PROBES
        ),
        "generator_b_target_mean_matches_preregistered": math.isclose(
            float(np.mean(b_target)),
            pooled["generator_b_target_mean_delta_psnr"],
            abs_tol=1e-12,
        ),
        "generator_a_target_mean_matches_preregistered": math.isclose(
            float(np.mean(a_target)),
            pooled["generator_a_target_mean_delta_psnr"],
            abs_tol=1e-12,
        ),
        "original_macro_mean_matches_preregistered": math.isclose(
            float(np.mean(original_macro)),
            pooled["original_macro_delta_psnr"],
            abs_tol=1e-12,
        ),
    }
    boolean_checks = {
        key: value for key, value in checks.items() if isinstance(value, bool)
    }
    if (
        not all(boolean_checks.values())
        or checks["scene_count"] != 25
        or checks["metric_rows_total"] != 1350
    ):
        raise RuntimeError(f"Post-hoc audit check failed: {checks}")

    largest_gaps = sorted(
        scene_rows,
        key=lambda row: row["target_generalization_gap_b_minus_a"],
        reverse=True,
    )[:5]
    largest_original_regressions = sorted(
        scene_rows, key=lambda row: row["original_macro_delta_psnr"]
    )[:5]

    return {
        "analysis_version": "s2b-posthoc-generator-transfer-v1",
        "analysis_status": "COMPLETE",
        "analysis_scope": "descriptive_posthoc_no_gate_changes",
        "source": {
            "archive": archive_path.name,
            "archive_sha256": sha256_file(archive_path),
            "run_id_utc": run["run_id_utc"],
            "project_commit": run["project_commit"],
            "protocol": preregistered["protocol"],
        },
        "audit_checks": checks,
        "preregistered_result": {
            "decision": preregistered["decision"],
            "criteria": preregistered["criteria"],
            "pooled": preregistered["pooled"],
            "candidate_method_versions_used": preregistered[
                "candidate_method_versions_used"
            ],
        },
        "exploratory_scene_analysis": {
            "correlations": {
                "generator_b_target_vs_generator_a_target": correlation(
                    b_target, a_target
                ),
                "generator_b_target_vs_original_macro": correlation(
                    b_target, original_macro
                ),
                "generator_a_target_vs_original_macro": correlation(
                    a_target, original_macro
                ),
            },
            "sign_quadrants": {
                "generator_b_target_vs_generator_a_target": quadrant_counts(
                    b_target, a_target
                ),
                "generator_b_target_vs_original_macro": quadrant_counts(
                    b_target, original_macro
                ),
            },
            "mean_target_generalization_gap_b_minus_a": float(
                np.mean(np.asarray(b_target) - np.asarray(a_target))
            ),
            "type_mean_delta_psnr": type_means,
            "family_mean_delta_psnr": family_means,
            "fold_mean_delta_psnr": fold_means,
            "largest_generator_specific_gaps": largest_gaps,
            "largest_original_regressions": largest_original_regressions,
            "scene_rows": scene_rows,
        },
        "interpretation_limits": [
            "All exploratory quantities were chosen after the preregistered result was known.",
            "Correlations and rankings are descriptive; no confirmatory p-values or new gates are reported.",
            "The analysis can identify generator-specific association but cannot identify its causal mechanism.",
            "CDD-11_test and the S1 holdout remain sealed.",
        ],
    }


def main() -> None:
    args = parse_args()
    result = analyze(Path(args.archive).resolve())
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
