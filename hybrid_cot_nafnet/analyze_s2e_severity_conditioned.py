"""CPU-only severity-conditioned falsification of the S2b A/B interaction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression, Ridge

from .analyze_s2b_generator_specific import ARMS, PROBES, load_metrics
from .calibrate_s2d_severity import _lpips_model, pixel_sha256, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_member(archive: zipfile.ZipFile, name: str) -> dict:
    try:
        value = json.loads(archive.read(name).decode("utf-8"))
    except KeyError as exc:
        raise RuntimeError(f"Missing archive member: {name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {name}")
    return value


def _csv_member(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    try:
        payload = archive.read(name).decode("utf-8")
    except KeyError as exc:
        raise RuntimeError(f"Missing archive member: {name}") from exc
    return list(csv.DictReader(io.StringIO(payload)))


def _decode_png(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _resize_lpips(image: np.ndarray, longest_side: int = 512) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, float(longest_side) / float(max(height, width)))
    if scale == 1.0:
        return np.ascontiguousarray(image)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )


def _lpips_batch(inputs: list[np.ndarray], targets: list[np.ndarray], model, torch) -> list[float]:
    if not inputs or len(inputs) != len(targets):
        raise ValueError("LPIPS input/target batches must be non-empty and aligned")
    shapes = {array.shape for array in [*inputs, *targets]}
    if len(shapes) != 1:
        raise ValueError(f"LPIPS batch shapes differ: {sorted(shapes)}")
    left = torch.from_numpy(np.stack(inputs)).permute(0, 3, 1, 2)
    right = torch.from_numpy(np.stack(targets)).permute(0, 3, 1, 2)
    left = left.float().div(127.5).sub(1.0)
    right = right.float().div(127.5).sub(1.0)
    with torch.inference_mode():
        values = model(left, right).reshape(-1).cpu().numpy()
    return [float(value) for value in values]


def _load_s2b_rows(
    archive_path: Path, primary_conditions: tuple[str, ...], expected_rows: int
) -> tuple[list[dict], dict]:
    with zipfile.ZipFile(archive_path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"S2b archive CRC failed at {corrupt}")
        run = _json_member(archive, "run.json")
        summary = _json_member(archive, "s2b_coverage_summary.json")
        metrics = load_metrics(archive)
        cache_payload = archive.read("cache_manifest.json")

    if run.get("status") != "COMPLETE" or run.get("errors"):
        raise RuntimeError("S2b archive is not complete")
    if summary.get("decision") != "COVERAGE_GAIN_GENERATOR_SPECIFIC":
        raise RuntimeError("Unexpected S2b registered decision")
    metric_count = sum(
        len(metrics[arm][probe]) for arm in ARMS for probe in PROBES
    )
    if metric_count != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} S2b rows, found {metric_count}")

    rows = []
    for generator, probe, prefix in (
        ("A", "generator_a", "a_"),
        ("B", "generator_b", "b_"),
    ):
        r0 = metrics["r0"][probe]
        r1 = metrics["r1"][probe]
        if set(r0) != set(r1):
            raise RuntimeError(f"R0/R1 keys differ for {probe}")
        for key in sorted(r0):
            fold, scene, kind = key
            if not kind.startswith(prefix):
                raise RuntimeError(f"Unexpected degradation type: {kind}")
            condition = kind[len(prefix) :]
            if condition not in primary_conditions:
                continue
            maximum_input_difference = max(
                abs(float(r0[key][field]) - float(r1[key][field]))
                for field in ("input_psnr", "input_ssim", "inner32_input_psnr")
            )
            if maximum_input_difference > 1e-12:
                raise RuntimeError(f"R0/R1 input mismatch at {key}")
            rows.append(
                {
                    "scene": scene,
                    "fold": int(fold),
                    "condition": condition,
                    "generator": generator,
                    "input_psnr": float(r0[key]["input_psnr"]),
                    "input_ssim": float(r0[key]["input_ssim"]),
                    "r0_psnr": float(r0[key]["psnr"]),
                    "r1_psnr": float(r1[key]["psnr"]),
                    "gain_psnr": float(r1[key]["psnr"])
                    - float(r0[key]["psnr"]),
                }
            )
    return rows, {
        "run_id_utc": run.get("run_id_utc"),
        "project_commit": run.get("project_commit"),
        "protocol": summary.get("protocol"),
        "metric_rows": metric_count,
        "cache_manifest_sha256": _sha256_bytes(cache_payload),
    }


def _attach_lpips(
    rows: list[dict], archive_path: Path, raw_expected: dict
) -> tuple[list[dict], dict]:
    model, torch = _lpips_model()
    by_key = {(row["scene"], row["condition"], row["generator"]): row for row in rows}
    if len(by_key) != len(rows):
        raise RuntimeError("Duplicate primary S2b key")

    with zipfile.ZipFile(archive_path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"S2d raw archive CRC failed at {corrupt}")
        manifest_payload = archive.read("probe/s2d_raw_manifest.json")
        manifest = json.loads(manifest_payload.decode("utf-8"))
        if manifest.get("input_record_count") != raw_expected["raw_inputs"]:
            raise RuntimeError("Unexpected S2d raw input count")
        if manifest.get("gt_record_count") != raw_expected["raw_ground_truth"]:
            raise RuntimeError("Unexpected S2d raw GT count")
        inputs = {
            (item["scene"], item["condition"], item["generator"]): item
            for item in manifest["inputs"]
        }
        ground_truth = {item["scene"]: item for item in manifest["ground_truth"]}
        if len(inputs) != raw_expected["raw_inputs"] or len(ground_truth) != raw_expected["raw_ground_truth"]:
            raise RuntimeError("Duplicate S2d raw manifest key")
        scenes = sorted({row["scene"] for row in rows})
        for scene in scenes:
            gt_record = ground_truth[scene]
            gt_payload = archive.read(f"probe/{gt_record['file']}")
            if _sha256_bytes(gt_payload) != gt_record["file_sha256"]:
                raise RuntimeError(f"GT file hash mismatch: {scene}")
            gt = _decode_png(gt_payload)
            if pixel_sha256(gt) != gt_record["pixel_sha256"]:
                raise RuntimeError(f"GT pixel hash mismatch: {scene}")
            resized_gt = _resize_lpips(gt)
            scene_rows = sorted(
                (row for row in rows if row["scene"] == scene),
                key=lambda row: (row["condition"], row["generator"]),
            )
            resized_inputs = []
            for row in scene_rows:
                key = (scene, row["condition"], row["generator"])
                record = inputs.get(key)
                if record is None:
                    raise RuntimeError(f"Missing raw input: {key}")
                if int(record["fold"]) != row["fold"]:
                    raise RuntimeError(f"Fold mismatch between archives: {key}")
                payload = archive.read(f"probe/{record['file']}")
                if _sha256_bytes(payload) != record["file_sha256"]:
                    raise RuntimeError(f"Input file hash mismatch: {key}")
                image = _decode_png(payload)
                if pixel_sha256(image) != record["pixel_sha256"]:
                    raise RuntimeError(f"Input pixel hash mismatch: {key}")
                resized = _resize_lpips(image)
                if resized.shape != resized_gt.shape:
                    raise RuntimeError(f"Input/GT resized shape mismatch: {key}")
                resized_inputs.append(resized)
            values = _lpips_batch(
                resized_inputs, [resized_gt] * len(resized_inputs), model, torch
            )
            for row, value in zip(scene_rows, values):
                row["input_lpips"] = value

        raw_cache_hash = manifest.get("cache_manifest_sha256")
        raw_split_hash = manifest.get("split_manifest_sha256")
        safety = {
            "cdd11_test_opened": manifest.get("cdd11_test_opened"),
            "raw_exported": manifest.get("raw_exported"),
        }
    if any("input_lpips" not in row for row in rows):
        raise RuntimeError("LPIPS was not attached to every primary row")
    return rows, {
        "raw_manifest_sha256": _sha256_bytes(manifest_payload),
        "cache_manifest_sha256": raw_cache_hash,
        "split_manifest_sha256": raw_split_hash,
        "selected_input_hashes_verified": len(rows),
        "gt_hashes_verified": len({row["scene"] for row in rows}),
        "safety": safety,
    }


def _design_matrices(
    train_rows: list[dict],
    test_rows: list[dict],
    conditions: tuple[str, ...],
    features: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    stats = {}
    for condition in conditions:
        selected = [row for row in train_rows if row["condition"] == condition]
        if not selected:
            raise RuntimeError(f"No training rows for condition {condition}")
        for feature in features:
            values = np.asarray([row[feature] for row in selected], dtype=np.float64)
            std = float(values.std(ddof=0))
            if not math.isfinite(std) or std <= 1e-12:
                raise RuntimeError(f"Degenerate feature {feature}/{condition}")
            stats[(condition, feature)] = (float(values.mean()), std)

    def matrix(source: list[dict]) -> np.ndarray:
        values = []
        for row in source:
            condition = row["condition"]
            vector = [1.0 if condition == name else 0.0 for name in conditions]
            for name in conditions:
                for feature in features:
                    if condition == name:
                        mean, std = stats[(name, feature)]
                        vector.append((float(row[feature]) - mean) / std)
                    else:
                        vector.append(0.0)
            values.append(vector)
        return np.asarray(values, dtype=np.float64)

    return matrix(train_rows), matrix(test_rows)


def cross_fit_rows(rows: list[dict], config: dict) -> list[dict]:
    conditions = tuple(config["primary_conditions"])
    features = tuple(config["severity"]["features"])
    ridge_alpha = float(config["estimators"]["ridge_alpha"])
    logistic_c = float(config["estimators"]["logistic_c"])
    folds = sorted({int(row["fold"]) for row in rows})
    adjusted: list[dict] = []
    for fold in folds:
        train = [row for row in rows if int(row["fold"]) != fold]
        test = [row for row in rows if int(row["fold"]) == fold]
        train_x, test_x = _design_matrices(train, test, conditions, features)
        outcome = np.asarray([row["gain_psnr"] for row in train], dtype=np.float64)
        treatment = np.asarray(
            [1 if row["generator"] == "B" else 0 for row in train], dtype=np.int64
        )
        ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
        ridge.fit(train_x, outcome)
        logistic = LogisticRegression(
            C=logistic_c,
            solver="lbfgs",
            max_iter=2000,
            random_state=0,
        )
        logistic.fit(train_x, treatment)
        predictions = ridge.predict(test_x)
        propensities = logistic.predict_proba(test_x)[:, 1]
        for row, prediction, propensity in zip(test, predictions, propensities):
            value = dict(row)
            value["predicted_gain_from_severity"] = float(prediction)
            value["residual_gain"] = float(row["gain_psnr"] - prediction)
            value["propensity_b"] = float(propensity)
            value["overlap_weight"] = float(
                1.0 - propensity if row["generator"] == "B" else propensity
            )
            adjusted.append(value)
    adjusted.sort(key=lambda row: (row["scene"], row["condition"], row["generator"]))
    return adjusted


def _percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _paired_scene_values(rows: list[dict], field: str, conditions: set[str]) -> dict[str, float]:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["condition"] in conditions:
            grouped[(row["scene"], row["condition"])][row["generator"]] = float(
                row[field]
            )
    by_scene: dict[str, list[float]] = defaultdict(list)
    for (scene, condition), values in grouped.items():
        if set(values) != {"A", "B"}:
            raise RuntimeError(f"Unpaired row: {(scene, condition)}")
        by_scene[scene].append(values["B"] - values["A"])
    return {scene: float(np.mean(values)) for scene, values in by_scene.items()}


def _paired_summary(
    rows: list[dict], field: str, conditions: set[str], config: dict
) -> dict:
    by_scene = _paired_scene_values(rows, field, conditions)
    scenes = sorted(by_scene)
    values = np.asarray([by_scene[scene] for scene in scenes], dtype=np.float64)
    rng = np.random.default_rng(int(config["estimators"]["bootstrap_seed"]))
    replicates = int(config["estimators"]["bootstrap_replicates"])
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    bootstrap = values[indices].mean(axis=1)
    sign_rng = np.random.default_rng(int(config["estimators"]["sign_flip_seed"]))
    sign_replicates = int(config["estimators"]["sign_flip_replicates"])
    signs = sign_rng.choice((-1.0, 1.0), size=(sign_replicates, len(values)))
    null = (signs * values[None, :]).mean(axis=1)
    estimate = float(values.mean())
    p_value = float((1 + np.sum(np.abs(null) >= abs(estimate))) / (sign_replicates + 1))
    return {
        "estimate_db": estimate,
        "ci95": _percentile_interval(bootstrap),
        "sign_flip_p_two_sided": p_value,
        "scene_count": len(scenes),
        "scene_values": {scene: by_scene[scene] for scene in scenes},
    }


def _weighted_mean(rows: list[dict], generator: str) -> float:
    selected = [row for row in rows if row["generator"] == generator]
    weights = np.asarray([row["overlap_weight"] for row in selected], dtype=np.float64)
    values = np.asarray([row["gain_psnr"] for row in selected], dtype=np.float64)
    if not len(values) or float(weights.sum()) <= 0:
        raise RuntimeError(f"Invalid overlap weights for generator {generator}")
    return float(np.sum(weights * values) / np.sum(weights))


def _weighted_summary(rows: list[dict], conditions: set[str], config: dict) -> dict:
    selected = [row for row in rows if row["condition"] in conditions]
    mean_a = _weighted_mean(selected, "A")
    mean_b = _weighted_mean(selected, "B")
    scenes = sorted({row["scene"] for row in selected})
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        by_scene[row["scene"]].append(row)
    rng = np.random.default_rng(int(config["estimators"]["bootstrap_seed"]) + 1)
    bootstrap = []
    for _ in range(int(config["estimators"]["bootstrap_replicates"])):
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        replicate_rows = [row for scene in sampled for row in by_scene[str(scene)]]
        bootstrap.append(
            _weighted_mean(replicate_rows, "B")
            - _weighted_mean(replicate_rows, "A")
        )
    return {
        "estimate_db": mean_b - mean_a,
        "ci95": _percentile_interval(np.asarray(bootstrap, dtype=np.float64)),
        "weighted_gain_a_db": mean_a,
        "weighted_gain_b_db": mean_b,
        "scene_count": len(scenes),
    }


def _effective_sample_size(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.sum(weights * weights))


def _weighted_moments(rows: list[dict], feature: str, generator: str) -> tuple[float, float]:
    selected = [row for row in rows if row["generator"] == generator]
    weights = np.asarray([row["overlap_weight"] for row in selected], dtype=np.float64)
    values = np.asarray([row[feature] for row in selected], dtype=np.float64)
    mean = float(np.sum(weights * values) / np.sum(weights))
    variance = float(np.sum(weights * (values - mean) ** 2) / np.sum(weights))
    return mean, variance


def _balance(rows: list[dict], features: tuple[str, ...]) -> dict:
    smd = {}
    for feature in features:
        mean_a, var_a = _weighted_moments(rows, feature, "A")
        mean_b, var_b = _weighted_moments(rows, feature, "B")
        scale = math.sqrt(max((var_a + var_b) / 2.0, 1e-24))
        smd[feature] = float((mean_b - mean_a) / scale)
    return {
        "weighted_smd": smd,
        "max_absolute_weighted_smd": max(abs(value) for value in smd.values()),
    }


def _overlap_summary(rows: list[dict], config: dict) -> dict:
    lower, upper = [float(value) for value in config["overlap_gate"]["propensity_interval"]]
    features = tuple(config["severity"]["features"])
    by_generator = {}
    for generator in ("A", "B"):
        selected = [row for row in rows if row["generator"] == generator]
        propensities = np.asarray([row["propensity_b"] for row in selected])
        weights = np.asarray([row["overlap_weight"] for row in selected])
        by_generator[generator] = {
            "rows": len(selected),
            "fraction_in_propensity_interval": float(
                np.mean((propensities >= lower) & (propensities <= upper))
            ),
            "effective_sample_size": _effective_sample_size(weights),
            "propensity_min": float(propensities.min()),
            "propensity_max": float(propensities.max()),
        }
    families = {
        name: _balance(
            [row for row in rows if row["condition"] in set(conditions)], features
        )
        for name, conditions in config["families"].items()
    }
    return {
        "by_generator": by_generator,
        "overall_balance": _balance(rows, features),
        "family_balance": families,
    }


def _decision_reversal(rows: list[dict], conditions: set[str], threshold: float) -> dict:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["condition"] in conditions:
            grouped[(row["scene"], row["condition"])][row["generator"]] = float(
                row["gain_psnr"]
            )
    b_only = 0
    a_only = 0
    same = 0
    for values in grouped.values():
        a_positive = values["A"] >= threshold
        b_positive = values["B"] >= threshold
        if b_positive and not a_positive:
            b_only += 1
        elif a_positive and not b_positive:
            a_only += 1
        else:
            same += 1
    total = len(grouped)
    return {
        "threshold_db": threshold,
        "pairs": total,
        "b_positive_a_not": b_only,
        "a_positive_b_not": a_only,
        "same_decision": same,
        "reversal_rate": float((b_only + a_only) / total),
    }


def summarize_adjusted_rows(rows: list[dict], config: dict) -> dict:
    primary = set(config["primary_conditions"])
    overlap = _overlap_summary(rows, config)
    raw = _paired_summary(rows, "gain_psnr", primary, config)
    residualized = _paired_summary(rows, "residual_gain", primary, config)
    weighted = _weighted_summary(rows, primary, config)
    family = {
        name: _paired_summary(rows, "residual_gain", set(conditions), config)
        for name, conditions in config["families"].items()
    }
    condition = {
        name: _paired_summary(rows, "residual_gain", {name}, config)
        for name in config["primary_conditions"]
    }
    reversal = _decision_reversal(
        rows, primary, float(config["outcome"]["decision_threshold_db"])
    )

    overlap_gate = config["overlap_gate"]
    overlap_checks = {
        "fraction_in_interval_a": overlap["by_generator"]["A"][
            "fraction_in_propensity_interval"
        ]
        >= float(overlap_gate["minimum_fraction_in_interval_per_generator"]),
        "fraction_in_interval_b": overlap["by_generator"]["B"][
            "fraction_in_propensity_interval"
        ]
        >= float(overlap_gate["minimum_fraction_in_interval_per_generator"]),
        "effective_sample_size_a": overlap["by_generator"]["A"][
            "effective_sample_size"
        ]
        >= float(overlap_gate["minimum_effective_sample_size_per_generator"]),
        "effective_sample_size_b": overlap["by_generator"]["B"][
            "effective_sample_size"
        ]
        >= float(overlap_gate["minimum_effective_sample_size_per_generator"]),
        "overall_balance": overlap["overall_balance"]["max_absolute_weighted_smd"]
        <= float(overlap_gate["maximum_absolute_weighted_smd_overall"]),
        "rain_balance": overlap["family_balance"]["rain"][
            "max_absolute_weighted_smd"
        ]
        <= float(overlap_gate["maximum_absolute_weighted_smd_per_weather_family"]),
        "snow_balance": overlap["family_balance"]["snow"][
            "max_absolute_weighted_smd"
        ]
        <= float(overlap_gate["maximum_absolute_weighted_smd_per_weather_family"]),
    }
    effect_gate = config["effect_gate"]
    effect_checks = {
        "primary_effect_size": residualized["estimate_db"]
        >= float(effect_gate["minimum_primary_adjusted_interaction_db"]),
        "primary_ci_lower": residualized["ci95"][0] > 0.0,
        "primary_sign_flip": residualized["sign_flip_p_two_sided"]
        <= float(effect_gate["primary_sign_flip_p_max"]),
        "overlap_weighted_effect_size": weighted["estimate_db"]
        >= float(effect_gate["overlap_weighted_interaction_must_be_at_least_db"]),
        "overlap_weighted_ci_lower": weighted["ci95"][0] > 0.0,
        "rain_point_estimate": family["rain"]["estimate_db"]
        >= float(effect_gate["minimum_family_point_estimate_db"]),
        "snow_point_estimate": family["snow"]["estimate_db"]
        >= float(effect_gate["minimum_family_point_estimate_db"]),
        "positive_condition_count": sum(
            value["estimate_db"] > 0.0 for value in condition.values()
        )
        >= int(effect_gate["minimum_positive_condition_count"]),
        "raw_decision_reversal_rate": reversal["reversal_rate"]
        >= float(effect_gate["minimum_raw_decision_reversal_rate"]),
    }
    overlap_pass = all(overlap_checks.values())
    effect_pass = all(effect_checks.values())
    if not overlap_pass:
        decision = config["decisions"]["overlap_failure"]
    elif not effect_pass:
        decision = config["decisions"]["effect_failure"]
    else:
        decision = config["decisions"]["pass"]
    return {
        "overlap": overlap,
        "effects": {
            "raw_paired_interaction": raw,
            "cross_fitted_residualized_interaction": residualized,
            "overlap_weighted_interaction": weighted,
            "residualized_family_interactions": family,
            "residualized_condition_interactions": condition,
            "raw_recipe_decision_reversal": reversal,
        },
        "gates": {
            "overlap_checks": overlap_checks,
            "overlap_pass": overlap_pass,
            "effect_checks": effect_checks,
            "effect_pass": effect_pass,
        },
        "decision": decision,
    }


def analyze(config_path: Path) -> dict:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent.parent
    s2b_path = (base / config["sources"]["s2b_archive"]).resolve()
    raw_path = (base / config["sources"]["s2d_raw_archive"]).resolve()
    for path, expected_hash in (
        (s2b_path, config["sources"]["s2b_archive_sha256"]),
        (raw_path, config["sources"]["s2d_raw_archive_sha256"]),
    ):
        actual = sha256_file(path)
        if actual != expected_hash:
            raise RuntimeError(f"Archive hash mismatch for {path}: {actual}")

    expected = config["expected"]
    primary_conditions = tuple(config["primary_conditions"])
    rows, s2b_info = _load_s2b_rows(
        s2b_path, primary_conditions, int(expected["s2b_metric_rows"])
    )
    if len(rows) != int(expected["primary_rows"]):
        raise RuntimeError(f"Expected {expected['primary_rows']} primary rows, found {len(rows)}")
    rows, raw_info = _attach_lpips(rows, raw_path, expected)
    if s2b_info["cache_manifest_sha256"] != raw_info["cache_manifest_sha256"]:
        raise RuntimeError("S2b/S2d cache manifest hash mismatch")
    adjusted = cross_fit_rows(rows, config)
    summary = summarize_adjusted_rows(adjusted, config)
    scenes = {row["scene"] for row in adjusted}
    pairs = {(row["scene"], row["condition"]) for row in adjusted}
    if len(scenes) != int(expected["scenes"]) or len(pairs) != int(expected["primary_pairs"]):
        raise RuntimeError("Unexpected scene/pair counts")
    if raw_info["safety"] != {"cdd11_test_opened": False, "raw_exported": True}:
        raise RuntimeError(f"Unexpected S2d safety state: {raw_info['safety']}")
    safety = {
        "cpu_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "restoration_inference_used": False,
        "training_used": False,
        "checkpoint_loaded": False,
        "network_download_used": False,
        "sealed_partition_accessed": False,
        "renderer_retuned": False,
        "gate_retuned_after_result": False,
    }
    return {
        "stage": "S2E_SEVERITY_CONDITIONED_FALSIFICATION",
        "protocol_version": config["protocol_version"],
        "analysis_scope": config["analysis_scope"],
        "config_sha256": sha256_file(config_path),
        "sources": {
            "s2b": s2b_info,
            "s2d_raw": raw_info,
            "s2b_archive_sha256": config["sources"]["s2b_archive_sha256"],
            "s2d_raw_archive_sha256": config["sources"]["s2d_raw_archive_sha256"],
        },
        "counts": {
            "scenes": len(scenes),
            "primary_conditions": len(primary_conditions),
            "primary_rows": len(adjusted),
            "primary_pairs": len(pairs),
        },
        **summary,
        "adjusted_rows": adjusted,
        "safety": safety,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_limits": [
            "The raw S2b outcome and generator-specific result were already known; this is exploratory falsification, not confirmation.",
            "Severity adjustment controls only observed PSNR, SSIM and resized LPIPS, not all image-formation differences.",
            "R0/R1 are training recipes for one backbone, not five independent restoration systems.",
            "A positive result only permits a separately locked five-model A/B pilot.",
        ],
    }


def main() -> None:
    args = parse_args()
    result = analyze(Path(args.config).resolve())
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "decision": result["decision"],
                "counts": result["counts"],
                "gates": result["gates"],
                "effects": result["effects"],
                "elapsed_seconds": result["elapsed_seconds"],
                "output": str(output),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
