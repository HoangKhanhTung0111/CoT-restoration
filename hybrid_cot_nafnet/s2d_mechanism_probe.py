"""CPU-only paired probe for degradation-mechanism cues in S2d.

The probe reads only the locked 25-scene S2b development manifest and the
materialized Generator-A/Generator-B cache.  It never enumerates CDD-11_test,
does not load a restoration checkpoint, and does not train a restorer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .datasets.s2b_coverage import (
    A_PROBE_TYPES,
    GENERATED_TYPES,
    find_cdd11_train_dir,
    load_cv_manifest,
    pixel_sha256,
    sha256_file,
)
from .prepare_s2b_coverage import validate_cache


PROTOCOL_VERSION = "s2d-mechanism-probe-v1"
CONDITIONS = tuple(name.removeprefix("b_") for name in GENERATED_TYPES)
FEATURE_SIZE = 128
RIDGE = 1e-3
PERMUTATIONS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--export-raw", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _resize(image: np.ndarray, size: int = FEATURE_SIZE) -> np.ndarray:
    pil = Image.fromarray(image)
    resized = pil.resize((size, size), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _color_features(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
    values: list[float] = []
    names: list[str] = []
    quantiles = (0.05, 0.25, 0.5, 0.75, 0.95)
    for channel, label in enumerate(("r", "g", "b")):
        data = image[..., channel].reshape(-1).astype(np.float64)
        values.extend((float(data.mean()), float(data.std())))
        names.extend((f"{label}_mean", f"{label}_std"))
        values.extend(float(item) for item in np.quantile(data, quantiles))
        names.extend(f"{label}_q{int(q * 100):02d}" for q in quantiles)
        histogram, _ = np.histogram(data, bins=16, range=(0.0, 1.0), density=False)
        histogram = histogram.astype(np.float64) / max(1, histogram.sum())
        values.extend(float(item) for item in histogram)
        names.extend(f"{label}_hist{index:02d}" for index in range(16))
    return np.asarray(values, dtype=np.float64), names


def _spectrum_features(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
    gray = (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    ).astype(np.float64)
    gray -= gray.mean()
    window = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
    power = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray * window))) ** 2)
    yy, xx = np.indices(power.shape)
    cy, cx = (np.asarray(power.shape) - 1.0) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius /= max(radius.max(), 1.0)
    edges = np.linspace(0.0, 1.0, 17)
    values = []
    for index in range(16):
        mask = (radius >= edges[index]) & (radius < edges[index + 1])
        values.append(float(power[mask].mean()) if mask.any() else 0.0)
    values_array = np.asarray(values, dtype=np.float64)
    values_array -= values_array.mean()
    scale = values_array.std()
    if scale > 1e-12:
        values_array /= scale
    return values_array, [f"spectrum_{index:02d}" for index in range(16)]


def _haar_features(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
    current = image.astype(np.float64)
    values: list[float] = []
    names: list[str] = []
    for level in range(1, 4):
        height = current.shape[0] - current.shape[0] % 2
        width = current.shape[1] - current.shape[1] % 2
        current = current[:height, :width]
        a = current[0::2, 0::2]
        b = current[0::2, 1::2]
        c = current[1::2, 0::2]
        d = current[1::2, 1::2]
        ll = (a + b + c + d) / 4.0
        bands = {
            "lh": (a - b + c - d) / 4.0,
            "hl": (a + b - c - d) / 4.0,
            "hh": (a - b - c + d) / 4.0,
        }
        for band_name, band in bands.items():
            for channel, label in enumerate(("r", "g", "b")):
                channel_values = band[..., channel]
                values.extend(
                    (
                        float(np.mean(np.abs(channel_values))),
                        float(np.sqrt(np.mean(channel_values * channel_values))),
                    )
                )
                names.extend(
                    (
                        f"haar_l{level}_{band_name}_{label}_mae",
                        f"haar_l{level}_{band_name}_{label}_rms",
                    )
                )
        current = ll
    return np.asarray(values, dtype=np.float64), names


def extract_feature_families(image_u8: np.ndarray) -> dict[str, tuple[np.ndarray, list[str]]]:
    image = _resize(image_u8)
    color, color_names = _color_features(image)
    spectrum, spectrum_names = _spectrum_features(image)
    wavelet, wavelet_names = _haar_features(image)
    return {
        "color": (color, color_names),
        "spectrum": (spectrum, spectrum_names),
        "wavelet": (wavelet, wavelet_names),
        "combined": (
            np.concatenate((color, spectrum, wavelet)),
            color_names + spectrum_names + wavelet_names,
        ),
    }


def extract_residual_feature_families(
    image_u8: np.ndarray, clean_u8: np.ndarray
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Extract signed degradation-residual features without leaking scene groups."""
    residual = _resize(image_u8) - _resize(clean_u8)
    values: list[float] = []
    names: list[str] = []
    quantiles = (0.05, 0.25, 0.5, 0.75, 0.95)
    for channel, label in enumerate(("r", "g", "b")):
        data = residual[..., channel].reshape(-1).astype(np.float64)
        values.extend(
            (
                float(data.mean()),
                float(data.std()),
                float(np.mean(np.abs(data))),
                float(np.sqrt(np.mean(data * data))),
            )
        )
        names.extend(
            (
                f"residual_{label}_mean",
                f"residual_{label}_std",
                f"residual_{label}_mae",
                f"residual_{label}_rms",
            )
        )
        values.extend(float(item) for item in np.quantile(data, quantiles))
        names.extend(f"residual_{label}_q{int(q * 100):02d}" for q in quantiles)
    statistics = np.asarray(values, dtype=np.float64)
    spectrum, spectrum_names = _spectrum_features(residual)
    wavelet, wavelet_names = _haar_features(residual)
    spectrum_names = [f"residual_{name}" for name in spectrum_names]
    wavelet_names = [f"residual_{name}" for name in wavelet_names]
    return {
        "statistics": (statistics, names),
        "spectrum": (spectrum, spectrum_names),
        "wavelet": (wavelet, wavelet_names),
        "combined": (
            np.concatenate((statistics, spectrum, wavelet)),
            names + spectrum_names + wavelet_names,
        ),
    }


def pair_moment_match(left_u8: np.ndarray, right_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Give a paired A/B view identical per-channel means/stds for sensitivity analysis."""
    left = left_u8.astype(np.float64) / 255.0
    right = right_u8.astype(np.float64) / 255.0
    outputs = []
    target_mean = (left.mean(axis=(0, 1)) + right.mean(axis=(0, 1))) / 2.0
    target_std = (left.std(axis=(0, 1)) + right.std(axis=(0, 1))) / 2.0
    for image in (left, right):
        mean = image.mean(axis=(0, 1))
        std = image.std(axis=(0, 1))
        safe_std = np.where(std > 1e-8, std, 1.0)
        matched = (image - mean) / safe_std * target_std + target_mean
        outputs.append(np.rint(np.clip(matched, 0, 1) * 255.0).astype(np.uint8))
    return outputs[0], outputs[1]


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    recalls = []
    for label in (-1, 1):
        mask = labels == label
        if not mask.any():
            raise ValueError("Both classes are required")
        recalls.append(float(np.mean(predictions[mask] == label)))
    return float(np.mean(recalls))


def _fit_ridge(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std > 1e-10, std, 1.0)
    train = (train_x - mean) / std
    test = (test_x - mean) / std
    train = np.column_stack((np.ones(len(train)), train))
    test = np.column_stack((np.ones(len(test)), test))
    penalty = np.eye(train.shape[1], dtype=np.float64) * RIDGE
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(train.T @ train + penalty, train.T @ train_y)
    return test @ weights


def grouped_probe(
    features: np.ndarray,
    labels: np.ndarray,
    scenes: np.ndarray,
    fold_by_scene: dict[str, int],
) -> dict:
    fold_rows = []
    all_labels, all_predictions, all_scores = [], [], []
    for fold in range(5):
        test_mask = np.asarray([fold_by_scene[str(scene)] == fold for scene in scenes])
        train_mask = ~test_mask
        scores = _fit_ridge(features[train_mask], labels[train_mask], features[test_mask])
        predictions = np.where(scores >= 0, 1, -1)
        fold_labels = labels[test_mask]
        fold_rows.append(
            {
                "fold": fold,
                "train_scenes": len(set(scenes[train_mask])),
                "test_scenes": len(set(scenes[test_mask])),
                "test_samples": int(test_mask.sum()),
                "balanced_accuracy": _balanced_accuracy(fold_labels, predictions),
            }
        )
        all_labels.extend(int(value) for value in fold_labels)
        all_predictions.extend(int(value) for value in predictions)
        all_scores.extend(float(value) for value in scores)
    label_array = np.asarray(all_labels)
    prediction_array = np.asarray(all_predictions)
    return {
        "balanced_accuracy": _balanced_accuracy(label_array, prediction_array),
        "folds": fold_rows,
        "score_mean_a": float(np.mean(np.asarray(all_scores)[label_array == -1])),
        "score_mean_b": float(np.mean(np.asarray(all_scores)[label_array == 1])),
    }


def _permutation_control(
    features: np.ndarray,
    labels: np.ndarray,
    scenes: np.ndarray,
    conditions: np.ndarray,
    fold_by_scene: dict[str, int],
) -> dict:
    values = []
    for draw in range(PERMUTATIONS):
        rng = np.random.default_rng(20260925 + draw)
        shuffled = labels.copy()
        for scene in sorted(set(scenes)):
            for condition in sorted(set(conditions)):
                indices = np.flatnonzero((scenes == scene) & (conditions == condition))
                if len(indices) != 2:
                    raise RuntimeError("Every scene/condition must have exactly one A/B pair")
                shuffled[indices] = shuffled[indices][rng.permutation(2)]
        values.append(grouped_probe(features, shuffled, scenes, fold_by_scene)["balanced_accuracy"])
    return {
        "draws": PERMUTATIONS,
        "mean": float(np.mean(values)),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_probe(
    data_root: Path,
    manifest_path: Path,
    cache_root: Path,
    output_dir: Path,
    *,
    export_raw: bool = True,
) -> dict:
    manifest = load_cv_manifest(manifest_path)
    cache_manifest = validate_cache(manifest_path, cache_root)
    if cache_manifest.get("record_count") != 25 * 16:
        raise RuntimeError("S2d requires all 400 paired A/B generated views")
    train_dir = find_cdd11_train_dir(data_root)
    fold_by_scene: dict[str, int] = {}
    for fold in manifest["folds"]:
        for scene in fold["validation"]:
            if scene in fold_by_scene:
                raise RuntimeError(f"Scene occurs in multiple validation folds: {scene}")
            fold_by_scene[scene] = int(fold["fold"])
    if len(fold_by_scene) != 25:
        raise RuntimeError("Expected exactly 25 out-of-fold scenes")

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_rows: dict[str, list[np.ndarray]] = {
        name: [] for name in ("color", "spectrum", "wavelet", "combined")
    }
    matched_rows: dict[str, list[np.ndarray]] = {
        name: [] for name in ("color", "spectrum", "wavelet", "combined")
    }
    gt_rows: dict[str, list[np.ndarray]] = {
        name: [] for name in ("color", "spectrum", "wavelet", "combined")
    }
    residual_rows: dict[str, list[np.ndarray]] = {
        name: [] for name in ("statistics", "spectrum", "wavelet", "combined")
    }
    feature_names: dict[str, list[str]] = {}
    residual_feature_names: dict[str, list[str]] = {}
    sample_rows = []
    raw_records = []
    gt_records = []
    duplicate_pair_members = 0

    by_scene = {record["scene"]: record for record in manifest["scenes"]}
    for scene in sorted(by_scene):
        record = by_scene[scene]
        clean_path = train_dir / "clear" / record["file"]
        if pixel_sha256(clean_path) != record["pixel_sha256"]:
            raise RuntimeError(f"Clean target hash changed: {scene}")
        clean = _read_rgb(clean_path)
        if export_raw:
            destination = output_dir / "raw" / "gt" / f"{scene}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(clean).save(destination)
            gt_records.append(
                {
                    "scene": scene,
                    "file": destination.relative_to(output_dir).as_posix(),
                    "file_sha256": sha256_file(destination),
                    "pixel_sha256": _array_sha256(clean),
                    "size_wh": [int(clean.shape[1]), int(clean.shape[0])],
                }
            )
        clean_features = extract_feature_families(clean)
        for condition in CONDITIONS:
            pair = {}
            for generator, prefix, label in (("A", "a_", -1), ("B", "b_", 1)):
                degradation_type = prefix + condition
                source = cache_root / "generated" / degradation_type / record["file"]
                image = _read_rgb(source)
                if image.shape != clean.shape:
                    raise RuntimeError(f"A/B/GT shape mismatch: {source}")
                pair[generator] = image
                if export_raw:
                    destination = (
                        output_dir / "raw" / generator.lower() / condition / f"{scene}.png"
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    raw_records.append(
                        {
                            "scene": scene,
                            "condition": condition,
                            "generator": generator,
                            "label": label,
                            "fold": fold_by_scene[scene],
                            "file": destination.relative_to(output_dir).as_posix(),
                            "file_sha256": sha256_file(destination),
                            "pixel_sha256": _array_sha256(image),
                            "size_wh": [int(image.shape[1]), int(image.shape[0])],
                        }
                    )
            matched_a, matched_b = pair_moment_match(pair["A"], pair["B"])
            if np.array_equal(pair["A"], pair["B"]):
                duplicate_pair_members += 1
            for generator, label, matched in (("A", -1, matched_a), ("B", 1, matched_b)):
                features = extract_feature_families(pair[generator])
                matched_features = extract_feature_families(matched)
                residual_features = extract_residual_feature_families(
                    pair[generator], clean
                )
                for family in feature_rows:
                    vector, names = features[family]
                    matched_vector, matched_names = matched_features[family]
                    if names != matched_names:
                        raise AssertionError("Matched feature schema changed")
                    if family in feature_names and feature_names[family] != names:
                        raise AssertionError("Feature schema changed between samples")
                    feature_names[family] = names
                    feature_rows[family].append(vector)
                    matched_rows[family].append(matched_vector)
                    gt_rows[family].append(clean_features[family][0])
                for family in residual_rows:
                    vector, names = residual_features[family]
                    if (
                        family in residual_feature_names
                        and residual_feature_names[family] != names
                    ):
                        raise AssertionError("Residual feature schema changed between samples")
                    residual_feature_names[family] = names
                    residual_rows[family].append(vector)
                sample_rows.append(
                    {
                        "scene": scene,
                        "condition": condition,
                        "generator": generator,
                        "label": label,
                        "fold": fold_by_scene[scene],
                    }
                )

    if len(sample_rows) != 25 * 8 * 2:
        raise RuntimeError(f"Unexpected S2d sample count: {len(sample_rows)}")
    labels = np.asarray([row["label"] for row in sample_rows], dtype=np.int64)
    scenes = np.asarray([row["scene"] for row in sample_rows])
    conditions = np.asarray([row["condition"] for row in sample_rows])
    probes = {}
    matched_probes = {}
    gt_controls = {}
    residual_probes = {}
    for family in feature_rows:
        matrix = np.vstack(feature_rows[family])
        matched_matrix = np.vstack(matched_rows[family])
        gt_matrix = np.vstack(gt_rows[family])
        if not (np.isfinite(matrix).all() and np.isfinite(matched_matrix).all()):
            raise RuntimeError(f"Non-finite features in {family}")
        probes[family] = grouped_probe(matrix, labels, scenes, fold_by_scene)
        matched_probes[family] = grouped_probe(
            matched_matrix, labels, scenes, fold_by_scene
        )
        gt_controls[family] = grouped_probe(gt_matrix, labels, scenes, fold_by_scene)
    for family in residual_rows:
        residual_matrix = np.vstack(residual_rows[family])
        if not np.isfinite(residual_matrix).all():
            raise RuntimeError(f"Non-finite residual features in {family}")
        residual_probes[family] = grouped_probe(
            residual_matrix, labels, scenes, fold_by_scene
        )

    best_family = max(probes, key=lambda name: probes[name]["balanced_accuracy"])
    permutation = _permutation_control(
        np.vstack(feature_rows[best_family]),
        labels,
        scenes,
        conditions,
        fold_by_scene,
    )
    sample_csv = output_dir / "sample_index.csv"
    _write_csv(
        sample_csv,
        ["scene", "condition", "generator", "label", "fold"],
        sample_rows,
    )
    raw_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "split_manifest_sha256": sha256_file(manifest_path),
        "cache_manifest_sha256": sha256_file(cache_root / "cache_manifest.json"),
        "cdd11_test_opened": False,
        "raw_exported": export_raw,
        "input_record_count": len(raw_records) if export_raw else 0,
        "gt_record_count": len(gt_records) if export_raw else 0,
        "inputs": raw_records,
        "ground_truth": gt_records,
    }
    raw_manifest_path = output_dir / "s2d_raw_manifest.json"
    raw_manifest_path.write_text(
        json.dumps(raw_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "COMPLETE",
        "scope": "cpu_only_input_mechanism_probe_no_restoration_training",
        "scene_count": 25,
        "condition_count": 8,
        "sample_count": len(sample_rows),
        "fold_count": 5,
        "feature_size": FEATURE_SIZE,
        "ridge": RIDGE,
        "raw_exported": export_raw,
        "raw_manifest_sha256": sha256_file(raw_manifest_path),
        "sample_index_sha256": sha256_file(sample_csv),
        "feature_dimensions": {
            family: len(names) for family, names in feature_names.items()
        },
        "residual_feature_dimensions": {
            family: len(names) for family, names in residual_feature_names.items()
        },
        "input_only_probes": probes,
        "gt_assisted_residual_probes": residual_probes,
        "pair_moment_matched_probes": matched_probes,
        "clean_gt_negative_controls": gt_controls,
        "best_raw_family": best_family,
        "best_raw_balanced_accuracy": probes[best_family]["balanced_accuracy"],
        "best_matched_balanced_accuracy": max(
            value["balanced_accuracy"] for value in matched_probes.values()
        ),
        "label_permutation_control_for_best_raw_family": permutation,
        "audit": {
            "scene_grouped_cv": True,
            "factor_stratified_by_complete_pairs": True,
            "a_b_pair_count": 25 * 8,
            "duplicate_pair_members": duplicate_pair_members,
            "cdd11_test_opened": False,
            "restoration_checkpoint_loaded": False,
            "restoration_training_performed": False,
        },
        "interpretation_limits": [
            "A/B decodability is evidence of mechanism-specific cues, not proof that a restorer uses them.",
            "Residual probes use clean GT and are therefore audit-only, not deployable detectors.",
            "Intermediate restoration features are not included because the archived S2b bundle contains no R0/R1 checkpoints.",
            "Pair moment matching removes only first- and second-order per-channel statistics.",
            "The 25-scene development pool is a pilot, not a CVPR-scale benchmark.",
        ],
    }
    result_path = output_dir / "s2d_probe_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    result = run_probe(
        Path(args.data_root).resolve(),
        Path(args.manifest).resolve(),
        Path(args.cache_root).resolve(),
        Path(args.output_dir).resolve(),
        export_raw=bool(args.export_raw),
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
