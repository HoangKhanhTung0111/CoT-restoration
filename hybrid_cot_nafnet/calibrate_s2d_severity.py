"""F2 input-only severity calibration for the frozen S2d audit."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity

from .audit_low_weather_replication import unique_scene_views
from . import low_weather_generator_b as generator_b
from . import low_weather_generator_c as generator_c


FEATURES = (
    "input_psnr",
    "input_ssim",
    "input_lpips",
    "mean_luminance_shift",
    "contrast_ratio",
    "radial_spectrum_distance",
)
CONDITIONS = generator_c.CONDITIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-scenes", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _luma(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32) / 255.0
    return (
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    ).astype(np.float32)


def _radial_spectrum(luma: np.ndarray) -> np.ndarray:
    resized = cv2.resize(luma, (128, 128), interpolation=cv2.INTER_AREA)
    centered = resized - float(resized.mean())
    window = np.outer(np.hanning(128), np.hanning(128)).astype(np.float32)
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(centered * window))))
    yy, xx = np.mgrid[-64:64, -64:64]
    radius = np.sqrt(xx * xx + yy * yy)
    bins = np.minimum((radius / radius.max() * 32).astype(np.int32), 31)
    profile = np.array(
        [spectrum[bins == index].mean() for index in range(32)], dtype=np.float64
    )
    norm = float(np.linalg.norm(profile))
    return profile / norm if norm > 0 else profile


def _lpips_model():
    # This script is CPU-only even on laptops that expose a CUDA runtime.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import lpips
    import torch

    model = lpips.LPIPS(
        net="squeeze", version="0.1", spatial=False, verbose=False
    ).cpu().eval()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("LPIPS unexpectedly left CPU")
    return model, torch


def _metric_rows(
    clean: np.ndarray,
    views: dict[str, np.ndarray],
    model,
    torch_module,
) -> list[dict[str, float | str]]:
    if tuple(views) != CONDITIONS:
        raise RuntimeError("Condition order changed")
    clean_luma = _luma(clean)
    clean_spectrum = _radial_spectrum(clean_luma)
    clean_contrast = max(float(clean_luma.std()), 1e-12)
    arrays = [np.ascontiguousarray(views[name]) for name in CONDITIONS]
    batch = torch_module.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
    target = torch_module.from_numpy(
        np.stack([np.ascontiguousarray(clean)] * len(arrays))
    ).permute(0, 3, 1, 2)
    batch = batch.float().div(127.5).sub(1.0)
    target = target.float().div(127.5).sub(1.0)
    with torch_module.inference_mode():
        lpips_values = model(batch, target).reshape(-1).cpu().numpy()

    rows = []
    for condition, degraded, lpips_value in zip(CONDITIONS, arrays, lpips_values):
        difference = degraded.astype(np.float64) - clean.astype(np.float64)
        mse = float(np.mean(difference * difference))
        psnr = 20.0 * math.log10(255.0) - 10.0 * math.log10(max(mse, 1e-12))
        degraded_luma = _luma(degraded)
        rows.append(
            {
                "condition": condition,
                "input_psnr": psnr,
                "input_ssim": float(
                    structural_similarity(
                        clean, degraded, channel_axis=2, data_range=255
                    )
                ),
                "input_lpips": float(lpips_value),
                "mean_luminance_shift": float(
                    degraded_luma.mean() - clean_luma.mean()
                ),
                "contrast_ratio": float(degraded_luma.std()) / clean_contrast,
                "radial_spectrum_distance": float(
                    np.linalg.norm(_radial_spectrum(degraded_luma) - clean_spectrum)
                ),
                "input_pixel_sha256": pixel_sha256(degraded),
            }
        )
    return rows


def _a_views(clean: np.ndarray, scene: str) -> dict[str, np.ndarray]:
    raw, _ = unique_scene_views("A", clean, f"s2d-f2:{scene}")
    return {
        "low_weak": raw["low_0.6"],
        "low_strong": raw["low_1.0"],
        "rain_haze": raw["weather:rain_haze"],
        "snow_haze": raw["weather:snow_haze"],
        "low_weak_rain_haze": raw["combined_0.6:rain_haze"],
        "low_strong_rain_haze": raw["combined_1.0:rain_haze"],
        "low_weak_snow_haze": raw["combined_0.6:snow_haze"],
        "low_strong_snow_haze": raw["combined_1.0:snow_haze"],
    }


def _b_views(clean: np.ndarray, scene: str) -> dict[str, np.ndarray]:
    realization = generator_b.build_realization(clean, f"s2d-f2:{scene}")
    rain = generator_b.make_views(clean, realization, "rain_haze")
    snow = generator_b.make_views(clean, realization, "snow_haze")
    return {
        "low_weak": rain["low_weak"],
        "low_strong": rain["low_strong"],
        "rain_haze": rain["weather"],
        "snow_haze": snow["weather"],
        "low_weak_rain_haze": rain["combined_weak"],
        "low_strong_rain_haze": rain["combined_strong"],
        "low_weak_snow_haze": snow["combined_weak"],
        "low_strong_snow_haze": snow["combined_strong"],
    }


def _parameters(profile: dict[str, Any]) -> generator_c.ParametersC:
    values = {key: value for key, value in profile.items() if key != "id"}
    return generator_c.ParametersC(**values)


def _c_views(clean: np.ndarray, scene: str, profile: dict[str, Any]):
    candidate = int(profile["id"])
    realization = generator_c.build_realization(
        clean,
        f"s2d-f2:{scene}",
        parameters=_parameters(profile),
        seed_variant=candidate,
    )
    return generator_c.make_views(clean, realization)


def energy_statistic(candidate: np.ndarray, reference: np.ndarray) -> float:
    if candidate.ndim != 2 or reference.ndim != 2 or candidate.shape[1] != reference.shape[1]:
        raise ValueError("Energy-distance inputs must be 2D with equal feature count")
    mean = reference.mean(axis=0)
    scale = reference.std(axis=0)
    scale[scale < 1e-12] = 1.0
    left = (candidate - mean) / scale
    right = (reference - mean) / scale
    value = (
        2.0 * float(cdist(left, right).mean())
        - float(cdist(left, left).mean())
        - float(cdist(right, right).mean())
    )
    return max(value, 0.0)


def analyze_candidates(
    rows: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    tolerances: dict[str, float],
) -> dict:
    summaries = []
    for profile in profiles:
        candidate_id = int(profile["id"])
        conditions = []
        normalized_ratios = []
        failed = 0
        for condition in CONDITIONS:
            reference_rows = [
                row
                for row in rows
                if row["condition"] == condition and row["mechanism"] in {"A", "B"}
            ]
            candidate_rows = [
                row
                for row in rows
                if row["condition"] == condition
                and row["mechanism"] == "C"
                and row["candidate_id"] == candidate_id
            ]
            if not reference_rows or not candidate_rows:
                raise RuntimeError(f"Missing rows for candidate {candidate_id}, {condition}")
            reference = np.array(
                [[float(row[name]) for name in FEATURES] for row in reference_rows]
            )
            candidate = np.array(
                [[float(row[name]) for name in FEATURES] for row in candidate_rows]
            )
            median_differences = {
                name: abs(float(np.median(candidate[:, index]) - np.median(reference[:, index])))
                for index, name in enumerate(FEATURES[:3])
            }
            energy = energy_statistic(candidate, reference)
            ratios = {
                "input_psnr": median_differences["input_psnr"]
                / tolerances["absolute_median_input_psnr_db"],
                "input_ssim": median_differences["input_ssim"]
                / tolerances["absolute_median_input_ssim"],
                "input_lpips": median_differences["input_lpips"]
                / tolerances["absolute_median_input_lpips"],
                "energy": energy / tolerances["standardized_energy_distance"],
            }
            checks = {name: value <= 1.0 for name, value in ratios.items()}
            failed += sum(not value for value in checks.values())
            normalized_ratios.extend(ratios.values())
            conditions.append(
                {
                    "condition": condition,
                    "reference_rows": len(reference_rows),
                    "candidate_rows": len(candidate_rows),
                    "median_differences": median_differences,
                    "standardized_energy_distance": energy,
                    "normalized_acceptance_ratios": ratios,
                    "checks": checks,
                    "pass": all(checks.values()),
                }
            )
        summaries.append(
            {
                "candidate_id": candidate_id,
                "failed_checks": failed,
                "mean_normalized_acceptance_ratio": float(np.mean(normalized_ratios)),
                "all_conditions_pass": failed == 0,
                "conditions": conditions,
            }
        )
    selected = min(
        summaries,
        key=lambda item: (
            item["failed_checks"],
            item["mean_normalized_acceptance_ratio"],
            item["candidate_id"],
        ),
    )
    return {"selected_candidate_id": selected["candidate_id"], "candidates": summaries}


def _load_train_records(manifest: dict, train_root: Path) -> list[dict[str, Any]]:
    if train_root.name != "train":
        raise RuntimeError("--train-root must end in the official train directory")
    records = [record for record in manifest["records"] if record["split"] == "train"]
    if len(records) != 200 or manifest.get("split_counts", {}).get("train") != 200:
        raise RuntimeError("Expected exactly 200 locked BSDS500 train records")
    for record in records:
        if not record["file"].startswith("train/"):
            raise RuntimeError("Manifest train record escaped the train split")
    return records


def _environment(model) -> dict[str, Any]:
    import lpips
    import torch

    cache = Path(torch.hub.get_dir()) / "checkpoints" / "squeezenet1_1-b8a52dc0.pth"
    linear = Path(lpips.__file__).resolve().parent / "weights" / "v0.1" / "squeeze.pth"
    return {
        "python_cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "lpips_parameters_device": sorted({p.device.type for p in model.parameters()}),
        "numpy": np.__version__,
        "opencv_runtime": cv2.__version__,
        "albumentations": importlib.metadata.version("albumentations"),
        "lpips": importlib.metadata.version("lpips"),
        "scikit_image": importlib.metadata.version("scikit-image"),
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "squeezenet_weight_sha256": sha256_file(cache),
        "lpips_linear_weight_sha256": sha256_file(linear),
    }


def run(
    config_path: Path,
    manifest_path: Path,
    train_root: Path,
    max_scenes: int | None,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = _load_train_records(manifest, train_root)
    if max_scenes is not None:
        if max_scenes <= 0 or max_scenes > len(records):
            raise ValueError("--max-scenes must be within the train split")
        records = records[:max_scenes]
    profiles = config["candidate_profiles"]
    if [profile["id"] for profile in profiles] != list(range(8)):
        raise RuntimeError("Candidate IDs must remain exactly 0..7")

    model, torch_module = _lpips_model()
    started = time.perf_counter()
    metric_rows: list[dict[str, Any]] = []
    for scene_index, record in enumerate(records, start=1):
        image_path = train_root / Path(record["file"]).name
        if sha256_file(image_path) != record["file_sha256"]:
            raise RuntimeError(f"Locked train image changed: {record['scene']}")
        with Image.open(image_path) as image:
            clean = np.asarray(image.convert("RGB"), dtype=np.uint8)
        clean_hash = pixel_sha256(clean)
        for mechanism, views in (("A", _a_views(clean, record["scene"])),
                                 ("B", _b_views(clean, record["scene"]))):
            for row in _metric_rows(clean, views, model, torch_module):
                row.update(
                    {
                        "scene": record["scene"],
                        "mechanism": mechanism,
                        "candidate_id": None,
                        "clean_pixel_sha256": clean_hash,
                    }
                )
                metric_rows.append(row)
        for profile in profiles:
            views = _c_views(clean, record["scene"], profile)
            for row in _metric_rows(clean, views, model, torch_module):
                row.update(
                    {
                        "scene": record["scene"],
                        "mechanism": "C",
                        "candidate_id": int(profile["id"]),
                        "clean_pixel_sha256": clean_hash,
                    }
                )
                metric_rows.append(row)
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "progress_scenes": scene_index,
                    "total_scenes": len(records),
                    "elapsed_seconds": round(elapsed, 3),
                }
            ),
            flush=True,
        )

    elapsed = time.perf_counter() - started
    projected = elapsed / len(records) * int(config["expected_scene_count"])
    analysis = analyze_candidates(
        metric_rows, profiles, config["acceptance_tolerances"]
    )
    full_run = len(records) == int(config["expected_scene_count"])
    runtime_pass = projected <= float(
        config["runtime_gate"]["maximum_projected_full_seconds"]
    )
    if not full_run:
        status = "PASS_RUNTIME_SMOKE" if runtime_pass else "FAIL_RUNTIME_SMOKE"
    else:
        selected = next(
            item
            for item in analysis["candidates"]
            if item["candidate_id"] == analysis["selected_candidate_id"]
        )
        status = (
            "PASS_SEVERITY_MATCHING"
            if selected["all_conditions_pass"]
            else "FAIL_SEVERITY_MATCHING"
        )
    return {
        "stage": "F2_INPUT_ONLY_SEVERITY_CALIBRATION",
        "status": status,
        "calibration_version": config["calibration_version"],
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest_path),
        "split": "train",
        "scenes_processed": len(records),
        "full_run": full_run,
        "severity_gate_evaluated": full_run,
        "metric_rows": len(metric_rows),
        "elapsed_seconds": elapsed,
        "projected_full_seconds": projected,
        "runtime_gate_pass": runtime_pass,
        "environment": _environment(model),
        "analysis": analysis,
        "safety": {
            "restoration_outputs_used": False,
            "restoration_checkpoints_loaded": False,
            "gpu_used": False,
            "validation_images_decoded": False,
            "final_images_decoded": False,
            "weatherbench_downloaded": False,
        },
        "rows": metric_rows,
    }


def main() -> None:
    args = parse_args()
    result = run(
        Path(args.config).resolve(),
        Path(args.manifest).resolve(),
        Path(args.train_root).resolve(),
        args.max_scenes,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(summary, indent=2, allow_nan=False))
    if result["status"].startswith("FAIL_RUNTIME"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
