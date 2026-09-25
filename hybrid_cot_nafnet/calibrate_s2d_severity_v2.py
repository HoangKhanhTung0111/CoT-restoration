"""One-shot F2-v2 factorized severity calibration on the BSDS500 train split."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import low_weather_generator_c as generator_c
from .calibrate_s2d_severity import (
    CONDITIONS,
    FEATURES,
    _a_views,
    _b_views,
    _environment,
    _load_train_records,
    _lpips_model,
    _metric_rows,
    energy_statistic,
    pixel_sha256,
    sha256_file,
)


FACTOR_CONDITIONS = {
    "low": ("low_weak", "low_strong"),
    "rain": ("rain_haze",),
    "snow": ("snow_haze",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-scenes", type=int)
    return parser.parse_args()


def _parameters(profile: dict[str, Any]) -> generator_c.ParametersC:
    return generator_c.ParametersC(
        **{key: value for key, value in profile.items() if key != "id"}
    )


def _low_views(clean: np.ndarray, scene: str, profile: dict[str, Any]):
    parameters = _parameters(profile)
    candidate = int(profile["id"])
    scene_id = f"s2d-f2-v2:{scene}"
    return {
        "low_weak": generator_c._apply_low(
            clean,
            parameters.weak_gamma,
            generator_c.stable_seed(scene_id, "low_weak", candidate),
            parameters.weak_noise_std,
        ),
        "low_strong": generator_c._apply_low(
            clean,
            parameters.strong_gamma,
            generator_c.stable_seed(scene_id, "low_strong", candidate),
            parameters.strong_noise_std,
        ),
    }


def _weather_view(
    clean: np.ndarray,
    scene: str,
    weather: str,
    profile: dict[str, Any],
) -> dict[str, np.ndarray]:
    parameters = _parameters(profile)
    candidate = int(profile["id"])
    replay, _ = generator_c._sample_weather(
        clean,
        f"s2d-f2-v2:{scene}",
        weather,
        parameters,
        candidate,
    )
    return {weather: generator_c._replay_weather(replay, clean)}


def _condition_gate(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    gate: dict[str, float | str],
) -> dict[str, Any]:
    by_mechanism = {
        mechanism: [row for row in reference_rows if row["mechanism"] == mechanism]
        for mechanism in ("A", "B")
    }
    arrays = {
        mechanism: np.array(
            [[float(row[name]) for name in FEATURES] for row in rows], dtype=np.float64
        )
        for mechanism, rows in by_mechanism.items()
    }
    candidate = np.array(
        [[float(row[name]) for name in FEATURES] for row in candidate_rows],
        dtype=np.float64,
    )
    pooled = np.vstack([arrays["A"], arrays["B"]])
    tolerance_keys = {
        "input_psnr": "absolute_median_input_psnr_db",
        "input_ssim": "absolute_median_input_ssim",
        "input_lpips": "absolute_median_input_lpips",
    }
    median_checks = {}
    median_details = {}
    ratios = []
    for feature_index, feature in enumerate(FEATURES[:3]):
        tolerance = float(gate[tolerance_keys[feature]])
        medians = {
            mechanism: float(np.median(array[:, feature_index]))
            for mechanism, array in arrays.items()
        }
        candidate_median = float(np.median(candidate[:, feature_index]))
        lower = min(medians.values()) - tolerance
        upper = max(medians.values()) + tolerance
        outside = max(lower - candidate_median, candidate_median - upper, 0.0)
        median_checks[feature] = outside <= 0.0
        ratios.append(outside / tolerance)
        median_details[feature] = {
            "a_median": medians["A"],
            "b_median": medians["B"],
            "candidate_median": candidate_median,
            "accepted_interval": [lower, upper],
            "outside_distance": outside,
        }

    candidate_energy = energy_statistic(candidate, pooled)
    component_energies = {
        mechanism: energy_statistic(array, pooled)
        for mechanism, array in arrays.items()
    }
    energy_threshold = max(component_energies.values()) + float(gate["energy_margin"])
    energy_check = candidate_energy <= energy_threshold
    ratios.append(candidate_energy / energy_threshold)
    checks = {**median_checks, "energy": energy_check}
    return {
        "reference_rows": len(reference_rows),
        "candidate_rows": len(candidate_rows),
        "median_details": median_details,
        "candidate_energy": candidate_energy,
        "component_to_pooled_energy": component_energies,
        "energy_threshold": energy_threshold,
        "checks": checks,
        "failed_checks": sum(not value for value in checks.values()),
        "mean_normalized_gate_ratio": float(np.mean(ratios)),
        "pass": all(checks.values()),
    }


def _select_factor(
    factor: str,
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    summaries = []
    for profile in profiles:
        profile_id = int(profile["id"])
        condition_summaries = []
        for condition in FACTOR_CONDITIONS[factor]:
            reference = [row for row in reference_rows if row["condition"] == condition]
            candidate = [
                row
                for row in candidate_rows
                if row["factor"] == factor
                and row["profile_id"] == profile_id
                and row["condition"] == condition
            ]
            summary = _condition_gate(reference, candidate, gate)
            summary["condition"] = condition
            condition_summaries.append(summary)
        summaries.append(
            {
                "profile_id": profile_id,
                "failed_checks": sum(item["failed_checks"] for item in condition_summaries),
                "mean_normalized_gate_ratio": float(
                    np.mean(
                        [item["mean_normalized_gate_ratio"] for item in condition_summaries]
                    )
                ),
                "all_conditions_pass": all(item["pass"] for item in condition_summaries),
                "conditions": condition_summaries,
            }
        )
    selected = min(
        summaries,
        key=lambda item: (
            item["failed_checks"],
            item["mean_normalized_gate_ratio"],
            item["profile_id"],
        ),
    )
    return {"selected_profile_id": selected["profile_id"], "profiles": summaries}


def _final_gate(
    reference_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    conditions = []
    for condition in CONDITIONS:
        summary = _condition_gate(
            [row for row in reference_rows if row["condition"] == condition],
            [row for row in final_rows if row["condition"] == condition],
            gate,
        )
        summary["condition"] = condition
        conditions.append(summary)
    return {
        "conditions_passed": sum(item["pass"] for item in conditions),
        "conditions_total": len(conditions),
        "failed_checks": sum(item["failed_checks"] for item in conditions),
        "all_conditions_pass": all(item["pass"] for item in conditions),
        "conditions": conditions,
    }


def run(
    config_path: Path,
    manifest_path: Path,
    train_root: Path,
    max_scenes: int | None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = _load_train_records(manifest, train_root)
    if max_scenes is not None:
        if max_scenes <= 0 or max_scenes > len(records):
            raise ValueError("--max-scenes must be within the train split")
        records = records[:max_scenes]
    for key in ("low_profiles", "rain_profiles", "snow_profiles"):
        if [profile["id"] for profile in config[key]] != list(range(8)):
            raise RuntimeError(f"{key} IDs must remain exactly 0..7")

    model, torch_module = _lpips_model()
    started = time.perf_counter()
    reference_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    clean_images: dict[str, np.ndarray] = {}

    for scene_index, record in enumerate(records, start=1):
        image_path = train_root / Path(record["file"]).name
        if sha256_file(image_path) != record["file_sha256"]:
            raise RuntimeError(f"Locked train image changed: {record['scene']}")
        with Image.open(image_path) as image:
            clean = np.asarray(image.convert("RGB"), dtype=np.uint8)
        clean_images[record["scene"]] = clean
        clean_hash = pixel_sha256(clean)
        for mechanism, views in (("A", _a_views(clean, record["scene"])),
                                 ("B", _b_views(clean, record["scene"]))):
            for row in _metric_rows(clean, views, model, torch_module):
                row.update({"scene": record["scene"], "mechanism": mechanism,
                            "clean_pixel_sha256": clean_hash})
                reference_rows.append(row)
        for factor, profiles in (
            ("low", config["low_profiles"]),
            ("rain", config["rain_profiles"]),
            ("snow", config["snow_profiles"]),
        ):
            for profile in profiles:
                if factor == "low":
                    views = _low_views(clean, record["scene"], profile)
                else:
                    views = _weather_view(
                        clean,
                        record["scene"],
                        f"{factor}_haze",
                        profile,
                    )
                for row in _metric_rows(clean, views, model, torch_module):
                    row.update(
                        {
                            "scene": record["scene"],
                            "mechanism": "C",
                            "factor": factor,
                            "profile_id": int(profile["id"]),
                            "clean_pixel_sha256": clean_hash,
                        }
                    )
                    factor_rows.append(row)
        if scene_index % 10 == 0 or scene_index == len(records):
            print(json.dumps({"phase": "candidate", "scenes": scene_index,
                              "total": len(records)}), flush=True)

    selections = {}
    for factor, profiles in (
        ("low", config["low_profiles"]),
        ("rain", config["rain_profiles"]),
        ("snow", config["snow_profiles"]),
    ):
        selections[factor] = _select_factor(
            factor,
            reference_rows,
            factor_rows,
            profiles,
            config["common_support_gate"],
        )

    selected_profiles = {
        factor: next(
            profile
            for profile in config[f"{factor}_profiles"]
            if profile["id"] == selection["selected_profile_id"]
        )
        for factor, selection in selections.items()
    }
    final_rows: list[dict[str, Any]] = []
    for scene_index, record in enumerate(records, start=1):
        clean = clean_images[record["scene"]]
        views = generator_c.make_factorized_views(
            clean,
            f"s2d-f2-v2:{record['scene']}",
            _parameters(selected_profiles["low"]),
            _parameters(selected_profiles["rain"]),
            _parameters(selected_profiles["snow"]),
            low_seed_variant=int(selected_profiles["low"]["id"]),
            rain_seed_variant=int(selected_profiles["rain"]["id"]),
            snow_seed_variant=int(selected_profiles["snow"]["id"]),
        )
        for row in _metric_rows(clean, views, model, torch_module):
            row.update(
                {
                    "scene": record["scene"],
                    "mechanism": "C_FINAL",
                    "clean_pixel_sha256": pixel_sha256(clean),
                }
            )
            final_rows.append(row)
        if scene_index % 20 == 0 or scene_index == len(records):
            print(json.dumps({"phase": "final", "scenes": scene_index,
                              "total": len(records)}), flush=True)

    elapsed = time.perf_counter() - started
    projected = elapsed / len(records) * int(config["expected_scene_count"])
    runtime_pass = projected <= float(
        config["runtime_gate"]["maximum_projected_full_seconds"]
    )
    final_gate = _final_gate(
        reference_rows, final_rows, config["common_support_gate"]
    )
    full_run = len(records) == int(config["expected_scene_count"])
    if not full_run:
        status = "PASS_RUNTIME_SMOKE" if runtime_pass else "FAIL_RUNTIME_SMOKE"
    else:
        status = (
            "PASS_SEVERITY_MATCHING"
            if final_gate["all_conditions_pass"]
            else "FAIL_SEVERITY_MATCHING"
        )
    return {
        "stage": "F2_V2_FACTORIZED_SEVERITY_CALIBRATION",
        "status": status,
        "calibration_version": config["calibration_version"],
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest_path),
        "split": "train",
        "scenes_processed": len(records),
        "full_run": full_run,
        "severity_gate_evaluated": full_run,
        "elapsed_seconds": elapsed,
        "projected_full_seconds": projected,
        "runtime_gate_pass": runtime_pass,
        "environment": _environment(model),
        "selections": selections,
        "selected_profiles": selected_profiles,
        "final_gate": final_gate,
        "counts": {
            "reference_rows": len(reference_rows),
            "factor_candidate_rows": len(factor_rows),
            "final_rows": len(final_rows),
        },
        "safety": {
            "restoration_outputs_used": False,
            "restoration_checkpoints_loaded": False,
            "gpu_used": False,
            "validation_images_decoded": False,
            "final_images_decoded": False,
            "weatherbench_downloaded": False,
            "thresholds_relaxed_after_run": False,
        },
        "reference_rows": reference_rows,
        "factor_candidate_rows": factor_rows,
        "final_rows": final_rows,
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
    print(
        json.dumps(
            {key: value for key, value in result.items() if not key.endswith("_rows")},
            indent=2,
            allow_nan=False,
        )
    )
    if result["status"] == "FAIL_RUNTIME_SMOKE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
