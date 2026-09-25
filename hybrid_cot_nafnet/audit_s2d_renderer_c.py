"""Run the CPU-only F1 determinism and structure smoke for mechanism C."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import cv2
import numpy as np

from . import low_weather_generator_c as gen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene-count", type=int, default=10)
    return parser.parse_args()


def _pixel_sha256(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _fixture(index: int) -> np.ndarray:
    """Create registered content with gradients, texture and hard boundaries."""
    height = 96 + 4 * index
    width = 128 + 3 * index
    y, x = np.mgrid[0:height, 0:width]
    checker = ((x // (8 + index % 3) + y // (9 + index % 2)) % 2).astype(np.float32)
    rgb = np.stack(
        (
            0.08 + 0.72 * x / max(width - 1, 1),
            0.10 + 0.70 * y / max(height - 1, 1),
            0.12 + 0.62 * checker,
        ),
        axis=-1,
    )
    image = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    cv2.circle(
        image,
        (width // 3, height // 2),
        max(8, min(height, width) // 7),
        (235, 80 + index * 5, 35),
        thickness=-1,
        lineType=cv2.LINE_8,
    )
    cv2.rectangle(
        image,
        (width // 2, height // 4),
        (width - 12, 3 * height // 4),
        (30, 215, 190),
        thickness=2,
        lineType=cv2.LINE_8,
    )
    return image


def _edge_correlation(clean: np.ndarray, degraded: np.ndarray) -> float:
    def magnitude(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(dx, dy).ravel()

    left = magnitude(clean)
    right = magnitude(degraded)
    if float(left.std()) == 0 or float(right.std()) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def audit(scene_count: int = 10) -> dict:
    if scene_count <= 0:
        raise ValueError("scene_count must be positive")
    gen._require_upstream()
    records = []
    all_deterministic = True
    all_shape_dtype = True
    all_input_unchanged = True
    all_nonidentity = True
    all_replay_hashes_stable = True
    edge_by_condition: dict[str, list[float]] = {
        condition: [] for condition in gen.CONDITIONS
    }

    for index in range(scene_count):
        scene_id = f"s2d-f1-fixture-{index:02d}"
        clean = _fixture(index)
        clean_before = clean.copy()
        realization = gen.build_realization(clean, scene_id)
        views = gen.make_views(clean, realization)
        repeated = gen.build_realization(clean, scene_id)
        repeated_views = gen.make_views(clean, repeated)
        deterministic = all(
            np.array_equal(views[name], repeated_views[name])
            for name in gen.CONDITIONS
        )
        shape_dtype = all(
            image.shape == clean.shape and image.dtype == np.uint8
            for image in views.values()
        )
        nonidentity = all(
            not np.array_equal(image, clean) for image in views.values()
        )
        replay_stable = (
            realization.replay_sha256 == repeated.replay_sha256
            and realization.seeds == repeated.seeds
        )
        all_deterministic &= deterministic
        all_shape_dtype &= shape_dtype
        all_input_unchanged &= np.array_equal(clean, clean_before)
        all_nonidentity &= nonidentity
        all_replay_hashes_stable &= replay_stable
        correlations = {
            name: _edge_correlation(clean, image) for name, image in views.items()
        }
        for name, value in correlations.items():
            edge_by_condition[name].append(value)
        records.append(
            {
                "scene": scene_id,
                "shape": list(clean.shape),
                "clean_sha256": _pixel_sha256(clean),
                "seeds": realization.seeds,
                "replay_sha256": realization.replay_sha256,
                "view_sha256": {
                    name: _pixel_sha256(image) for name, image in views.items()
                },
                "edge_correlation": correlations,
            }
        )

    rain_transform = gen._weather_transform("rain_haze")
    snow_transform = gen._weather_transform("snow_haze")
    operators_image_only = all(
        isinstance(transform, gen.A.ImageOnlyTransform)
        for transform in [*rain_transform.transforms, *snow_transform.transforms]
    )
    unique_weather_draws = (
        len({record["replay_sha256"]["rain_haze"] for record in records})
        == scene_count
        and len({record["replay_sha256"]["snow_haze"] for record in records})
        == scene_count
    )
    hard_checks = {
        "deterministic_exact_replay": all_deterministic,
        "input_not_mutated": all_input_unchanged,
        "shape_and_dtype_preserved": all_shape_dtype,
        "all_conditions_nonidentity": all_nonidentity,
        "replay_metadata_stable": all_replay_hashes_stable,
        "weather_operators_are_image_only": operators_image_only,
        "scene_ids_produce_unique_weather_draws": unique_weather_draws,
        "condition_order_locked": tuple(edge_by_condition) == gen.CONDITIONS,
    }
    return {
        "stage": "F1_CPU_RENDERER_C_SMOKE",
        "status": "PASS" if all(hard_checks.values()) else "FAIL",
        "generator": gen.GENERATOR_VERSION,
        "albumentations_version": gen.A.__version__,
        "albumentations_distribution_version": importlib.metadata.version("albumentations"),
        "albucore_distribution_version": importlib.metadata.version("albucore"),
        "numpy_version": np.__version__,
        "opencv_python_headless_distribution_version": importlib.metadata.version(
            "opencv-python-headless"
        ),
        "opencv_runtime_version": cv2.__version__,
        "upstream_revision": gen.UPSTREAM_REVISION,
        "scene_count": scene_count,
        "condition_count": len(gen.CONDITIONS),
        "conditions": list(gen.CONDITIONS),
        "canonical_operator_order": ["low", "precipitation", "fog"],
        "hard_checks": hard_checks,
        "edge_correlation_descriptive_only": {
            name: {
                "minimum": min(values),
                "median": float(np.median(values)),
                "maximum": max(values),
            }
            for name, values in edge_by_condition.items()
        },
        "safety": {
            "external_data_downloaded": False,
            "checkpoint_downloaded": False,
            "gpu_used": False,
            "restoration_model_outputs_used": False,
            "severity_calibrated": False,
        },
        "records": records,
    }


def main() -> None:
    args = parse_args()
    result = audit(args.scene_count)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
