"""Lock CDD-11-30 CV folds and materialize S2b views from attached input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .datasets.s2b_coverage import (
    A_PROBE_TYPES,
    GENERATED_TYPES,
    build_cv_manifest,
    find_cdd11_train_dir,
    load_cv_manifest,
    pixel_sha256,
    sha256_file,
    write_new_manifest,
)
from .low_weather_generator_b import build_realization, make_views
from .audit_low_weather_replication import unique_scene_views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--development-count", type=int, default=25)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _view_map(clean: np.ndarray, scene: str) -> dict[str, np.ndarray]:
    realization = build_realization(clean, f"s2b:{scene}")
    rain = make_views(clean, realization, "rain_haze")
    snow = make_views(clean, realization, "snow_haze")
    return {
        "b_low_weak": rain["low_weak"],
        "b_low_strong": rain["low_strong"],
        "b_rain_haze": rain["weather"],
        "b_snow_haze": snow["weather"],
        "b_low_weak_rain_haze": rain["combined_weak"],
        "b_low_strong_rain_haze": rain["combined_strong"],
        "b_low_weak_snow_haze": snow["combined_weak"],
        "b_low_strong_snow_haze": snow["combined_strong"],
    }


def _a_probe_map(clean: np.ndarray, scene: str) -> dict[str, np.ndarray]:
    views, _ = unique_scene_views("A", clean, f"s2b:{scene}")
    return {
        "a_low_weak": views["low_0.6"],
        "a_low_strong": views["low_1.0"],
        "a_rain_haze": views["weather:rain_haze"],
        "a_snow_haze": views["weather:snow_haze"],
        "a_low_weak_rain_haze": views["combined_0.6:rain_haze"],
        "a_low_strong_rain_haze": views["combined_1.0:rain_haze"],
        "a_low_weak_snow_haze": views["combined_0.6:snow_haze"],
        "a_low_strong_snow_haze": views["combined_1.0:snow_haze"],
    }


def materialize(data_root: Path, manifest_path: Path, cache_root: Path) -> dict:
    manifest = load_cv_manifest(manifest_path)
    train_dir = find_cdd11_train_dir(data_root)
    records = []
    for record in manifest["scenes"]:
        target_path = train_dir / "clear" / record["file"]
        with Image.open(target_path) as image:
            clean = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if hashlib.sha256(np.ascontiguousarray(clean).tobytes()).hexdigest() != record["pixel_sha256"]:
            raise RuntimeError(f"Locked target changed: {record['scene']}")
        views = _view_map(clean, record["scene"])
        views.update(_a_probe_map(clean, record["scene"]))
        expected_types = GENERATED_TYPES + A_PROBE_TYPES
        if tuple(views) != expected_types:
            raise AssertionError("Generated view ordering changed")
        for degradation_type, array in views.items():
            destination = cache_root / "generated" / degradation_type / record["file"]
            if destination.exists():
                with Image.open(destination) as image:
                    existing = np.asarray(image.convert("RGB"), dtype=np.uint8)
                if not np.array_equal(existing, array):
                    raise RuntimeError(f"Refusing to overwrite changed cache: {destination}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(array).save(destination)
            records.append(
                {
                    "scene": record["scene"],
                    "type": degradation_type,
                    "file": str(destination.relative_to(cache_root)).replace("\\", "/"),
                    "pixel_sha256": pixel_sha256(destination),
                    "file_sha256": sha256_file(destination),
                }
            )
    cache_manifest = {
        "protocol_version": manifest["protocol_version"],
        "split_manifest_sha256": sha256_file(manifest_path),
        "generator": "cvpr2027-generator-b-v1",
        "record_count": len(records),
        "records": records,
    }
    destination = cache_root / "cache_manifest.json"
    write_new_manifest(destination, cache_manifest)
    return cache_manifest


def validate_cache(manifest_path: Path, cache_root: Path) -> dict:
    expected_split_hash = sha256_file(manifest_path)
    cache_manifest_path = cache_root / "cache_manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if cache_manifest.get("split_manifest_sha256") != expected_split_hash:
        raise RuntimeError("Generated cache belongs to a different split manifest")
    records = cache_manifest.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Malformed cache manifest")
    for record in records:
        path = cache_root / record["file"]
        if sha256_file(path) != record["file_sha256"]:
            raise RuntimeError(f"Cached file hash changed: {path}")
        if pixel_sha256(path) != record["pixel_sha256"]:
            raise RuntimeError(f"Cached pixel hash changed: {path}")
    return cache_manifest


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    cache_root = Path(args.cache_root).resolve()
    if not manifest_path.exists():
        manifest = build_cv_manifest(
            data_root,
            development_count=args.development_count,
            fold_count=args.fold_count,
            seed=args.seed,
        )
        write_new_manifest(manifest_path, manifest)
    if args.validate_only:
        result = validate_cache(manifest_path, cache_root)
    else:
        result = materialize(data_root, manifest_path, cache_root)
        validate_cache(manifest_path, cache_root)
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "cache_root": str(cache_root),
                "cached_views": result["record_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
