"""Independently recompute and validate an S2e severity-conditioned result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .analyze_s2e_severity_conditioned import summarize_adjusted_rows
from .calibrate_s2d_severity import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _all_finite(value) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    return False


def audit(result: dict, config: dict, config_hash: str, workspace: Path) -> dict:
    errors: list[str] = []
    if result.get("protocol_version") != config.get("protocol_version"):
        errors.append("protocol_version_mismatch")
    if result.get("config_sha256") != config_hash:
        errors.append("config_hash_mismatch")
    for name in ("s2b", "s2d_raw"):
        path = (workspace / config["sources"][f"{name}_archive"]).resolve()
        expected = config["sources"][f"{name}_archive_sha256"]
        if sha256_file(path) != expected:
            errors.append(f"{name}_archive_hash_mismatch")

    rows = result.get("adjusted_rows", [])
    expected = config["expected"]
    if len(rows) != int(expected["primary_rows"]):
        errors.append("primary_row_count_mismatch")
    keys = [
        (row.get("scene"), row.get("condition"), row.get("generator"))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        errors.append("duplicate_adjusted_key")
    if {row.get("generator") for row in rows} != {"A", "B"}:
        errors.append("generator_set_mismatch")
    if {row.get("condition") for row in rows} != set(config["primary_conditions"]):
        errors.append("condition_set_mismatch")
    if {int(row.get("fold", -1)) for row in rows} != set(range(int(expected["folds"]))):
        errors.append("fold_set_mismatch")
    if len({row.get("scene") for row in rows}) != int(expected["scenes"]):
        errors.append("scene_count_mismatch")
    if not _all_finite(rows):
        errors.append("nonfinite_adjusted_value")

    recomputed = summarize_adjusted_rows(rows, config) if not errors else None
    if recomputed is not None:
        for key in ("overlap", "effects", "gates", "decision"):
            if recomputed[key] != result.get(key):
                errors.append(f"recomputed_{key}_mismatch")

    safety = result.get("safety", {})
    if safety.get("cpu_only") is not True:
        errors.append("cpu_only_not_true")
    for name in (
        "restoration_inference_used",
        "training_used",
        "checkpoint_loaded",
        "network_download_used",
        "sealed_partition_accessed",
        "renderer_retuned",
        "gate_retuned_after_result",
    ):
        if safety.get(name) is not False:
            errors.append(f"safety_{name}_not_false")

    raw_source = result.get("sources", {}).get("s2d_raw", {})
    if "input_lpips" not in config["severity"]["features"]:
        if raw_source.get("used_for_severity") is not False:
            errors.append("mismatched_raw_was_used_for_severity")
        alignment = raw_source.get("cache_alignment", {}).get("by_generator", {})
        if alignment.get("A", {}).get("exact_pixel_and_file_hash_matches") != 200:
            errors.append("unexpected_a_cache_alignment")
        if alignment.get("B", {}).get("exact_pixel_and_file_hash_matches") != 0:
            errors.append("unexpected_b_cache_alignment")

    return {
        "stage": "S2E_RESULT_AUDIT",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "scientific_decision": (
            recomputed["decision"] if recomputed is not None else result.get("decision")
        ),
        "counts": result.get("counts"),
        "gates": recomputed["gates"] if recomputed is not None else None,
        "primary_effects": (
            {
                "residualized": recomputed["effects"][
                    "cross_fitted_residualized_interaction"
                ],
                "overlap_weighted": recomputed["effects"][
                    "overlap_weighted_interaction"
                ],
            }
            if recomputed is not None
            else None
        ),
        "safety": safety,
    }


def main() -> None:
    args = parse_args()
    result_path = Path(args.result).resolve()
    config_path = Path(args.config).resolve()
    output_path = Path(args.output).resolve()
    workspace = config_path.parent.parent
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    audited = audit(result, config, sha256_file(config_path), workspace)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audited, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audited, indent=2, allow_nan=False))
    if audited["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

