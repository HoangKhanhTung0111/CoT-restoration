"""Independently validate the one-shot S2d F2-v2 result."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .calibrate_s2d_severity import FEATURES, sha256_file
from .calibrate_s2d_severity_v2 import _final_gate, _select_factor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def audit(result: dict, config: dict, config_hash: str, manifest_hash: str) -> dict:
    errors: list[str] = []
    reference = result.get("reference_rows", [])
    factors = result.get("factor_candidate_rows", [])
    final = result.get("final_rows", [])
    expected_counts = {
        "reference_rows": 3200,
        "factor_candidate_rows": 6400,
        "final_rows": 1600,
    }
    actual_counts = {
        "reference_rows": len(reference),
        "factor_candidate_rows": len(factors),
        "final_rows": len(final),
    }
    if actual_counts != expected_counts or result.get("counts") != expected_counts:
        errors.append("row_count_mismatch")
    if result.get("config_sha256") != config_hash:
        errors.append("config_hash_mismatch")
    if result.get("manifest_sha256") != manifest_hash:
        errors.append("manifest_hash_mismatch")
    if result.get("scenes_processed") != 200 or result.get("split") != "train":
        errors.append("not_exact_full_train_split")
    if result.get("full_run") is not True or result.get("severity_gate_evaluated") is not True:
        errors.append("severity_gate_not_evaluated")

    safety = result.get("safety", {})
    required_false = {
        "restoration_outputs_used",
        "restoration_checkpoints_loaded",
        "gpu_used",
        "validation_images_decoded",
        "final_images_decoded",
        "weatherbench_downloaded",
        "thresholds_relaxed_after_run",
    }
    if any(safety.get(name) is not False for name in required_false):
        errors.append("safety_invariant_broken")

    key_sets = [
        [(row["scene"], row["condition"], row["mechanism"]) for row in reference],
        [
            (row["scene"], row["condition"], row["factor"], row["profile_id"])
            for row in factors
        ],
        [(row["scene"], row["condition"]) for row in final],
    ]
    if any(len(keys) != len(set(keys)) for keys in key_sets):
        errors.append("duplicate_metric_key")
    if any(
        not math.isfinite(float(row.get(feature, math.nan)))
        for row in [*reference, *factors, *final]
        for feature in FEATURES
    ):
        errors.append("nonfinite_metric")

    recomputed_selections = {}
    for factor, profiles in (
        ("low", config["low_profiles"]),
        ("rain", config["rain_profiles"]),
        ("snow", config["snow_profiles"]),
    ):
        recomputed_selections[factor] = _select_factor(
            factor,
            reference,
            factors,
            profiles,
            config["common_support_gate"],
        )
    if recomputed_selections != result.get("selections"):
        errors.append("selection_recompute_mismatch")
    recomputed_final = _final_gate(reference, final, config["common_support_gate"])
    if recomputed_final != result.get("final_gate"):
        errors.append("final_gate_recompute_mismatch")
    expected_status = (
        "PASS_SEVERITY_MATCHING"
        if recomputed_final["all_conditions_pass"]
        else "FAIL_SEVERITY_MATCHING"
    )
    if result.get("status") != expected_status:
        errors.append("status_mismatch")

    selected = {
        factor: selection["selected_profile_id"]
        for factor, selection in recomputed_selections.items()
    }
    return {
        "stage": "F2_V2_RESULT_AUDIT",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "scientific_result": expected_status,
        "selected_profiles": selected,
        "final_conditions_passed": recomputed_final["conditions_passed"],
        "final_conditions_total": recomputed_final["conditions_total"],
        "final_failed_checks": recomputed_final["failed_checks"],
        "final_condition_summary": recomputed_final["conditions"],
        "counts": actual_counts,
        "safety": safety,
    }


def main() -> None:
    args = parse_args()
    result_path = Path(args.result).resolve()
    config_path = Path(args.config).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    audited = audit(
        result, config, sha256_file(config_path), sha256_file(manifest_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audited, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audited, indent=2, allow_nan=False))
    if audited["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
