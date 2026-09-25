"""Independently validate a completed S2d F2 calibration artifact."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .calibrate_s2d_severity import CONDITIONS, FEATURES, analyze_candidates, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def audit_result(result: dict, config: dict, config_hash: str, manifest_hash: str) -> dict:
    errors: list[str] = []
    rows = result.get("rows", [])
    if result.get("config_sha256") != config_hash:
        errors.append("config_hash_mismatch")
    if result.get("manifest_sha256") != manifest_hash:
        errors.append("manifest_hash_mismatch")
    if result.get("split") != "train" or result.get("scenes_processed") != 200:
        errors.append("not_exactly_full_train_split")
    if result.get("full_run") is not True or result.get("severity_gate_evaluated") is not True:
        errors.append("full_gate_not_evaluated")
    if result.get("metric_rows") != 16000 or len(rows) != 16000:
        errors.append("metric_row_count_mismatch")

    forbidden_safety = [
        "restoration_outputs_used",
        "restoration_checkpoints_loaded",
        "gpu_used",
        "validation_images_decoded",
        "final_images_decoded",
        "weatherbench_downloaded",
    ]
    safety = result.get("safety", {})
    if any(safety.get(name) is not False for name in forbidden_safety):
        errors.append("safety_invariant_broken")

    keys = []
    scene_clean_hashes: dict[str, set[str]] = defaultdict(set)
    finite = True
    for row in rows:
        keys.append(
            (
                row.get("scene"),
                row.get("condition"),
                row.get("mechanism"),
                row.get("candidate_id"),
            )
        )
        scene_clean_hashes[str(row.get("scene"))].add(str(row.get("clean_pixel_sha256")))
        finite &= all(math.isfinite(float(row.get(name, math.nan))) for name in FEATURES)
    if len(keys) != len(set(keys)):
        errors.append("duplicate_metric_key")
    if not finite:
        errors.append("nonfinite_metric")
    if len(scene_clean_hashes) != 200 or any(len(values) != 1 for values in scene_clean_hashes.values()):
        errors.append("scene_or_clean_hash_mismatch")

    mechanism_counts = Counter(row.get("mechanism") for row in rows)
    condition_counts = Counter(row.get("condition") for row in rows)
    candidate_counts = Counter(
        row.get("candidate_id") for row in rows if row.get("mechanism") == "C"
    )
    if mechanism_counts != {"A": 1600, "B": 1600, "C": 12800}:
        errors.append("mechanism_count_mismatch")
    if condition_counts != {condition: 2000 for condition in CONDITIONS}:
        errors.append("condition_count_mismatch")
    if candidate_counts != {candidate: 1600 for candidate in range(8)}:
        errors.append("candidate_count_mismatch")

    recomputed = analyze_candidates(
        rows, config["candidate_profiles"], config["acceptance_tolerances"]
    )
    if recomputed != result.get("analysis"):
        errors.append("analysis_recompute_mismatch")
    selected_id = recomputed["selected_candidate_id"]
    selected = next(
        item for item in recomputed["candidates"] if item["candidate_id"] == selected_id
    )
    expected_status = (
        "PASS_SEVERITY_MATCHING"
        if selected["all_conditions_pass"]
        else "FAIL_SEVERITY_MATCHING"
    )
    if result.get("status") != expected_status:
        errors.append("status_does_not_match_recomputed_gate")

    return {
        "stage": "F2_RESULT_AUDIT",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "scientific_result": expected_status,
        "selected_candidate_id": selected_id,
        "selected_failed_checks": selected["failed_checks"],
        "selected_conditions_passed": sum(
            condition["pass"] for condition in selected["conditions"]
        ),
        "selected_conditions_total": len(selected["conditions"]),
        "candidate_summary": [
            {
                "candidate_id": item["candidate_id"],
                "failed_checks": item["failed_checks"],
                "mean_normalized_acceptance_ratio": item[
                    "mean_normalized_acceptance_ratio"
                ],
                "all_conditions_pass": item["all_conditions_pass"],
            }
            for item in recomputed["candidates"]
        ],
        "selected_condition_summary": selected["conditions"],
        "counts": {
            "scenes": len(scene_clean_hashes),
            "rows": len(rows),
            "mechanisms": dict(mechanism_counts),
            "conditions": dict(condition_counts),
            "candidate_rows": {str(key): value for key, value in candidate_counts.items()},
        },
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
    audit = audit_result(
        result, config, sha256_file(config_path), sha256_file(manifest_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, allow_nan=False))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
