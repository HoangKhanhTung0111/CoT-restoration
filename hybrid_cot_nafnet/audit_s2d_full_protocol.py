"""Validate the locked S2d full frozen-model audit before any data/GPU work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROTOCOL_VERSION = "s2d-full-frozen-audit-v1.0"
EXPECTED_CONDITIONS = {
    "low_weak",
    "low_strong",
    "rain_haze",
    "snow_haze",
    "low_weak_rain_haze",
    "low_strong_rain_haze",
    "low_weak_snow_haze",
    "low_strong_snow_haze",
}
READY_STATUSES = {"PASS", "PASS_DESIGN_ONLY", "PASS_METADATA_ONLY", "PASS_WITH_MONITORING"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(config: dict) -> dict:
    errors: list[str] = []
    if config.get("protocol_version") != PROTOCOL_VERSION:
        errors.append("unexpected_protocol_version")

    scope = config.get("scientific_scope", {})
    forbidden_flags = {
        "restoration_training_allowed": scope.get("restoration_training_allowed"),
        "test_time_adaptation_allowed": scope.get("test_time_adaptation_allowed"),
        "sealed_partition_access_allowed": scope.get("sealed_partition_access_allowed"),
        "large_dataset_download_allowed": scope.get("large_dataset_download_allowed"),
    }
    errors.extend(name for name, value in forbidden_flags.items() if value is not False)
    if scope.get("candidate_method_versions_consumed") != 0:
        errors.append("candidate_method_budget_changed")

    corpus = config.get("controlled_corpus", {})
    splits = corpus.get("splits", {})
    scene_count = corpus.get("scene_count")
    if scene_count != 500 or sum(splits.values()) != scene_count:
        errors.append("controlled_scene_split_mismatch")
    if set(splits) != {"severity_calibration", "development", "final_once"}:
        errors.append("controlled_split_schema_changed")
    if corpus.get("preserve_official_split") is not True:
        errors.append("official_split_not_preserved")

    conditions = config.get("conditions", [])
    if len(conditions) != len(set(conditions)) or set(conditions) != EXPECTED_CONDITIONS:
        errors.append("condition_grid_changed")
    if config.get("canonical_order") != ["low", "precipitation", "haze"]:
        errors.append("canonical_order_changed")

    mechanisms = config.get("mechanisms", [])
    mechanism_ids = [item.get("id") for item in mechanisms]
    mechanism_families = [item.get("family") for item in mechanisms]
    if mechanism_ids != ["A", "B", "C"]:
        errors.append("mechanism_roster_changed")
    if len(set(mechanism_families)) < 3:
        errors.append("mechanism_families_not_independent")
    held_out = [item for item in mechanisms if item.get("role") == "mechanism_holdout"]
    if len(held_out) != 1 or held_out[0].get("id") != "C":
        errors.append("heldout_mechanism_changed")
    if held_out and held_out[0].get("license") != "MIT":
        errors.append("heldout_mechanism_license_not_locked")

    systems = config.get("frozen_systems", [])
    if len(systems) < 5:
        errors.append("fewer_than_five_frozen_systems")
    if len({item.get("id") for item in systems}) != len(systems):
        errors.append("duplicate_frozen_system")
    if len({item.get("family") for item in systems}) < 3:
        errors.append("fewer_than_three_system_families")
    if any(not item.get("license") for item in systems):
        errors.append("missing_system_license")
    if any(item.get("checkpoint_available") is not True for item in systems):
        errors.append("missing_system_checkpoint")

    matrix = config.get("evaluation_matrix", {})
    expected_development = splits.get("development", -1) * len(conditions) * len(mechanisms) * len(systems)
    expected_final = splits.get("final_once", -1) * len(conditions) * len(mechanisms) * len(systems)
    endpoint_pairs = config.get("real_endpoint", {}).get("reported_test_pairs", -1)
    expected_real = endpoint_pairs * len(systems)
    if matrix.get("development_forwards") != expected_development:
        errors.append("development_forward_count_mismatch")
    if matrix.get("final_forwards") != expected_final:
        errors.append("final_forward_count_mismatch")
    if matrix.get("real_endpoint_forwards") != expected_real:
        errors.append("real_forward_count_mismatch")

    calibration = config.get("severity_calibration", {})
    if calibration.get("uses_restoration_outputs") is not False:
        errors.append("severity_calibration_uses_model_outputs")
    if calibration.get("calibration_split_only") is not True:
        errors.append("severity_calibration_split_leakage")
    required_tolerances = {
        "absolute_median_input_psnr_db",
        "absolute_median_input_ssim",
        "absolute_median_input_lpips",
        "standardized_energy_distance",
    }
    if set(calibration.get("acceptance_tolerances", {})) != required_tolerances:
        errors.append("severity_tolerance_schema_changed")

    endpoint = config.get("real_endpoint", {})
    if endpoint.get("download_allowed") is not False:
        errors.append("weatherbench_download_enabled_before_permission")

    feasibility = config.get("current_feasibility", {})
    blocking_gates = sorted(
        name for name, status in feasibility.items() if status not in READY_STATUSES
    )
    pre_f2_gates = {
        "controlled_corpus_license",
        "controlled_manifest_materialized",
        "three_independent_mechanisms",
        "novelty_search",
    }
    pre_f2_ready = all(
        feasibility.get(name) in READY_STATUSES for name in pre_f2_gates
    )
    calibration = config.get("severity_calibration", {})
    if feasibility.get("severity_matching") == "FAIL":
        expected_decision = "STOP_F2_REVISE_OR_REJECT_RENDERER_C"
    elif pre_f2_ready and calibration.get("runtime_smoke_status") == "PASS_RUNTIME_SMOKE":
        expected_decision = "READY_FOR_F2_FULL_CPU_CALIBRATION"
    elif pre_f2_ready:
        expected_decision = "READY_FOR_F2_CPU_CALIBRATION"
    else:
        expected_decision = "HOLD_BEFORE_DATA_DOWNLOAD_OR_GPU"
    if config.get("current_decision") != expected_decision:
        errors.append("decision_does_not_match_gate_state")

    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "VALID" if not errors else "INVALID",
        "decision": expected_decision if not errors else "INVALID_PROTOCOL",
        "errors": errors,
        "blocking_gates": blocking_gates,
        "counts": {
            "controlled_scenes": scene_count,
            "conditions": len(conditions),
            "mechanisms": len(mechanisms),
            "mechanism_families": len(set(mechanism_families)),
            "frozen_systems": len(systems),
            "system_families": len({item.get("family") for item in systems}),
            "development_forwards": expected_development,
            "final_forwards": expected_final,
            "real_endpoint_forwards": expected_real,
        },
        "safety": {
            **forbidden_flags,
            "weatherbench_download_allowed": endpoint.get("download_allowed"),
        },
    }


def main() -> None:
    args = parse_args()
    result = audit(load_config(Path(args.config)))
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if result["status"] != "VALID":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
