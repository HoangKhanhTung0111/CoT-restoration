"""Combine the S2d mechanism probe with the completed S2b interaction evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from .analyze_s2b_generator_specific import analyze as analyze_s2b


BOOTSTRAP_DRAWS = 10000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-archive", required=True)
    parser.add_argument("--s2b-archive", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_probe_archive(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"Probe archive CRC failed at {corrupt}")
        candidates = [
            name for name in archive.namelist() if name.endswith("s2d_probe_result.json")
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one s2d_probe_result.json in probe archive, found {candidates}"
            )
        result = json.loads(archive.read(candidates[0]).decode("utf-8"))
        run_candidates = [name for name in archive.namelist() if name.endswith("run.json")]
        run = json.loads(archive.read(run_candidates[0]).decode("utf-8")) if len(run_candidates) == 1 else None
    if result.get("status") != "COMPLETE":
        raise RuntimeError("S2d probe is incomplete")
    if run is not None and (run.get("status") != "COMPLETE" or run.get("errors")):
        raise RuntimeError("S2d notebook run is incomplete")
    return result


def _bootstrap_ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(20260925)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def evaluate_gate(probe: dict, s2b: dict) -> dict:
    scene_rows = s2b["exploratory_scene_analysis"]["scene_rows"]
    interaction = np.asarray(
        [row["target_generalization_gap_b_minus_a"] for row in scene_rows],
        dtype=np.float64,
    )
    interaction_ci = _bootstrap_ci(interaction)
    fold_rows = s2b["exploratory_scene_analysis"]["fold_mean_delta_psnr"]
    positive_fold_gaps = sum(
        row["generator_b_target_delta_psnr"]
        - row["generator_a_target_delta_psnr"]
        > 0
        for row in fold_rows
    )
    gt_best = max(
        value["balanced_accuracy"]
        for value in probe["clean_gt_negative_controls"].values()
    )
    permutation = probe["label_permutation_control_for_best_raw_family"]
    criteria = {
        "interaction_bootstrap_ci_lower_above_zero": interaction_ci[0] > 0,
        "raw_input_probe_balanced_accuracy_at_least_0p80": (
            probe["best_raw_balanced_accuracy"] >= 0.80
        ),
        "clean_gt_control_at_most_0p60": gt_best <= 0.60,
        "at_least_4_of_5_fold_interaction_gaps_positive": positive_fold_gaps >= 4,
        "moment_matched_probe_balanced_accuracy_at_least_0p70": (
            probe["best_matched_balanced_accuracy"] >= 0.70
        ),
        "raw_probe_above_permutation_q95": (
            probe["best_raw_balanced_accuracy"] > permutation["q95"]
        ),
    }
    decision = "GO_FULL_AUDIT" if all(criteria.values()) else "NO_GO_OR_REVISE_S2D"
    return {
        "protocol_version": "s2d-mechanism-pilot-decision-v1",
        "status": "COMPLETE",
        "decision": decision,
        "criteria": criteria,
        "interaction": {
            "definition": "(R1-R0 on Generator B targets) - (R1-R0 on Generator A targets)",
            "scene_count": len(interaction),
            "mean_delta_psnr": float(interaction.mean()),
            "scene_bootstrap_ci95": interaction_ci,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "positive_fold_gaps": positive_fold_gaps,
        },
        "probe": {
            "best_raw_family": probe["best_raw_family"],
            "best_raw_balanced_accuracy": probe["best_raw_balanced_accuracy"],
            "best_moment_matched_balanced_accuracy": probe[
                "best_matched_balanced_accuracy"
            ],
            "best_clean_gt_control_balanced_accuracy": gt_best,
            "permutation_q95": permutation["q95"],
        },
        "limits": [
            "Passing this pilot authorizes a broader frozen-model audit, not a CVPR claim.",
            "The probe shows generator cues are present; it does not prove a restorer causally uses them.",
            "No restoration training or sealed-partition access is authorized by this decision.",
        ],
    }


def summarize(probe_archive: Path, s2b_archive: Path) -> dict:
    probe = _read_probe_archive(probe_archive)
    s2b = analyze_s2b(s2b_archive)
    result = evaluate_gate(probe, s2b)
    result["sources"] = {
        "probe_archive": probe_archive.name,
        "probe_archive_sha256": sha256_file(probe_archive),
        "s2b_archive": s2b_archive.name,
        "s2b_archive_sha256": sha256_file(s2b_archive),
    }
    return result


def main() -> None:
    args = parse_args()
    result = summarize(
        Path(args.probe_archive).resolve(), Path(args.s2b_archive).resolve()
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
