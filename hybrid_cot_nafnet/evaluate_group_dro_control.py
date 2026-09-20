"""Evaluate one Group DRO control against the locked three-control result ZIP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch

from .datasets.order_controls import ORDER_A, ORDER_B, OrderControlDataset
from .evaluate import load_torch_file
from .evaluate_order_controls import (
    _cluster_ci,
    _mean,
    _write_csv,
    _write_json,
    evaluate_checkpoint,
)
from .model import build_model
from .project_config import KAGGLE_CDD11_ROOT


ORDER_NAMES = (">".join(ORDER_A), ">".join(ORDER_B))
REFERENCE_MODELS = ("fixed_a", "fixed_b", "balanced")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--reference-zip",
        required=True,
        help="order_controls_low_haze_results.zip from the locked three-control run.",
    )
    parser.add_argument("--data-root", default=str(KAGGLE_CDD11_ROOT))
    parser.add_argument(
        "--output-dir", default="/kaggle/working/order_group_dro_evaluation"
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=20260920)
    parser.add_argument("--realizations", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--practical-improvement-db", type=float, default=0.2)
    parser.add_argument("--equivalence-margin-db", type=float, default=0.1)
    parser.add_argument("--mean-noninferiority-margin-db", type=float, default=0.1)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_reference(path: str | Path, allow_smoke: bool = False):
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Reference ZIP CRC failure: {bad}")
        required = {
            "evaluation/per_sample.csv",
            "evaluation/protocol.json",
            "evaluation/decision.json",
            "evaluation/contrasts.json",
        }
        missing = sorted(required.difference(archive.namelist()))
        if missing:
            raise RuntimeError(f"Reference ZIP is missing files: {missing}")
        rows = list(
            csv.DictReader(
                io.StringIO(archive.read("evaluation/per_sample.csv").decode())
            )
        )
        protocol = json.loads(archive.read("evaluation/protocol.json"))
        decision = json.loads(archive.read("evaluation/decision.json"))
        contrasts = json.loads(archive.read("evaluation/contrasts.json"))
    if set(row["model"] for row in rows) != set(REFERENCE_MODELS):
        raise RuntimeError("Reference ZIP does not contain exactly the three controls")
    if protocol.get("test_split_loaded") is not False:
        raise RuntimeError("Reference protocol did not keep the test split locked")
    if not allow_smoke and not decision.get("training_order_exposure_supported"):
        raise RuntimeError("Reference result does not support the preregistered next step")
    if not allow_smoke and not decision.get("joint_learning_limitation_supported"):
        raise RuntimeError("Reference result does not support the Group DRO control")
    return rows, protocol, decision, contrasts


def _load_group_dro_model(path: str | Path, device: torch.device, allow_smoke: bool):
    checkpoint = load_torch_file(path)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"Invalid Group DRO checkpoint: {path}")
    training_args = checkpoint.get("args", {})
    observed = {
        "model_type": checkpoint.get("model_type"),
        "preset": checkpoint.get("preset"),
        "training_data": training_args.get("training_data"),
        "order_policy": training_args.get("order_policy"),
        "group_dro": training_args.get("group_dro"),
        "group_dro_eta": training_args.get("group_dro_eta"),
        "epochs": training_args.get("epochs"),
        "seed": training_args.get("seed"),
        "generation_seed": training_args.get("generation_seed"),
        "selection_metric": checkpoint.get("selection_metric"),
    }
    expected = {
        "model_type": "baseline",
        "preset": "sidd32",
        "training_data": "synthetic_order",
        "order_policy": "balanced",
        "group_dro": True,
        "group_dro_eta": 0.1,
        "epochs": 20,
        "seed": 42,
        "generation_seed": 20260920,
        "selection_metric": "validation_worst_order_psnr",
    }
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches and not allow_smoke:
        raise RuntimeError(f"Group DRO checkpoint protocol mismatch: {mismatches}")
    model = build_model(
        "baseline",
        str(checkpoint.get("preset", "sidd32")),
        int(checkpoint.get("adapter_hidden", 64)),
        use_skip_gates=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), observed, mismatches


def _key(row: Mapping[str, object]) -> Tuple[str, int, int, str]:
    return (
        str(row["scene_id"]),
        int(row["realization"]),
        int(row["generation_seed"]),
        str(row["order"]),
    )


def _validate_reference_alignment(
    reference_rows: Sequence[Mapping[str, object]],
    group_rows: Sequence[Mapping[str, object]],
) -> None:
    balanced = {
        _key(row): row for row in reference_rows if row["model"] == "balanced"
    }
    group = {_key(row): row for row in group_rows}
    if set(balanced) != set(group):
        raise RuntimeError("Group DRO samples do not match the locked reference keys")
    for key in balanced:
        for metric in ("input_psnr", "input_ssim"):
            if abs(float(balanced[key][metric]) - float(group[key][metric])) > 1e-6:
                raise RuntimeError(
                    f"Reference input mismatch for {key}, metric={metric}"
                )


def _scene_order_values(
    rows: Sequence[Mapping[str, object]], metric: str = "output_psnr"
) -> Dict[Tuple[str, str, str], float]:
    grouped: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["order"]), str(row["scene_id"]))].append(
            float(row[metric])
        )
    return {key: _mean(values) for key, values in grouped.items()}


def _contrast(
    description: str,
    per_scene: Mapping[str, float],
    samples: int,
    seed: int,
) -> Dict[str, object]:
    low, high = _cluster_ci(per_scene, samples, seed)
    return {
        "description": description,
        "mean": _mean(per_scene.values()),
        "scene_cluster_bootstrap_ci95": [low, high],
        "per_scene": dict(sorted(per_scene.items())),
    }


def _per_order_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["order"]))].append(row)
    result = []
    for (model, order), items in sorted(grouped.items()):
        result.append(
            {
                "model": model,
                "order": order,
                "mean_input_psnr": _mean(float(row["input_psnr"]) for row in items),
                "mean_output_psnr": _mean(float(row["output_psnr"]) for row in items),
                "mean_gain_psnr": _mean(float(row["gain_psnr"]) for row in items),
                "mean_output_ssim": _mean(float(row["output_ssim"]) for row in items),
                "harm_rate": _mean(float(row["harm"]) for row in items),
            }
        )
    return result


def compare(
    rows: Sequence[Mapping[str, object]], args: argparse.Namespace
) -> Tuple[Dict[str, object], Dict[str, object]]:
    values = _scene_order_values(rows)
    order_a, order_b = ORDER_NAMES
    scenes = sorted({key[2] for key in values})
    gdro_minus_balanced_a = {
        scene: values[("group_dro", order_a, scene)]
        - values[("balanced", order_a, scene)]
        for scene in scenes
    }
    gdro_minus_balanced_b = {
        scene: values[("group_dro", order_b, scene)]
        - values[("balanced", order_b, scene)]
        for scene in scenes
    }
    worst_improvement = {
        scene: min(
            values[("group_dro", order_a, scene)],
            values[("group_dro", order_b, scene)],
        )
        - min(
            values[("balanced", order_a, scene)],
            values[("balanced", order_b, scene)],
        )
        for scene in scenes
    }
    mean_improvement = {
        scene: (
            values[("group_dro", order_a, scene)]
            + values[("group_dro", order_b, scene)]
            - values[("balanced", order_a, scene)]
            - values[("balanced", order_b, scene)]
        )
        / 2.0
        for scene in scenes
    }
    specialist_gap = {
        scene: (
            values[("fixed_a", order_a, scene)]
            + values[("fixed_b", order_b, scene)]
            - values[("group_dro", order_a, scene)]
            - values[("group_dro", order_b, scene)]
        )
        / 2.0
        for scene in scenes
    }
    contrasts = {
        "group_dro_minus_balanced_order_a": _contrast(
            "Positive favors Group DRO on low>haze",
            gdro_minus_balanced_a,
            args.bootstrap_samples,
            args.generation_seed + 11,
        ),
        "group_dro_minus_balanced_order_b": _contrast(
            "Positive favors Group DRO on haze>low",
            gdro_minus_balanced_b,
            args.bootstrap_samples,
            args.generation_seed + 22,
        ),
        "group_dro_minus_balanced_worst_order": _contrast(
            "Primary endpoint: positive improves scene-level worst-order PSNR",
            worst_improvement,
            args.bootstrap_samples,
            args.generation_seed + 33,
        ),
        "group_dro_minus_balanced_mean": _contrast(
            "Safety endpoint: positive improves mean A/B PSNR",
            mean_improvement,
            args.bootstrap_samples,
            args.generation_seed + 44,
        ),
        "specialists_minus_group_dro_on_matched_orders": _contrast(
            "Positive means Group DRO remains below corresponding specialists",
            specialist_gap,
            args.bootstrap_samples,
            args.generation_seed + 55,
        ),
    }
    worst = contrasts["group_dro_minus_balanced_worst_order"]
    mean = contrasts["group_dro_minus_balanced_mean"]
    gap = contrasts["specialists_minus_group_dro_on_matched_orders"]
    worst_mean = float(worst["mean"])
    worst_low = float(worst["scene_cluster_bootstrap_ci95"][0])
    mean_low = float(mean["scene_cluster_bootstrap_ci95"][0])
    gap_high = float(gap["scene_cluster_bootstrap_ci95"][1])
    robust_improvement = bool(
        worst_mean >= args.practical_improvement_db and worst_low > 0.0
    )
    mean_noninferior = bool(mean_low >= -args.mean_noninferiority_margin_db)
    specialist_equivalent = bool(gap_high <= args.equivalence_margin_db)
    if robust_improvement and mean_noninferior and specialist_equivalent:
        outcome = "full_success_standard_robust_objective_is_sufficient"
        next_step = "stop_new_architecture_and_expand_independent_validation"
    elif robust_improvement and mean_noninferior:
        outcome = "partial_success_worst_order_improves_but_specialist_gap_remains"
        next_step = "stop_gpu_and_define_any_new_mechanism_against_group_dro"
    else:
        outcome = "failure_group_dro_does_not_resolve_the_joint_learning_limit"
        next_step = "stop_gpu_and_write_the_mechanism_hypothesis_before_new_runs"
    decision = {
        "status": "exploratory_pilot" if len(scenes) < 30 else "controlled_development",
        "independent_validation_scenes": len(scenes),
        "outcome": outcome,
        "robust_worst_order_improvement_supported": robust_improvement,
        "mean_psnr_noninferior": mean_noninferior,
        "equivalent_to_order_specialists": specialist_equivalent,
        "recommended_next_step": next_step,
        "project_management_gates_not_universal_thresholds": {
            "practical_improvement_db": args.practical_improvement_db,
            "equivalence_margin_db": args.equivalence_margin_db,
            "mean_noninferiority_margin_db": args.mean_noninferiority_margin_db,
        },
        "scope_warning": (
            f"One synthetic low+haze pair and {len(scenes)} validation scenes cannot "
            "support a general AiOIR or publication-level claim."
        ),
    }
    return contrasts, decision


def main() -> None:
    args = parse_args()
    reference_rows, reference_protocol, reference_decision, reference_contrasts = (
        _read_reference(args.reference_zip, args.allow_smoke_checkpoint)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable; use Kaggle GPU or pass --allow-cpu")
    dataset = OrderControlDataset(
        args.data_root,
        mode="val",
        order_policy="both",
        realizations=args.realizations,
        crop_size=0,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        generation_seed=args.generation_seed,
    )
    if list(dataset.scene_ids) != reference_protocol.get("scene_ids"):
        raise RuntimeError("Validation scenes differ from the locked reference")
    model, observed, mismatches = _load_group_dro_model(
        args.checkpoint, device, args.allow_smoke_checkpoint
    )
    group_rows = evaluate_checkpoint(
        "group_dro",
        model,
        dataset,
        device,
        args.tile,
        args.overlap,
        bool(args.amp and device.type == "cuda"),
    )
    _validate_reference_alignment(reference_rows, group_rows)
    all_rows: List[Mapping[str, object]] = [*reference_rows, *group_rows]
    contrasts, decision = compare(all_rows, args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_sample_all.csv", all_rows)
    _write_csv(output_dir / "per_order_all.csv", _per_order_rows(all_rows))
    _write_json(output_dir / "contrasts.json", contrasts)
    _write_json(output_dir / "decision.json", decision)
    _write_json(
        output_dir / "protocol.json",
        {
            "reference_zip": str(Path(args.reference_zip).resolve()),
            "reference_zip_sha256": _sha256(args.reference_zip),
            "reference_decision": reference_decision,
            "reference_contrasts": reference_contrasts,
            "group_dro_checkpoint": str(Path(args.checkpoint).resolve()),
            "group_dro_checkpoint_sha256": _sha256(args.checkpoint),
            "group_dro_checkpoint_observed": observed,
            "group_dro_checkpoint_protocol_mismatches": mismatches,
            "scene_ids": list(dataset.scene_ids),
            "test_split_loaded": False,
            "reference_sample_alignment_passed": True,
            "args": vars(args),
        },
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
