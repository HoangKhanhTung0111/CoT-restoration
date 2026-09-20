"""Compare true privileged order conditioning with an architecture-matched fixed code."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch

from .datasets.order_controls import ORDER_A, ORDER_B, OrderControlDataset
from .evaluate import calculate_psnr, calculate_ssim, load_torch_file
from .evaluate_order_controls import _cluster_ci, _mean, _write_csv, _write_json
from .model import build_model
from .project_config import KAGGLE_CDD11_ROOT
from .train_kaggle import amp_context


ORDER_NAMES = (">".join(ORDER_A), ">".join(ORDER_B))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--true-checkpoint", required=True)
    parser.add_argument("--fixed-checkpoint", required=True)
    parser.add_argument(
        "--reference-json",
        default="configs/order_controls_reference_20260920.json",
    )
    parser.add_argument("--data-root", default=str(KAGGLE_CDD11_ROOT))
    parser.add_argument(
        "--output-dir", default="/kaggle/working/order_condition_oracle_evaluation"
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=20260920)
    parser.add_argument("--realizations", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--practical-improvement-db", type=float, default=0.2)
    parser.add_argument("--equivalence-margin-db", type=float, default=0.1)
    parser.add_argument("--worst-noninferiority-margin-db", type=float, default=0.1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--allow-smoke-checkpoints", action="store_true")
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_reference(path: str | Path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("test_split_loaded") is not False:
        raise RuntimeError("Locked reference did not preserve the test split")
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Locked reference has no per-sample rows")
    if set(row["model"] for row in rows) != {"fixed_a", "fixed_b", "balanced"}:
        raise RuntimeError("Locked reference must contain exactly the three controls")
    return value, rows


def _load_model(
    path: str | Path,
    expected_mode: str,
    device: torch.device,
    allow_smoke: bool,
):
    checkpoint = load_torch_file(path)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"Invalid checkpoint: {path}")
    training_args = checkpoint.get("args", {})
    observed = {
        "model_type": checkpoint.get("model_type"),
        "preset": checkpoint.get("preset"),
        "order_hidden": checkpoint.get("order_hidden"),
        "training_data": training_args.get("training_data"),
        "order_policy": training_args.get("order_policy"),
        "order_condition_mode": training_args.get("order_condition_mode"),
        "group_dro": training_args.get("group_dro"),
        "epochs": training_args.get("epochs"),
        "seed": training_args.get("seed"),
        "generation_seed": training_args.get("generation_seed"),
        "selection_metric": checkpoint.get("selection_metric"),
    }
    expected = {
        "model_type": "order_conditioned",
        "preset": "sidd32",
        "order_hidden": 32,
        "training_data": "synthetic_order",
        "order_policy": "balanced",
        "order_condition_mode": expected_mode,
        "group_dro": False,
        "epochs": 20,
        "seed": 42,
        "generation_seed": 20260920,
        "selection_metric": "validation_psnr",
    }
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches and not allow_smoke:
        raise RuntimeError(f"Checkpoint protocol mismatch for {expected_mode}: {mismatches}")
    model = build_model(
        "order_conditioned",
        str(checkpoint.get("preset", "sidd32")),
        order_hidden=int(checkpoint.get("order_hidden", 32)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), observed, mismatches


@torch.inference_mode()
def _evaluate(
    model_key: str,
    mode: str,
    model,
    dataset: OrderControlDataset,
    device: torch.device,
    use_amp: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        lq = sample["lq"].unsqueeze(0).float()
        gt = sample["gt"].unsqueeze(0).float()
        true_id = 0 if sample["formation_order"] == "low>haze" else 1
        order_id = true_id if mode == "true" else 0
        with amp_context(device, use_amp):
            prediction = model(
                lq.to(device),
                torch.tensor([order_id], dtype=torch.long, device=device),
            ).float().cpu()
        if not torch.isfinite(prediction).all() and use_amp:
            prediction = model(
                lq.to(device),
                torch.tensor([order_id], dtype=torch.long, device=device),
            ).float().cpu()
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f"{model_key} produced non-finite output")
        prediction = prediction.clamp(0.0, 1.0)
        input_psnr = calculate_psnr(lq, gt)
        output_psnr = calculate_psnr(prediction, gt)
        input_ssim = calculate_ssim(lq, gt)
        output_ssim = calculate_ssim(prediction, gt)
        rows.append(
            {
                "model": model_key,
                "scene_id": str(sample["scene_id"]),
                "realization": int(sample["realization"]),
                "generation_seed": int(sample["generation_seed"]),
                "order": str(sample["formation_order"]),
                "input_psnr": input_psnr,
                "output_psnr": output_psnr,
                "gain_psnr": output_psnr - input_psnr,
                "input_ssim": input_ssim,
                "output_ssim": output_ssim,
                "gain_ssim": output_ssim - input_ssim,
                "harm": int(output_psnr < input_psnr),
            }
        )
    return rows


def _key(row: Mapping[str, object]) -> Tuple[str, int, int, str]:
    return (
        str(row["scene_id"]),
        int(row["realization"]),
        int(row["generation_seed"]),
        str(row["order"]),
    )


def _validate_alignment(
    reference_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> None:
    balanced = {
        _key(row): row for row in reference_rows if row["model"] == "balanced"
    }
    candidate = {_key(row): row for row in candidate_rows}
    if set(balanced) != set(candidate):
        raise RuntimeError("Candidate samples do not match the locked reference keys")
    for key in balanced:
        for metric in ("input_psnr", "input_ssim"):
            if abs(float(balanced[key][metric]) - float(candidate[key][metric])) > 1e-6:
                raise RuntimeError(f"Input mismatch for {key}, metric={metric}")


def _scene_order_values(rows: Sequence[Mapping[str, object]]):
    grouped: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["order"]), str(row["scene_id"]))].append(
            float(row["output_psnr"])
        )
    return {key: _mean(values) for key, values in grouped.items()}


def _contrast(description: str, values: Mapping[str, float], samples: int, seed: int):
    low, high = _cluster_ci(values, samples, seed)
    return {
        "description": description,
        "mean": _mean(values.values()),
        "scene_cluster_bootstrap_ci95": [low, high],
        "per_scene": dict(sorted(values.items())),
    }


def _per_order_rows(rows: Sequence[Mapping[str, object]]):
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["order"]))].append(row)
    result = []
    for (model, order), items in sorted(grouped.items()):
        result.append(
            {
                "model": model,
                "order": order,
                "mean_output_psnr": _mean(float(row["output_psnr"]) for row in items),
                "mean_output_ssim": _mean(float(row["output_ssim"]) for row in items),
                "mean_gain_psnr": _mean(float(row["gain_psnr"]) for row in items),
                "harm_rate": _mean(float(row["harm"]) for row in items),
            }
        )
    return result


def _compare(rows: Sequence[Mapping[str, object]], args: argparse.Namespace):
    values = _scene_order_values(rows)
    order_a, order_b = ORDER_NAMES
    scenes = sorted({key[2] for key in values})
    true_minus_fixed_a = {
        scene: values[("true_order", order_a, scene)]
        - values[("fixed_code", order_a, scene)]
        for scene in scenes
    }
    true_minus_fixed_b = {
        scene: values[("true_order", order_b, scene)]
        - values[("fixed_code", order_b, scene)]
        for scene in scenes
    }
    mean_improvement = {
        scene: (true_minus_fixed_a[scene] + true_minus_fixed_b[scene]) / 2.0
        for scene in scenes
    }
    worst_improvement = {
        scene: min(
            values[("true_order", order_a, scene)],
            values[("true_order", order_b, scene)],
        )
        - min(
            values[("fixed_code", order_a, scene)],
            values[("fixed_code", order_b, scene)],
        )
        for scene in scenes
    }
    specialist_minus_true = {
        scene: (
            values[("fixed_a", order_a, scene)]
            + values[("fixed_b", order_b, scene)]
            - values[("true_order", order_a, scene)]
            - values[("true_order", order_b, scene)]
        )
        / 2.0
        for scene in scenes
    }
    true_minus_balanced = {
        scene: (
            values[("true_order", order_a, scene)]
            + values[("true_order", order_b, scene)]
            - values[("balanced", order_a, scene)]
            - values[("balanced", order_b, scene)]
        )
        / 2.0
        for scene in scenes
    }
    contrasts = {
        "true_minus_fixed_order_a": _contrast(
            "Effect of the true order code on low>haze",
            true_minus_fixed_a,
            args.bootstrap_samples,
            args.generation_seed + 11,
        ),
        "true_minus_fixed_order_b": _contrast(
            "Effect of the true order code on haze>low",
            true_minus_fixed_b,
            args.bootstrap_samples,
            args.generation_seed + 22,
        ),
        "true_minus_fixed_mean": _contrast(
            "Primary endpoint: mean A/B benefit of privileged true order",
            mean_improvement,
            args.bootstrap_samples,
            args.generation_seed + 33,
        ),
        "true_minus_fixed_worst_order": _contrast(
            "Safety endpoint: scene-level worst-order benefit",
            worst_improvement,
            args.bootstrap_samples,
            args.generation_seed + 44,
        ),
        "specialists_minus_true_order": _contrast(
            "Remaining matched-order gap to the two specialists",
            specialist_minus_true,
            args.bootstrap_samples,
            args.generation_seed + 55,
        ),
        "true_order_minus_locked_balanced": _contrast(
            "Descriptive comparison with the earlier unconditioned Balanced model",
            true_minus_balanced,
            args.bootstrap_samples,
            args.generation_seed + 66,
        ),
    }
    primary = contrasts["true_minus_fixed_mean"]
    worst = contrasts["true_minus_fixed_worst_order"]
    gap = contrasts["specialists_minus_true_order"]
    useful = bool(
        float(primary["mean"]) >= args.practical_improvement_db
        and float(primary["scene_cluster_bootstrap_ci95"][0]) > 0.0
        and float(worst["scene_cluster_bootstrap_ci95"][0])
        >= -args.worst_noninferiority_margin_db
    )
    closes_gap = bool(
        float(gap["scene_cluster_bootstrap_ci95"][1])
        <= args.equivalence_margin_db
    )
    if useful and closes_gap:
        outcome = "oracle_success_order_information_closes_the_specialist_gap"
        next_step = "novelty_audit_then_one_predicted_order_restoration_test"
    elif useful:
        outcome = "oracle_partial_success_order_information_is_useful_but_not_sufficient"
        next_step = "stop_gpu_and_audit_mechanism_novelty_before_any_blind_model"
    else:
        outcome = "oracle_failure_true_order_does_not_beat_the_matched_control"
        next_step = "close_order_as_the_main_contribution"
    decision = {
        "status": "exploratory_pilot" if len(scenes) < 30 else "controlled_development",
        "independent_validation_scenes": len(scenes),
        "outcome": outcome,
        "privileged_order_information_supported": useful,
        "equivalent_to_order_specialists": closes_gap,
        "recommended_next_step": next_step,
        "project_management_gates_not_universal_thresholds": {
            "practical_improvement_db": args.practical_improvement_db,
            "equivalence_margin_db": args.equivalence_margin_db,
            "worst_noninferiority_margin_db": args.worst_noninferiority_margin_db,
        },
        "deployment_warning": (
            "The true-order model uses generator labels unavailable in blind AiOIR. "
            "Oracle success would identify useful information, not a deployable method."
        ),
        "scope_warning": (
            f"One synthetic low+haze pair and {len(scenes)} validation scenes cannot "
            "support a general AiOIR or publication-level claim."
        ),
    }
    return contrasts, decision


def main() -> None:
    args = parse_args()
    reference, reference_rows = _load_reference(args.reference_json)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable; use Kaggle GPU or pass --allow-cpu")
    use_amp = bool(args.amp and device.type == "cuda")
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
    if list(dataset.scene_ids) != reference.get("scene_ids"):
        raise RuntimeError("Validation scenes differ from the locked reference")
    true_model, true_observed, true_mismatches = _load_model(
        args.true_checkpoint, "true", device, args.allow_smoke_checkpoints
    )
    fixed_model, fixed_observed, fixed_mismatches = _load_model(
        args.fixed_checkpoint, "fixed", device, args.allow_smoke_checkpoints
    )
    true_rows = _evaluate("true_order", "true", true_model, dataset, device, use_amp)
    fixed_rows = _evaluate("fixed_code", "fixed", fixed_model, dataset, device, use_amp)
    _validate_alignment(reference_rows, true_rows)
    _validate_alignment(reference_rows, fixed_rows)
    all_rows: List[Mapping[str, object]] = [*reference_rows, *true_rows, *fixed_rows]
    contrasts, decision = _compare(all_rows, args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_sample_all.csv", all_rows)
    _write_csv(output_dir / "per_order_all.csv", _per_order_rows(all_rows))
    _write_json(output_dir / "contrasts.json", contrasts)
    _write_json(output_dir / "decision.json", decision)
    _write_json(
        output_dir / "protocol.json",
        {
            "reference_json": str(Path(args.reference_json).resolve()),
            "reference_json_sha256": _sha256(args.reference_json),
            "reference_source_zip_sha256": reference.get("source_zip_sha256"),
            "true_checkpoint": {
                "path": str(Path(args.true_checkpoint).resolve()),
                "sha256": _sha256(args.true_checkpoint),
                "observed": true_observed,
                "protocol_mismatches": true_mismatches,
            },
            "fixed_checkpoint": {
                "path": str(Path(args.fixed_checkpoint).resolve()),
                "sha256": _sha256(args.fixed_checkpoint),
                "observed": fixed_observed,
                "protocol_mismatches": fixed_mismatches,
            },
            "scene_ids": list(dataset.scene_ids),
            "test_split_loaded": False,
            "reference_sample_alignment_passed": True,
            "args": vars(args),
        },
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
