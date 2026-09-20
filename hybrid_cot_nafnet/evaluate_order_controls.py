"""Joint evaluation of Fixed-A, Fixed-B, and Balanced low+haze controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from .datasets.order_controls import ORDER_A, ORDER_B, OrderControlDataset
from .evaluate import calculate_psnr, calculate_ssim, load_torch_file, tiled_inference
from .model import build_model
from .project_config import KAGGLE_CDD11_ROOT


MODEL_KEYS = ("fixed_a", "fixed_b", "balanced")
ORDER_NAMES = (">".join(ORDER_A), ">".join(ORDER_B))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-a-checkpoint", required=True)
    parser.add_argument("--fixed-b-checkpoint", required=True)
    parser.add_argument("--balanced-checkpoint", required=True)
    parser.add_argument("--data-root", default=str(KAGGLE_CDD11_ROOT))
    parser.add_argument(
        "--output-dir", default="/kaggle/working/order_control_evaluation"
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=20260920)
    parser.add_argument("--realizations", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument(
        "--practical-specialization-db",
        type=float,
        default=0.2,
        help="Pilot management gate, not a universal scientific threshold.",
    )
    parser.add_argument(
        "--equivalence-margin-db",
        type=float,
        default=0.1,
        help="Pilot management margin, not a universal scientific threshold.",
    )
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--allow-smoke-checkpoints",
        action="store_true",
        help="Permit non-preregistered checkpoints for local code smoke tests only.",
    )
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_model(
    checkpoint_path: str | Path,
    expected_policy: str,
    device: torch.device,
    allow_smoke: bool,
):
    checkpoint = load_torch_file(checkpoint_path)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"Invalid checkpoint: {checkpoint_path}")
    training_args = checkpoint.get("args", {})
    observed = {
        "model_type": checkpoint.get("model_type"),
        "preset": checkpoint.get("preset"),
        "training_data": training_args.get("training_data"),
        "order_policy": training_args.get("order_policy"),
        "epochs": training_args.get("epochs"),
        "seed": training_args.get("seed"),
        "generation_seed": training_args.get("generation_seed"),
    }
    expected = {
        "model_type": "baseline",
        "preset": "sidd32",
        "training_data": "synthetic_order",
        "order_policy": expected_policy,
        "epochs": 20,
        "seed": 42,
        "generation_seed": 20260920,
    }
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches and not allow_smoke:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} violates the preregistered protocol: "
            f"{mismatches}"
        )
    model = build_model(
        "baseline",
        str(checkpoint.get("preset", "sidd32")),
        int(checkpoint.get("adapter_hidden", 64)),
        use_skip_gates=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), observed, mismatches


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def _cluster_ci(
    scene_values: Mapping[str, float], samples: int, seed: int
) -> Tuple[float, float]:
    values = np.asarray(list(scene_values.values()), dtype=np.float64)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _scene_means(
    rows: Sequence[Mapping[str, object]], metric: str
) -> Dict[Tuple[str, str, str], float]:
    grouped: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["order"]), str(row["scene_id"]))].append(
            float(row[metric])
        )
    return {key: _mean(values) for key, values in grouped.items()}


def _contrast(
    name: str,
    scene_values: Mapping[str, float],
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, object]:
    low, high = _cluster_ci(scene_values, bootstrap_samples, seed)
    return {
        "name": name,
        "unit": "dB",
        "scene_count": len(scene_values),
        "mean": _mean(scene_values.values()),
        "scene_cluster_bootstrap_ci95": [low, high],
        "per_scene": dict(sorted(scene_values.items())),
    }


@torch.inference_mode()
def evaluate_checkpoint(
    model_key: str,
    model,
    dataset: OrderControlDataset,
    device: torch.device,
    tile: int,
    overlap: int,
    use_amp: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        lq = sample["lq"].unsqueeze(0).float()
        gt = sample["gt"].unsqueeze(0).float()
        prediction, _ = tiled_inference(model, lq, device, tile, overlap, use_amp)
        if not torch.isfinite(prediction).all() and use_amp:
            prediction, _ = tiled_inference(model, lq, device, tile, overlap, False)
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


def summarize(
    rows: Sequence[Mapping[str, object]], args: argparse.Namespace
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    Dict[str, object],
    Dict[str, object],
]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["order"]))].append(row)
    per_order: List[Dict[str, object]] = []
    for (model, order), items in sorted(grouped.items()):
        scene_output = defaultdict(list)
        for item in items:
            scene_output[str(item["scene_id"])].append(float(item["output_psnr"]))
        scene_output_mean = {
            scene: _mean(values) for scene, values in scene_output.items()
        }
        low, high = _cluster_ci(
            scene_output_mean,
            args.bootstrap_samples,
            args.generation_seed + len(per_order),
        )
        per_order.append(
            {
                "model": model,
                "order": order,
                "scenes": len(scene_output_mean),
                "realizations_per_scene": args.realizations,
                "mean_input_psnr": _mean(float(item["input_psnr"]) for item in items),
                "mean_output_psnr": _mean(float(item["output_psnr"]) for item in items),
                "output_psnr_scene_ci95_low": low,
                "output_psnr_scene_ci95_high": high,
                "mean_gain_psnr": _mean(float(item["gain_psnr"]) for item in items),
                "mean_output_ssim": _mean(float(item["output_ssim"]) for item in items),
                "mean_gain_ssim": _mean(float(item["gain_ssim"]) for item in items),
                "harm_rate": _mean(float(item["harm"]) for item in items),
            }
        )

    per_model: List[Dict[str, object]] = []
    for model in MODEL_KEYS:
        model_orders = [item for item in per_order if item["model"] == model]
        if len(model_orders) != 2:
            raise RuntimeError(f"Expected two order summaries for {model}")
        per_model.append(
            {
                "model": model,
                "mean_output_psnr": _mean(
                    float(item["mean_output_psnr"]) for item in model_orders
                ),
                "worst_order_output_psnr": min(
                    float(item["mean_output_psnr"]) for item in model_orders
                ),
                "mean_gain_psnr": _mean(
                    float(item["mean_gain_psnr"]) for item in model_orders
                ),
                "worst_order_gain_psnr": min(
                    float(item["mean_gain_psnr"]) for item in model_orders
                ),
                "mean_output_ssim": _mean(
                    float(item["mean_output_ssim"]) for item in model_orders
                ),
                "worst_order_output_ssim": min(
                    float(item["mean_output_ssim"]) for item in model_orders
                ),
                "harm_rate": _mean(float(item["harm_rate"]) for item in model_orders),
            }
        )

    values = _scene_means(rows, "output_psnr")
    order_a, order_b = ORDER_NAMES
    scenes = sorted({key[2] for key in values})
    specialization = {
        scene: (
            (values[("fixed_a", order_a, scene)] - values[("fixed_b", order_a, scene)])
            + (values[("fixed_b", order_b, scene)] - values[("fixed_a", order_b, scene)])
        )
        / 2.0
        for scene in scenes
    }
    specialist_gap = {
        scene: (
            (values[("fixed_a", order_a, scene)] - values[("balanced", order_a, scene)])
            + (values[("fixed_b", order_b, scene)] - values[("balanced", order_b, scene)])
        )
        / 2.0
        for scene in scenes
    }
    fixed_a_order_gap = {
        scene: values[("fixed_a", order_a, scene)]
        - values[("fixed_a", order_b, scene)]
        for scene in scenes
    }
    fixed_b_order_gap = {
        scene: values[("fixed_b", order_a, scene)]
        - values[("fixed_b", order_b, scene)]
        for scene in scenes
    }
    balanced_order_gap = {
        scene: values[("balanced", order_a, scene)]
        - values[("balanced", order_b, scene)]
        for scene in scenes
    }
    matched_specialist_gap = {
        scene: values[("fixed_a", order_a, scene)]
        - values[("fixed_b", order_b, scene)]
        for scene in scenes
    }
    contrasts = {
        "specialization_difference_in_differences": _contrast(
            "Positive means each fixed model is relatively better on its trained order",
            specialization,
            args.bootstrap_samples,
            args.generation_seed + 101,
        ),
        "specialist_minus_balanced_on_matched_orders": _contrast(
            "Positive means Balanced ERM remains below the corresponding specialists",
            specialist_gap,
            args.bootstrap_samples,
            args.generation_seed + 202,
        ),
        "fixed_a_order_a_minus_b": _contrast(
            "Within Fixed-A: positive means A has higher output PSNR than B",
            fixed_a_order_gap,
            args.bootstrap_samples,
            args.generation_seed + 303,
        ),
        "fixed_b_order_a_minus_b": _contrast(
            "Within Fixed-B: positive means A has higher output PSNR than B",
            fixed_b_order_gap,
            args.bootstrap_samples,
            args.generation_seed + 404,
        ),
        "balanced_order_a_minus_b": _contrast(
            "Within Balanced ERM: positive means A has higher output PSNR than B",
            balanced_order_gap,
            args.bootstrap_samples,
            args.generation_seed + 505,
        ),
        "matched_specialist_a_minus_b": _contrast(
            "Fixed-A on A minus Fixed-B on B; descriptive evidence of residual order difficulty",
            matched_specialist_gap,
            args.bootstrap_samples,
            args.generation_seed + 606,
        ),
    }
    specialization_result = contrasts["specialization_difference_in_differences"]
    gap_result = contrasts["specialist_minus_balanced_on_matched_orders"]
    specialization_mean = float(specialization_result["mean"])
    specialization_low = float(specialization_result["scene_cluster_bootstrap_ci95"][0])
    gap_low, gap_high = [
        float(value)
        for value in gap_result["scene_cluster_bootstrap_ci95"]
    ]
    exposure_supported = bool(
        specialization_mean >= args.practical_specialization_db
        and specialization_low > 0.0
    )
    balanced_equivalent = bool(gap_high <= args.equivalence_margin_db)
    joint_limitation = bool(gap_low > args.equivalence_margin_db)
    if not exposure_supported:
        next_step = "order_exposure_not_supported_in_this_pilot_stop_the_claim"
    elif balanced_equivalent:
        next_step = "stop_new_method_random_order_erm_is_sufficient"
    elif joint_limitation:
        next_step = "run_one_preregistered_group_dro_control"
    else:
        next_step = "inconclusive_expand_independent_scenes_before_new_method"
    decision = {
        "status": "exploratory_pilot" if len(scenes) < 30 else "controlled_development",
        "independent_validation_scenes": len(scenes),
        "small_sample_warning": (
            "Fewer than 30 held-out clean scenes; do not make a generality or "
            "publication-level claim from this pilot."
            if len(scenes) < 30
            else None
        ),
        "project_management_gates_not_universal_scientific_thresholds": {
            "practical_specialization_db": args.practical_specialization_db,
            "equivalence_margin_db": args.equivalence_margin_db,
        },
        "training_order_exposure_supported": exposure_supported,
        "balanced_erm_equivalent_to_specialists": balanced_equivalent,
        "joint_learning_limitation_supported": joint_limitation,
        "recommended_next_step": next_step,
        "interpretation_guardrail": (
            "Specialists are controls, not theoretical ceilings. A positive pilot "
            "does not establish novelty or validate a new architecture."
        ),
    }
    return per_order, per_model, contrasts, decision


def main() -> None:
    args = parse_args()
    if args.realizations <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("realizations and bootstrap-samples must be positive")
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
    checkpoints = {
        "fixed_a": args.fixed_a_checkpoint,
        "fixed_b": args.fixed_b_checkpoint,
        "balanced": args.balanced_checkpoint,
    }
    rows: List[Dict[str, object]] = []
    checkpoint_protocol = {}
    for model_key in MODEL_KEYS:
        model, observed, mismatches = _load_model(
            checkpoints[model_key],
            model_key,
            device,
            args.allow_smoke_checkpoints,
        )
        checkpoint_protocol[model_key] = {
            "path": str(Path(checkpoints[model_key]).resolve()),
            "sha256": _sha256(checkpoints[model_key]),
            "observed": observed,
            "protocol_mismatches": mismatches,
        }
        rows.extend(
            evaluate_checkpoint(
                model_key,
                model,
                dataset,
                device,
                args.tile,
                args.overlap,
                use_amp,
            )
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Evaluated {model_key}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_order, per_model, contrasts, decision = summarize(rows, args)
    _write_csv(output_dir / "per_sample.csv", rows)
    _write_csv(output_dir / "per_order.csv", per_order)
    _write_csv(output_dir / "per_model.csv", per_model)
    _write_json(output_dir / "contrasts.json", contrasts)
    _write_json(output_dir / "decision.json", decision)
    _write_json(
        output_dir / "protocol.json",
        {
            "formation_orders": {"A": list(ORDER_A), "B": list(ORDER_B)},
            "dataset": dataset.summary(),
            "scene_ids": list(dataset.scene_ids),
            "scene_disjoint_development_split": True,
            "test_split_loaded": False,
            "same_realization_policy": (
                "A/B reuse the same clean image, illumination, noise, haze "
                "transmission, and atmospheric light."
            ),
            "raw_inputs_are_primary": True,
            "severity_matching_used": False,
            "checkpoint_protocol": checkpoint_protocol,
            "args": vars(args),
        },
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
