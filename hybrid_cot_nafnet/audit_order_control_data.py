"""Read-only data gate for the low+haze formation-order control experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .datasets.order_controls import ORDER_A, ORDER_B, OrderControlDataset
from .project_config import KAGGLE_CDD11_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(KAGGLE_CDD11_ROOT))
    parser.add_argument(
        "--output", default="/kaggle/working/order_control_data_gate.json"
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=20260920)
    parser.add_argument("--train-realizations", type=int, default=16)
    parser.add_argument("--val-realizations", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = {
        "root": args.data_root,
        "val_fraction": args.val_fraction,
        "split_seed": args.split_seed,
        "generation_seed": args.generation_seed,
    }
    fixed_a = OrderControlDataset(
        mode="train",
        order_policy="fixed_a",
        realizations=args.train_realizations,
        crop_size=args.crop_size,
        **common,
    )
    fixed_b = OrderControlDataset(
        mode="train",
        order_policy="fixed_b",
        realizations=args.train_realizations,
        crop_size=args.crop_size,
        **common,
    )
    balanced = OrderControlDataset(
        mode="train",
        order_policy="balanced",
        realizations=args.train_realizations,
        crop_size=args.crop_size,
        **common,
    )
    validation = OrderControlDataset(
        mode="val",
        order_policy="both",
        realizations=args.val_realizations,
        crop_size=0,
        **common,
    )
    checks = {
        "identical_train_scene_ids": (
            fixed_a.scene_ids == fixed_b.scene_ids == balanced.scene_ids
        ),
        "train_validation_scene_disjoint": not bool(
            set(fixed_a.scene_ids).intersection(validation.scene_ids)
        ),
        "equal_training_update_budget": (
            len(fixed_a) == len(fixed_b) == len(balanced)
        ),
        "balanced_realization_count_is_even": args.train_realizations % 2 == 0,
    }
    pair_mse = []
    for pair_index in range(len(validation) // 2):
        sample_a = validation[pair_index * 2]
        sample_b = validation[pair_index * 2 + 1]
        if sample_a["formation_order"] != ">".join(ORDER_A):
            raise RuntimeError("Validation A/B ordering is corrupted")
        if sample_b["formation_order"] != ">".join(ORDER_B):
            raise RuntimeError("Validation A/B ordering is corrupted")
        if (
            sample_a["scene_id"] != sample_b["scene_id"]
            or sample_a["realization"] != sample_b["realization"]
            or sample_a["generation_seed"] != sample_b["generation_seed"]
            or not torch.equal(sample_a["gt"], sample_b["gt"])
        ):
            raise RuntimeError("A/B pair does not share scene, target, and realization")
        pair_mse.append(
            float((sample_a["lq"] - sample_b["lq"]).float().square().mean().item())
        )
    checks["all_validation_pairs_have_distinct_inputs"] = all(
        value > 0.0 for value in pair_mse
    )
    passed = all(checks.values())
    report = {
        "passes": passed,
        "checks": checks,
        "formation_orders": {"A": list(ORDER_A), "B": list(ORDER_B)},
        "train_scene_ids": list(fixed_a.scene_ids),
        "validation_scene_ids": list(validation.scene_ids),
        "test_split_loaded": False,
        "train_samples_per_policy": len(fixed_a),
        "validation_ab_samples": len(validation),
        "validation_pairs": len(pair_mse),
        "ab_input_mse_range": [min(pair_mse), max(pair_mse)],
        "scope_warning": (
            "This verifies implementation invariants for a synthetic stress test; "
            "it does not establish physical realism or inverse-problem equivalence."
        ),
        "args": vars(args),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("Order-control data gate failed")


if __name__ == "__main__":
    main()
