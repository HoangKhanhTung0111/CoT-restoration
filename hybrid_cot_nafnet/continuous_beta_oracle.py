"""Audit image- and block-level continuous residual-strength oracles.

This is a privileged validation-only diagnostic: ground truth is used to solve
the MSE-optimal beta analytically.  It is an upper bound, not a deployable
controller, and the CDD-11 test split is never loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .datasets import CDD11Dataset, find_cdd11_root
from .evaluate import (
    calculate_psnr,
    calculate_ssim,
    oracle_block_output,
)
from .project_config import KAGGLE_CDD11_ROOT
from .train_gain_predictor import load_restorer, restore, seed_everything


SINGLE_DEGRADATIONS = {"low", "haze", "rain", "snow"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Frozen A3-M best.pt")
    parser.add_argument("--data-root", default=str(KAGGLE_CDD11_ROOT))
    parser.add_argument(
        "--output-dir", default="/kaggle/working/continuous_beta_oracle"
    )
    parser.add_argument("--git-commit", default="", help="Repository commit for provenance")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--fixed-beta", type=float, default=0.99,
        help="Calibration-selected control from the preceding gain-predictor run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def optimal_beta(error: Tensor, residual: Tensor) -> float:
    """Solve min_beta ||error + beta * residual||^2 for beta in [0,1]."""
    denominator = residual.double().square().sum().item()
    if denominator <= 1e-20:
        return 0.0
    numerator = -(error.double() * residual.double()).sum().item()
    return float(min(1.0, max(0.0, numerator / denominator)))


def continuous_image_oracle(
    degraded: Tensor, restored: Tensor, target: Tensor
) -> Tuple[Tensor, float]:
    residual = restored - degraded
    beta = optimal_beta(degraded - target, residual)
    return degraded + beta * residual, beta


def continuous_block_oracle(
    degraded: Tensor, restored: Tensor, target: Tensor, block_size: int
) -> Tuple[Tensor, Tensor]:
    """Return the analytic MSE-optimal beta independently for each block."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    height, width = target.shape[-2:]
    pad_height, pad_width = (-height) % block_size, (-width) % block_size
    residual = restored - degraded
    error = degraded - target
    products = torch.cat((residual.square(), error * residual), dim=1)
    products = F.pad(products, (0, pad_width, 0, pad_height))
    unfolded = F.unfold(products, kernel_size=block_size, stride=block_size)[0]
    channel_area = block_size * block_size
    denominator = unfolded[: 3 * channel_area].double().sum(dim=0)
    cross = unfolded[3 * channel_area :].double().sum(dim=0)
    beta = torch.where(
        denominator > 1e-20,
        (-cross / denominator.clamp_min(1e-20)).clamp(0, 1),
        torch.zeros_like(denominator),
    ).float()
    rows, columns = math.ceil(height / block_size), math.ceil(width / block_size)
    beta_grid = beta.reshape(1, 1, rows, columns)
    beta_map = beta_grid.repeat_interleave(block_size, 2).repeat_interleave(
        block_size, 3
    )[..., :height, :width]
    return degraded + beta_map * residual, beta


def image_binary_oracle(
    degraded: Tensor, restored: Tensor, target: Tensor
) -> Tuple[Tensor, float]:
    use_restored = bool(
        (restored - target).square().sum() < (degraded - target).square().sum()
    )
    return (restored if use_restored else degraded), float(use_restored)


def metric_pair(output: Tensor, target: Tensor) -> Tuple[float, float]:
    return calculate_psnr(output, target), calculate_ssim(output, target)


def summarize_method(rows: Sequence[dict], prefix: str) -> dict:
    return {
        "mean_psnr": float(np.mean([row[f"{prefix}_psnr"] for row in rows])),
        "mean_ssim": float(np.mean([row[f"{prefix}_ssim"] for row in rows])),
        "images_worse_than_input": int(sum(
            row[f"{prefix}_psnr"] < row["input_psnr"] - 1e-12 for row in rows
        )),
    }


def grouped_gain(
    rows: Sequence[dict], group_key: str, continuous_key: str, binary_key: str
) -> Dict[str, dict]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    return {
        name: {
            "count": len(items),
            "continuous_psnr": float(np.mean([row[continuous_key] for row in items])),
            "binary_psnr": float(np.mean([row[binary_key] for row in items])),
            "gain_db": float(np.mean([
                row[continuous_key] - row[binary_key] for row in items
            ])),
        }
        for name, items in sorted(groups.items())
    }


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> Tuple[List[dict], dict]:
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable; enable a Kaggle GPU or pass --allow-cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = bool(args.amp and device.type == "cuda")
    seed_everything(args.seed)
    restorer, checkpoint = load_restorer(args.checkpoint, device)
    dataset = CDD11Dataset(
        find_cdd11_root(args.data_root), mode="val", crop_size=0,
        val_fraction=args.val_fraction, split_seed=args.seed, augment=False,
    )
    rows: List[dict] = []
    all_block_betas: List[np.ndarray] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        degraded, target = sample["lq"].unsqueeze(0), sample["gt"].unsqueeze(0)
        restored = restore(
            restorer, degraded, device, args.tile, args.overlap, args.amp
        )
        fixed = degraded + args.fixed_beta * (restored - degraded)
        binary_image, binary_image_fraction = image_binary_oracle(
            degraded, restored, target
        )
        binary_block, binary_block_fraction = oracle_block_output(
            degraded, restored, target, args.block_size
        )
        continuous_image, image_beta = continuous_image_oracle(
            degraded, restored, target
        )
        continuous_block, block_betas = continuous_block_oracle(
            degraded, restored, target, args.block_size
        )
        all_block_betas.append(block_betas.numpy())
        outputs = {
            "input": degraded,
            "beta1": restored,
            "fixed_beta": fixed,
            "binary_image": binary_image,
            "binary_block": binary_block,
            "continuous_image": continuous_image,
            "continuous_block": continuous_block,
        }
        row = {
            "scene_id": sample["scene_id"],
            "type": sample["degradation_type"],
            "group": (
                "single" if sample["degradation_type"] in SINGLE_DEGRADATIONS
                else "composite"
            ),
            "image_beta": image_beta,
            "block_beta_mean": float(block_betas.mean()),
            "block_beta_near_zero_fraction": float((block_betas <= 0.05).float().mean()),
            "block_beta_interior_fraction": float(
                ((block_betas > 0.05) & (block_betas < 0.95)).float().mean()
            ),
            "block_beta_near_one_fraction": float((block_betas >= 0.95).float().mean()),
            "binary_image_restored_fraction": binary_image_fraction,
            "binary_block_restored_fraction": binary_block_fraction,
        }
        for name, output in outputs.items():
            row[f"{name}_psnr"], row[f"{name}_ssim"] = metric_pair(output, target)
        row["continuous_over_binary_block_db"] = (
            row["continuous_block_psnr"] - row["binary_block_psnr"]
        )
        row["continuous_block_over_image_db"] = (
            row["continuous_block_psnr"] - row["continuous_image_psnr"]
        )
        rows.append(row)
        print(
            f"[{index + 1:03d}/{len(dataset)}] {row['type']}/{row['scene_id']} "
            f"binary={row['binary_block_psnr']:.3f} "
            f"continuous={row['continuous_block_psnr']:.3f} "
            f"delta={row['continuous_over_binary_block_db']:+.3f}"
        )

    method_names = (
        "input", "beta1", "fixed_beta", "binary_image", "binary_block",
        "continuous_image", "continuous_block",
    )
    methods = {name: summarize_method(rows, name) for name in method_names}
    per_scene = grouped_gain(
        rows, "scene_id", "continuous_block_psnr", "binary_block_psnr"
    )
    per_type = grouped_gain(
        rows, "type", "continuous_block_psnr", "binary_block_psnr"
    )
    per_group = grouped_gain(
        rows, "group", "continuous_block_psnr", "binary_block_psnr"
    )
    block_betas = np.concatenate(all_block_betas)
    continuous_gain = (
        methods["continuous_block"]["mean_psnr"]
        - methods["binary_block"]["mean_psnr"]
    )
    positive_scene_count = sum(item["gain_db"] > 0 for item in per_scene.values())
    both_groups_positive = all(item["gain_db"] > 0 for item in per_group.values())
    gates = {
        "at_least_0_1_db_over_binary_block": continuous_gain >= 0.1,
        "positive_on_at_least_4_of_5_scenes": positive_scene_count >= 4,
        "positive_for_single_and_composite_groups": both_groups_positive,
    }
    summary = {
        "split": "validation_only",
        "count": len(rows),
        "block_size": args.block_size,
        "fixed_beta": args.fixed_beta,
        "git_commit": args.git_commit or None,
        "restorer": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "epoch_index": checkpoint.get("epoch"),
            "model_type": checkpoint.get("model_type"),
            "preset": checkpoint.get("preset"),
        },
        "methods": methods,
        "continuous_block_gain_over_binary_block_db": continuous_gain,
        "continuous_block_gain_over_fixed_beta_db": (
            methods["continuous_block"]["mean_psnr"]
            - methods["fixed_beta"]["mean_psnr"]
        ),
        "continuous_block_gain_over_continuous_image_db": (
            methods["continuous_block"]["mean_psnr"]
            - methods["continuous_image"]["mean_psnr"]
        ),
        "block_beta_distribution": {
            "mean": float(block_betas.mean()),
            "near_zero_fraction_beta_le_0_05": float(np.mean(block_betas <= 0.05)),
            "interior_fraction_0_05_to_0_95": float(np.mean(
                (block_betas > 0.05) & (block_betas < 0.95)
            )),
            "near_one_fraction_beta_ge_0_95": float(np.mean(block_betas >= 0.95)),
        },
        "per_scene": per_scene,
        "per_type": per_type,
        "per_group": per_group,
        "decision": {
            "status": "continue_to_controller_pilot" if all(gates.values()) else "stop_continuous_controller",
            "gates": gates,
            "positive_scene_count": positive_scene_count,
            "rule": (
                "All resource-allocation gates must pass. The 0.1 dB threshold "
                "is a prespecified engineering gate, not a scientific boundary."
            ),
        },
    }
    return rows, summary


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.block_size <= 0:
        raise ValueError("block-size must be positive")
    if not 0 <= args.fixed_beta <= 1:
        raise ValueError("fixed-beta must be in [0,1]")
    rows, summary = evaluate(args)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "per_image_metrics.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    protocol = {
        "purpose": "privileged continuous-beta feasibility audit",
        "test_split_touched": False,
        "validation_only": True,
        "args": vars(args),
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
