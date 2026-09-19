"""Oracle-guided error decomposition and controller learnability audit.

The audit separates residual-strength error from error that remains after the
best clipped beta, reports the unconstrained beta regime, and evaluates the
same frozen controller on source-disjoint train/calibration/validation scenes.
It is diagnostic only and never loads CDD-11_test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from .datasets import CDD11Dataset, find_cdd11_root
from .evaluate import calculate_psnr, calculate_ssim, load_torch_file
from .project_config import KAGGLE_CDD11_ROOT
from .train_beta_controller import (
    BetaController,
    block_quadratics,
    controller_output,
    predict_betas,
)
from .train_gain_predictor import blocks_from_pair, load_restorer, restore, seed_everything


SINGLE_DEGRADATIONS = {"low", "haze", "rain", "snow"}
METHODS = ("input", "beta1", "fixed_beta", "controller", "oracle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Frozen A3-M best.pt")
    parser.add_argument("--controller", required=True, help="Frozen beta_controller.pt")
    parser.add_argument("--data-root", default=str(KAGGLE_CDD11_ROOT))
    parser.add_argument(
        "--output-dir", default="/kaggle/working/error_decomposition_audit"
    )
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--fixed-beta", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def load_controller(path: str | Path, device: torch.device) -> Tuple[BetaController, dict]:
    payload = load_torch_file(path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError("Expected beta_controller.pt with a 'model' state dict")
    controller = BetaController(int(payload.get("width", 24)))
    controller.load_state_dict(payload["model"], strict=True)
    return controller.to(device).eval(), payload


def iter_selected(dataset: CDD11Dataset, scene_ids: Iterable[str]):
    selected = set(scene_ids)
    for index, sample in enumerate(dataset.samples):
        if sample[3] in selected:
            yield dataset[index]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def method_summary(rows: Sequence[dict], method: str) -> dict:
    return {
        "mean_psnr": float(np.mean([row[f"{method}_psnr"] for row in rows])),
        "mean_ssim": float(np.mean([row[f"{method}_ssim"] for row in rows])),
        "images_worse_than_input": int(sum(
            row[f"{method}_psnr"] < row["input_psnr"] - 1e-12 for row in rows
        )),
    }


def grouped_controller_gain(rows: Sequence[dict], key: str) -> Dict[str, dict]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {
        name: {
            "count": len(items),
            "controller_gain_over_fixed_beta_db": float(np.mean([
                row["controller_psnr"] - row["fixed_beta_psnr"] for row in items
            ])),
            "oracle_gain_over_fixed_beta_db": float(np.mean([
                row["oracle_psnr"] - row["fixed_beta_psnr"] for row in items
            ])),
            "recoverable_error_fraction": float(
                sum(row["recoverable_sse"] for row in items)
                / max(1e-12, sum(row["fixed_sse"] for row in items))
            ),
        }
        for name, items in sorted(groups.items())
    }


def summarize_split(
    split_name: str,
    rows: Sequence[dict],
    predicted_betas: Sequence[np.ndarray],
    oracle_betas: Sequence[np.ndarray],
    raw_betas: Sequence[np.ndarray],
    raw_categories: Sequence[np.ndarray],
    category_headroom: Sequence[np.ndarray],
    pipeline_max_relative_error: float,
) -> dict:
    methods = {name: method_summary(rows, name) for name in METHODS}
    predicted = np.concatenate(predicted_betas)
    oracle = np.concatenate(oracle_betas)
    raw = np.concatenate(raw_betas)
    categories = np.concatenate(raw_categories)
    headroom_by_block = np.concatenate(category_headroom)
    fixed_sse = float(sum(row["fixed_sse"] for row in rows))
    oracle_sse = float(sum(row["oracle_sse"] for row in rows))
    controller_sse = float(sum(row["controller_sse"] for row in rows))
    recoverable_sse = fixed_sse - oracle_sse
    controller_recovered_sse = fixed_sse - controller_sse

    category_names = {0: "raw_beta_le_0", 1: "raw_beta_between_0_and_1", 2: "raw_beta_ge_1"}
    regime = {}
    for category, name in category_names.items():
        mask = categories == category
        regime[name] = {
            "block_fraction": float(mask.mean()),
            "headroom_fraction": float(
                headroom_by_block[mask].sum() / max(1e-12, headroom_by_block.sum())
            ),
            "raw_beta_median": float(np.median(raw[mask])) if mask.any() else None,
        }

    gain_over_fixed = (
        methods["controller"]["mean_psnr"] - methods["fixed_beta"]["mean_psnr"]
    )
    oracle_gain = methods["oracle"]["mean_psnr"] - methods["fixed_beta"]["mean_psnr"]
    return {
        "split": split_name,
        "count": len(rows),
        "methods": methods,
        "controller_gain_over_fixed_beta_db": gain_over_fixed,
        "oracle_gain_over_fixed_beta_db": oracle_gain,
        "error_decomposition": {
            "fixed_sse": fixed_sse,
            "recoverable_sse_by_clipped_beta": recoverable_sse,
            "remaining_sse_after_oracle": oracle_sse,
            "recoverable_fraction_of_fixed_error": recoverable_sse / max(fixed_sse, 1e-12),
            "remaining_fraction_of_fixed_error": oracle_sse / max(fixed_sse, 1e-12),
            "controller_recovered_sse": controller_recovered_sse,
            "controller_fraction_of_oracle_recoverable_sse": (
                controller_recovered_sse / max(recoverable_sse, 1e-12)
            ),
        },
        "block_beta_prediction": {
            "predicted_mean": float(predicted.mean()),
            "oracle_clipped_mean": float(oracle.mean()),
            "mae": float(np.mean(np.abs(predicted - oracle))),
            "pearson": pearson(predicted, oracle),
            "predicted_near_zero_fraction": float(np.mean(predicted <= 0.05)),
            "oracle_near_zero_fraction": float(np.mean(oracle <= 0.05)),
            "predicted_interior_fraction": float(np.mean((predicted > 0.05) & (predicted < 0.95))),
            "oracle_interior_fraction": float(np.mean((oracle > 0.05) & (oracle < 0.95))),
            "predicted_near_one_fraction": float(np.mean(predicted >= 0.95)),
            "oracle_near_one_fraction": float(np.mean(oracle >= 0.95)),
        },
        "unclipped_beta_regimes": regime,
        "per_scene": grouped_controller_gain(rows, "scene_id"),
        "per_type": grouped_controller_gain(rows, "type"),
        "per_group": grouped_controller_gain(rows, "group"),
        "pipeline_max_relative_sse_error": pipeline_max_relative_error,
    }


@torch.inference_mode()
def audit_split(
    split_name: str,
    dataset: CDD11Dataset,
    scene_ids: Sequence[str],
    restorer: nn.Module,
    controller: nn.Module,
    block_size: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    predicted_betas: List[np.ndarray] = []
    oracle_betas: List[np.ndarray] = []
    raw_betas: List[np.ndarray] = []
    raw_categories: List[np.ndarray] = []
    category_headroom: List[np.ndarray] = []
    pipeline_errors: List[float] = []
    selected_ids = set(scene_ids)
    sample_count = sum(sample[3] in selected_ids for sample in dataset.samples)

    for index, sample in enumerate(iter_selected(dataset, scene_ids), start=1):
        degraded, target = sample["lq"].unsqueeze(0), sample["gt"].unsqueeze(0)
        restored = restore(restorer, degraded, device, args.tile, args.overlap, args.amp)
        pairs = blocks_from_pair(degraded, restored, block_size)
        predicted = predict_betas(controller, pairs.half(), device, args.batch_size)
        coefficients = block_quadratics(degraded, restored, target, block_size)
        a, b, c, oracle_sse_blocks = coefficients.unbind(dim=1)
        valid = a > 1e-20
        raw = torch.zeros_like(a)
        raw[valid] = -b[valid] / (2 * a[valid])
        oracle_beta = raw.clamp(0, 1)
        controller_sse_blocks = (a * predicted.square() + b * predicted + c).clamp_min(0)
        fixed_sse_blocks = (
            a * args.fixed_beta**2 + b * args.fixed_beta + c
        ).clamp_min(0)
        headroom = (fixed_sse_blocks - oracle_sse_blocks).clamp_min(0)
        categories = torch.where(
            raw <= 0, torch.zeros_like(raw, dtype=torch.long),
            torch.where(raw < 1, torch.ones_like(raw, dtype=torch.long),
                        torch.full_like(raw, 2, dtype=torch.long)),
        )

        fixed = degraded + args.fixed_beta * (restored - degraded)
        controlled = controller_output(degraded, restored, predicted, block_size)
        rows_count, columns_count = (
            math.ceil(target.shape[-2] / block_size),
            math.ceil(target.shape[-1] / block_size),
        )
        oracle_map = oracle_beta.reshape(1, 1, rows_count, columns_count)
        oracle_map = oracle_map.repeat_interleave(block_size, 2).repeat_interleave(
            block_size, 3
        )[..., : target.shape[-2], : target.shape[-1]]
        oracle_output = degraded + oracle_map * (restored - degraded)
        outputs = {
            "input": degraded,
            "beta1": restored,
            "fixed_beta": fixed,
            "controller": controlled,
            "oracle": oracle_output,
        }
        row = {
            "audit_split": split_name,
            "scene_id": sample["scene_id"],
            "type": sample["degradation_type"],
            "group": (
                "single" if sample["degradation_type"] in SINGLE_DEGRADATIONS
                else "composite"
            ),
            "fixed_sse": float(fixed_sse_blocks.double().sum()),
            "controller_sse": float(controller_sse_blocks.double().sum()),
            "oracle_sse": float(oracle_sse_blocks.double().sum()),
            "recoverable_sse": float(headroom.double().sum()),
            "raw_beta_le_zero_fraction": float((raw <= 0).float().mean()),
            "raw_beta_between_zero_one_fraction": float(((raw > 0) & (raw < 1)).float().mean()),
            "raw_beta_ge_one_fraction": float((raw >= 1).float().mean()),
            "oracle_beta_mean": float(oracle_beta.mean()),
            "controller_beta_mean": float(predicted.mean()),
            "block_beta_mae": float((predicted - oracle_beta).abs().mean()),
        }
        for name, output in outputs.items():
            row[f"{name}_psnr"] = calculate_psnr(output, target)
            row[f"{name}_ssim"] = calculate_ssim(output, target)

        pixel_fixed_sse = float((fixed - target).double().square().sum())
        pixel_controller_sse = float((controlled - target).double().square().sum())
        pixel_oracle_sse = float((oracle_output - target).double().square().sum())
        for analytic, pixel in (
            (row["fixed_sse"], pixel_fixed_sse),
            (row["controller_sse"], pixel_controller_sse),
            (row["oracle_sse"], pixel_oracle_sse),
        ):
            pipeline_errors.append(abs(analytic - pixel) / max(1.0, abs(pixel)))

        rows.append(row)
        predicted_betas.append(predicted.numpy())
        oracle_betas.append(oracle_beta.numpy())
        raw_betas.append(raw.numpy())
        raw_categories.append(categories.numpy())
        category_headroom.append(headroom.numpy())
        if index % 11 == 0 or index == sample_count:
            print(
                f"{split_name} [{index:03d}/{sample_count}] "
                f"{row['type']}/{row['scene_id']} controller={row['controller_psnr']:.3f} "
                f"fixed={row['fixed_beta_psnr']:.3f} oracle={row['oracle_psnr']:.3f}"
            )

    max_pipeline_error = max(pipeline_errors, default=0.0)
    if max_pipeline_error > 1e-5:
        raise RuntimeError(
            f"Block/pixel SSE consistency check failed: relative error={max_pipeline_error}"
        )
    summary = summarize_split(
        split_name, rows, predicted_betas, oracle_betas, raw_betas,
        raw_categories, category_headroom, max_pipeline_error,
    )
    return rows, summary


def diagnose(summaries: Dict[str, dict]) -> dict:
    train = summaries["controller_train"]
    calibration = summaries["calibration"]
    validation = summaries["validation"]
    train_capture = train["error_decomposition"][
        "controller_fraction_of_oracle_recoverable_sse"
    ]
    calibration_capture = calibration["error_decomposition"][
        "controller_fraction_of_oracle_recoverable_sse"
    ]
    validation_recoverable = validation["error_decomposition"][
        "recoverable_fraction_of_fixed_error"
    ]
    pipeline_consistent = all(
        summary["pipeline_max_relative_sse_error"] <= 1e-5
        for summary in summaries.values()
    )
    if not pipeline_consistent:
        primary = "pipeline_inconsistency"
    elif train_capture < 0.2 or train["controller_gain_over_fixed_beta_db"] <= 0:
        primary = "controller_fit_or_objective_failure"
    elif calibration_capture < 0.1 or calibration["controller_gain_over_fixed_beta_db"] <= 0:
        primary = "scene_generalization_failure_before_validation"
    elif validation["controller_gain_over_fixed_beta_db"] <= 0:
        primary = "validation_generalization_failure"
    else:
        primary = "controller_has_deployable_signal"
    return {
        "primary_diagnosis": primary,
        "pipeline_consistent": pipeline_consistent,
        "train_controller_capture_fraction": train_capture,
        "calibration_controller_capture_fraction": calibration_capture,
        "validation_recoverable_error_fraction": validation_recoverable,
        "amplitude_is_material_engineering_signal": validation_recoverable >= 0.2,
        "rules": {
            "fit": "train gain must be positive and capture >=20% of oracle-recoverable SSE",
            "calibration_generalization": "calibration gain must be positive and capture >=10%",
            "amplitude_materiality": "clipped beta must remove >=20% of fixed-beta SSE",
            "pipeline": "analytic and rendered SSE relative error <=1e-5",
        },
        "next_step": (
            "run_controlled_information-source probes only if the controller fits train; "
            "otherwise inspect objective/sampling/capacity before any new architecture"
        ),
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0 <= args.fixed_beta <= 1:
        raise ValueError("fixed-beta must be in [0,1]")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable; enable a Kaggle GPU or pass --allow-cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = bool(args.amp and device.type == "cuda")
    seed_everything(args.seed)
    restorer, _ = load_restorer(args.checkpoint, device)
    controller, controller_payload = load_controller(args.controller, device)
    block_size = int(controller_payload.get("block_size", 32))
    train_ids = list(controller_payload.get("controller_train_scene_ids", []))
    calibration_ids = list(controller_payload.get("calibration_scene_ids", []))
    if not train_ids or not calibration_ids:
        raise RuntimeError("Controller checkpoint is missing its scene-level split")
    root = find_cdd11_root(args.data_root)
    train_dataset = CDD11Dataset(
        root, mode="train", crop_size=0, val_fraction=args.val_fraction,
        split_seed=args.seed, augment=False,
    )
    validation_dataset = CDD11Dataset(
        root, mode="val", crop_size=0, val_fraction=args.val_fraction,
        split_seed=args.seed, augment=False,
    )
    split_specs = {
        "controller_train": (train_dataset, train_ids),
        "calibration": (train_dataset, calibration_ids),
        "validation": (validation_dataset, list(validation_dataset.scene_ids)),
    }
    all_rows: List[dict] = []
    summaries: Dict[str, dict] = {}
    for split_name, (dataset, scene_ids) in split_specs.items():
        rows, summary = audit_split(
            split_name, dataset, scene_ids, restorer, controller, block_size,
            device, args,
        )
        all_rows.extend(rows)
        summaries[split_name] = summary
    diagnosis = diagnose(summaries)
    result = {
        "git_commit": args.git_commit or None,
        "test_split_touched": False,
        "block_size": block_size,
        "fixed_beta": args.fixed_beta,
        "splits": summaries,
        "diagnosis": diagnosis,
    }
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "per_image_metrics.csv", all_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    protocol = {
        "purpose": "oracle-guided error decomposition and controller learnability audit",
        "test_split_touched": False,
        "restorer_checkpoint": str(Path(args.checkpoint).resolve()),
        "controller_checkpoint": str(Path(args.controller).resolve()),
        "controller_train_scene_ids": train_ids,
        "calibration_scene_ids": calibration_ids,
        "validation_scene_ids": list(validation_dataset.scene_ids),
        "args": vars(args),
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
