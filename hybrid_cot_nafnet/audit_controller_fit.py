"""Training-only sampling and memorization audit for the beta controller.

This reproduces the exact 64-block sampling stream used by the controller,
compares its frozen checkpoint with fixed beta on sampled and full training
blocks, then asks the same architecture to memorize all blocks of one
high-headroom training image.  Calibration, validation, and test are not loaded.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .datasets import CDD11Dataset, find_cdd11_root
from .evaluate import calculate_psnr, load_torch_file
from .project_config import KAGGLE_CDD11_ROOT
from .train_beta_controller import (
    BetaController,
    ControllerBlocks,
    block_quadratics,
    controller_output,
    normalized_regret,
    predict_betas,
)
from .train_gain_predictor import blocks_from_pair, load_restorer, restore, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Frozen A3-M best.pt")
    parser.add_argument("--controller", required=True, help="Frozen beta_controller.pt")
    parser.add_argument("--data-root", default=str(KAGGLE_CDD11_ROOT))
    parser.add_argument("--output-dir", default="/kaggle/working/controller_fit_audit")
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--fixed-beta", type=float, default=0.99)
    parser.add_argument("--blocks-per-image", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--memorization-epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controller-split-seed", type=int, default=31415)
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


def regret_values(beta: Tensor, coefficients: Tensor) -> Tensor:
    a, b, c, oracle = coefficients.unbind(dim=1)
    predicted_sse = a * beta.square() + b * beta + c
    restored_sse = (a + b + c).clamp_min(0)
    scale = (c + restored_sse).clamp_min(1e-6)
    return (predicted_sse - oracle).clamp_min(0) / scale


def coefficient_metrics(
    beta: Tensor, coefficients: Tensor, fixed_beta: float
) -> Dict[str, float]:
    a, b, c, oracle = coefficients.unbind(dim=1)
    predicted = (a * beta.square() + b * beta + c).clamp_min(0)
    fixed = (a * fixed_beta**2 + b * fixed_beta + c).clamp_min(0)
    headroom = (fixed - oracle).sum().item()
    recovered = (fixed - predicted).sum().item()
    fixed_regret = regret_values(torch.full_like(beta, fixed_beta), coefficients)
    predicted_regret = regret_values(beta, coefficients)
    raw = torch.where(
        a > 1e-20, -b / (2 * a.clamp_min(1e-20)), torch.zeros_like(a)
    )
    oracle_beta = raw.clamp(0, 1)
    beta_std = beta.std(unbiased=False).item()
    oracle_std = oracle_beta.std(unbiased=False).item()
    return {
        "block_count": len(beta),
        "fixed_normalized_regret": float(fixed_regret.mean()),
        "controller_normalized_regret": float(predicted_regret.mean()),
        "fixed_sse": float(fixed.sum()),
        "controller_sse": float(predicted.sum()),
        "oracle_sse": float(oracle.sum()),
        "oracle_recoverable_sse": headroom,
        "controller_recovered_sse": recovered,
        "controller_capture_fraction": recovered / max(headroom, 1e-12),
        "beta_mae": float((beta - oracle_beta).abs().mean()),
        "beta_pearson": (
            float(np.corrcoef(beta.numpy(), oracle_beta.numpy())[0, 1])
            if beta_std > 0 and oracle_std > 0 else 0.0
        ),
    }


@torch.inference_mode()
def collect_training_audit(
    dataset: CDD11Dataset,
    scene_ids: Sequence[str],
    restorer: nn.Module,
    controller: nn.Module,
    block_size: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[dict, dict, dict]:
    generator = torch.Generator().manual_seed(args.controller_split_seed)
    sampled_betas, sampled_coefficients = [], []
    full_betas, full_coefficients = [], []
    selected_ids = set(scene_ids)
    sample_count = sum(sample[3] in selected_ids for sample in dataset.samples)
    best_candidate = None

    for index, sample in enumerate(iter_selected(dataset, scene_ids), start=1):
        degraded, target = sample["lq"].unsqueeze(0), sample["gt"].unsqueeze(0)
        restored = restore(restorer, degraded, device, args.tile, args.overlap, args.amp)
        # Match build_cache exactly: the controller was trained on clamped FP16
        # pairs, while the quadratic target retains the unclipped restoration.
        pairs = blocks_from_pair(degraded, restored, block_size).clamp(0, 1).half()
        coefficients = block_quadratics(degraded, restored, target, block_size)
        beta = predict_betas(controller, pairs, device, args.batch_size)
        keep = min(args.blocks_per_image, len(pairs))
        sampled_indices = torch.randperm(len(pairs), generator=generator)[:keep]
        sampled_betas.append(beta[sampled_indices])
        sampled_coefficients.append(coefficients[sampled_indices])
        full_betas.append(beta)
        full_coefficients.append(coefficients)

        a, b, c, oracle = coefficients.unbind(dim=1)
        fixed = (a * args.fixed_beta**2 + b * args.fixed_beta + c).clamp_min(0)
        headroom = (fixed - oracle).clamp_min(0).sum().item()
        recoverable_fraction = headroom / max(fixed.sum().item(), 1e-12)
        candidate_key = (recoverable_fraction, headroom)
        if best_candidate is None or candidate_key > best_candidate["key"]:
            best_candidate = {
                "key": candidate_key,
                "scene_id": sample["scene_id"],
                "type": sample["degradation_type"],
                "pairs": pairs.clone(),
                "coefficients": coefficients.clone(),
                "degraded": degraded.clone(),
                "restored": restored.clone(),
                "target": target.clone(),
                "recoverable_fraction": recoverable_fraction,
                "headroom_sse": headroom,
            }
        if index % 11 == 0 or index == sample_count:
            print(f"training audit [{index:03d}/{sample_count}] {sample['degradation_type']}/{sample['scene_id']}")

    if best_candidate is None:
        raise RuntimeError("No memorization candidate was selected")
    sampled_beta = torch.cat(sampled_betas)
    sampled_coeff = torch.cat(sampled_coefficients)
    full_beta = torch.cat(full_betas)
    full_coeff = torch.cat(full_coefficients)
    return (
        coefficient_metrics(sampled_beta, sampled_coeff, args.fixed_beta),
        coefficient_metrics(full_beta, full_coeff, args.fixed_beta),
        best_candidate,
    )


@torch.inference_mode()
def evaluate_memorizer(
    model: nn.Module,
    pairs: Tensor,
    coefficients: Tensor,
    device: torch.device,
    batch_size: int,
    fixed_beta: float,
) -> dict:
    beta = predict_betas(model, pairs, device, batch_size)
    result = coefficient_metrics(beta, coefficients, fixed_beta)
    result["predicted_beta_mean"] = float(beta.mean())
    result["predicted_beta_near_zero_fraction"] = float((beta <= 0.05).float().mean())
    result["predicted_beta_interior_fraction"] = float(
        ((beta > 0.05) & (beta < 0.95)).float().mean()
    )
    result["predicted_beta_near_one_fraction"] = float((beta >= 0.95).float().mean())
    return result


def memorize_candidate(
    candidate: dict,
    width: int,
    block_size: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[dict, List[dict]]:
    pairs, coefficients = candidate["pairs"], candidate["coefficients"]
    model = BetaController(width).to(device)
    loader = DataLoader(
        ControllerBlocks(pairs, coefficients), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(args.seed),
        num_workers=0, pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_capture, best_state, best_epoch = -math.inf, None, 0
    best_objective, best_objective_state, best_objective_epoch = math.inf, None, 0
    history: List[dict] = []
    for epoch in range(1, args.memorization_epochs + 1):
        model.train()
        for pair_batch, coefficient_batch in loader:
            pair_batch = pair_batch.to(device, non_blocking=True)
            coefficient_batch = coefficient_batch.to(device, non_blocking=True)
            beta = model(pair_batch)
            loss = normalized_regret(beta, coefficient_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        metrics = evaluate_memorizer(
            model, pairs, coefficients, device, args.batch_size, args.fixed_beta
        )
        history.append({
            "epoch": epoch,
            "normalized_regret": metrics["controller_normalized_regret"],
            "capture_fraction": metrics["controller_capture_fraction"],
            "beta_mae": metrics["beta_mae"],
        })
        if metrics["controller_capture_fraction"] > best_capture:
            best_capture = metrics["controller_capture_fraction"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if metrics["controller_normalized_regret"] < best_objective:
            best_objective = metrics["controller_normalized_regret"]
            best_objective_state = copy.deepcopy(model.state_dict())
            best_objective_epoch = epoch
        if epoch == 1 or epoch % 20 == 0:
            print(
                f"memorize {epoch:03d}/{args.memorization_epochs}: "
                f"regret={metrics['controller_normalized_regret']:.6f} "
                f"capture={metrics['controller_capture_fraction']:.4f}"
            )
    if best_state is None or best_objective_state is None:
        raise RuntimeError("Memorization did not produce a checkpoint")

    model.load_state_dict(best_state, strict=True)
    best_capture_metrics = evaluate_memorizer(
        model, pairs, coefficients, device, args.batch_size, args.fixed_beta
    )
    betas = predict_betas(model, pairs, device, args.batch_size)
    output = controller_output(
        candidate["degraded"], candidate["restored"], betas, block_size
    )
    fixed = candidate["degraded"] + args.fixed_beta * (
        candidate["restored"] - candidate["degraded"]
    )
    best_capture_metrics.update({
        "epoch": best_epoch,
        "full_image_psnr": calculate_psnr(output, candidate["target"]),
        "fixed_beta_full_image_psnr": calculate_psnr(fixed, candidate["target"]),
    })

    model.load_state_dict(best_objective_state, strict=True)
    best_objective_metrics = evaluate_memorizer(
        model, pairs, coefficients, device, args.batch_size, args.fixed_beta
    )
    best_objective_metrics["epoch"] = best_objective_epoch
    return {
        "candidate": {
            "scene_id": candidate["scene_id"],
            "type": candidate["type"],
            "block_count": len(pairs),
            "recoverable_fraction_of_fixed_sse": candidate["recoverable_fraction"],
            "oracle_headroom_sse": candidate["headroom_sse"],
        },
        "best_capture_checkpoint": best_capture_metrics,
        "best_objective_checkpoint": best_objective_metrics,
    }, history


def diagnose(sampled: dict, full: dict, memorization: dict) -> dict:
    memory_capture = memorization["best_capture_checkpoint"]["controller_capture_fraction"]
    sampled_objective_better = (
        sampled["controller_normalized_regret"] < sampled["fixed_normalized_regret"]
    )
    if memory_capture < 0.8:
        primary = "local_representation_capacity_or_optimization_failure"
    elif not sampled_objective_better:
        primary = "original_training_optimization_failure"
    elif sampled["controller_capture_fraction"] >= 0.2 and full["controller_capture_fraction"] < 0.2:
        primary = "sampling_or_objective_distribution_mismatch"
    elif full["controller_capture_fraction"] < 0.2:
        primary = "full_training_fit_failure"
    else:
        primary = "training_fit_established_information_probe_allowed"
    return {
        "primary_diagnosis": primary,
        "memorization_pass": memory_capture >= 0.8,
        "exact_sampled_objective_beats_fixed_beta": sampled_objective_better,
        "exact_sampled_capture_at_least_20_percent": sampled["controller_capture_fraction"] >= 0.2,
        "full_training_capture_at_least_20_percent": full["controller_capture_fraction"] >= 0.2,
        "rules": {
            "memorization": "same architecture must capture >=80% of oracle-recoverable SSE on one fixed high-headroom training image",
            "sampled_fit": "frozen checkpoint must beat fixed beta in normalized regret and capture >=20% on exact sampled blocks",
            "full_fit": "frozen checkpoint must capture >=20% on all training blocks",
        },
        "information_probe_allowed": primary == "training_fit_established_information_probe_allowed",
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.blocks_per_image <= 0 or args.memorization_epochs <= 0:
        raise ValueError("blocks-per-image and memorization-epochs must be positive")
    if not 0 <= args.fixed_beta <= 1:
        raise ValueError("fixed-beta must be in [0,1]")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable; enable a Kaggle GPU or pass --allow-cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = bool(args.amp and device.type == "cuda")
    seed_everything(args.seed)
    restorer, _ = load_restorer(args.checkpoint, device)
    controller, payload = load_controller(args.controller, device)
    train_ids = list(payload.get("controller_train_scene_ids", []))
    if not train_ids:
        raise RuntimeError("Controller checkpoint lacks controller_train_scene_ids")
    block_size = int(payload.get("block_size", 32))
    width = int(payload.get("width", 24))
    dataset = CDD11Dataset(
        find_cdd11_root(args.data_root), mode="train", crop_size=0,
        val_fraction=args.val_fraction, split_seed=args.seed, augment=False,
    )
    sampled, full, candidate = collect_training_audit(
        dataset, train_ids, restorer, controller, block_size, device, args
    )
    memorization, history = memorize_candidate(
        candidate, width, block_size, device, args
    )
    diagnosis = diagnose(sampled, full, memorization)
    result = {
        "git_commit": args.git_commit or None,
        "data_scope": "controller_training_scenes_only",
        "calibration_loaded": False,
        "validation_loaded": False,
        "test_loaded": False,
        "exact_sampled_blocks": sampled,
        "all_training_blocks": full,
        "memorization": memorization,
        "diagnosis": diagnosis,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "memorization_log.csv", history)
    protocol = {
        "purpose": "training-only exact-sampling and memorization audit",
        "calibration_loaded": False,
        "validation_loaded": False,
        "test_loaded": False,
        "restorer_checkpoint": str(Path(args.checkpoint).resolve()),
        "controller_checkpoint": str(Path(args.controller).resolve()),
        "controller_train_scene_ids": train_ids,
        "args": vars(args),
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
