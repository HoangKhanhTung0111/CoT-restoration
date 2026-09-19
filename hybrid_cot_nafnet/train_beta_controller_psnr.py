"""Train the unchanged beta controller with a mean-PSNR-aligned objective.

All blocks from each controller-training image contribute jointly to one
image-level log-MSE loss.  The restorer and controller architecture stay fixed.
Validation is loaded only if the calibration-selected checkpoint captures the
prespecified fraction of oracle headroom on its training scenes.  Test is never
loaded.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from .datasets import CDD11Dataset, find_cdd11_root
from .model import count_parameters
from .project_config import KAGGLE_CDD11_ROOT
from .train_beta_controller import (
    BetaController,
    build_cache,
    calibration_psnr,
    evaluate_validation,
    predict_betas,
    write_csv,
)
from .train_gain_predictor import choose_scene_split, load_restorer, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Frozen A3-M best.pt")
    parser.add_argument("--data-root", default=str(KAGGLE_CDD11_ROOT))
    parser.add_argument("--output-dir", default="/kaggle/working/beta_controller_psnr")
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--calibration-scenes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--fixed-beta", type=float, default=0.99)
    parser.add_argument("--train-headroom-gate", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controller-split-seed", type=int, default=31415)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def image_slices(image_indices: Tensor, image_count: int) -> List[slice]:
    counts = torch.bincount(image_indices, minlength=image_count).tolist()
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + int(count))
    if offsets[-1] != len(image_indices):
        raise RuntimeError("Image index counts do not cover the block cache")
    for image_index, block_slice in enumerate(
        slice(offsets[i], offsets[i + 1]) for i in range(image_count)
    ):
        values = image_indices[block_slice]
        if len(values) == 0 or not torch.all(values == image_index):
            raise RuntimeError("Expected image blocks to be contiguous and non-empty")
    return [slice(offsets[i], offsets[i + 1]) for i in range(image_count)]


def forward_betas(
    controller: nn.Module,
    pairs: Tensor,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    outputs = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size].float().to(
            device, non_blocking=True
        )
        outputs.append(controller(batch))
    return torch.cat(outputs)


def image_log_mse_ratio(
    beta: Tensor, coefficients: Tensor, fixed_beta: float
) -> Tensor:
    """PSNR-aligned loss; minimizing log(SSE/fixed-SSE) maximizes PSNR."""
    a, b, c, _ = coefficients.unbind(dim=1)
    predicted_sse = (a * beta.square() + b * beta + c).sum().clamp_min(1e-12)
    fixed_sse = (
        a * fixed_beta**2 + b * fixed_beta + c
    ).sum().clamp_min(1e-12)
    return torch.log(predicted_sse / fixed_sse)


@torch.inference_mode()
def cache_metrics(
    controller: nn.Module,
    pairs: Tensor,
    coefficients: Tensor,
    image_indices: Tensor,
    image_numels: Sequence[int],
    device: torch.device,
    batch_size: int,
    fixed_beta: float,
) -> Dict[str, float]:
    beta = predict_betas(controller, pairs, device, batch_size)
    a, b, c, oracle = coefficients.unbind(dim=1)
    controller_block_sse = (a * beta.square() + b * beta + c).clamp_min(0)
    fixed_block_sse = (a * fixed_beta**2 + b * fixed_beta + c).clamp_min(0)
    image_count = len(image_numels)
    controller_sse = torch.zeros(image_count, dtype=torch.float64)
    fixed_sse = torch.zeros(image_count, dtype=torch.float64)
    oracle_sse = torch.zeros(image_count, dtype=torch.float64)
    controller_sse.scatter_add_(0, image_indices, controller_block_sse.double())
    fixed_sse.scatter_add_(0, image_indices, fixed_block_sse.double())
    oracle_sse.scatter_add_(0, image_indices, oracle.double())
    numels = torch.tensor(image_numels, dtype=torch.float64)

    def mean_psnr(sse: Tensor) -> float:
        return float((-10 * torch.log10((sse / numels).clamp_min(1e-12))).mean())

    fixed_psnr = mean_psnr(fixed_sse)
    controller_psnr = mean_psnr(controller_sse)
    oracle_psnr = mean_psnr(oracle_sse)
    psnr_headroom = oracle_psnr - fixed_psnr
    psnr_gain = controller_psnr - fixed_psnr
    aggregate_headroom = float((fixed_sse - oracle_sse).sum())
    aggregate_recovered = float((fixed_sse - controller_sse).sum())
    return {
        "image_count": image_count,
        "block_count": len(pairs),
        "fixed_beta_mean_psnr": fixed_psnr,
        "controller_mean_psnr": controller_psnr,
        "oracle_mean_psnr": oracle_psnr,
        "gain_over_fixed_beta_db": psnr_gain,
        "oracle_headroom_over_fixed_beta_db": psnr_headroom,
        "mean_psnr_headroom_captured_fraction": (
            psnr_gain / psnr_headroom if psnr_headroom > 0 else 0.0
        ),
        "aggregate_sse_headroom_captured_fraction": (
            aggregate_recovered / aggregate_headroom
            if aggregate_headroom > 0 else 0.0
        ),
        "images_beating_fixed_beta": int((controller_sse < fixed_sse).sum()),
        "predicted_beta_mean": float(beta.mean()),
        "predicted_beta_near_zero_fraction": float((beta <= 0.05).float().mean()),
        "predicted_beta_interior_fraction": float(
            ((beta > 0.05) & (beta < 0.95)).float().mean()
        ),
        "predicted_beta_near_one_fraction": float((beta >= 0.95).float().mean()),
    }


def train_controller(
    train_cache: Tuple[Tensor, Tensor, Tensor, List[int]],
    calibration_cache: Tuple[Tensor, Tensor, Tensor, List[int]],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[BetaController, List[dict], int, dict]:
    train_pairs, train_coefficients, train_indices, train_numels = train_cache
    cal_pairs, cal_coefficients, cal_indices, cal_numels = calibration_cache
    slices = image_slices(train_indices, len(train_numels))
    controller = BetaController(args.width).to(device)
    optimizer = torch.optim.AdamW(
        controller.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed)
    best_state, best_calibration_psnr, best_epoch = None, -math.inf, 0
    best_train_state, best_train_capture = None, -math.inf
    history: List[dict] = []

    for epoch in range(1, args.epochs + 1):
        controller.train()
        losses = []
        for image_index in torch.randperm(len(slices), generator=generator).tolist():
            block_slice = slices[image_index]
            coefficients = train_coefficients[block_slice].to(
                device, non_blocking=True
            )
            beta = forward_betas(
                controller, train_pairs[block_slice], device, args.batch_size
            )
            loss = image_log_mse_ratio(beta, coefficients, args.fixed_beta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        controller.eval()
        train_metrics = cache_metrics(
            controller, *train_cache, device, args.batch_size, args.fixed_beta
        )
        cal_psnr = calibration_psnr(
            controller, cal_pairs, cal_coefficients, cal_indices, cal_numels,
            device, args.batch_size,
        )
        row = {
            "epoch": epoch,
            "train_mean_log_sse_ratio": float(np.mean(losses)),
            "train_controller_mean_psnr": train_metrics["controller_mean_psnr"],
            "train_gain_over_fixed_beta_db": train_metrics["gain_over_fixed_beta_db"],
            "train_mean_psnr_headroom_captured_fraction": train_metrics[
                "mean_psnr_headroom_captured_fraction"
            ],
            "train_aggregate_sse_headroom_captured_fraction": train_metrics[
                "aggregate_sse_headroom_captured_fraction"
            ],
            "calibration_mean_psnr": cal_psnr,
        }
        history.append(row)
        train_capture = train_metrics["mean_psnr_headroom_captured_fraction"]
        if train_capture > best_train_capture:
            best_train_capture = train_capture
            best_train_state = copy.deepcopy(controller.state_dict())
        if cal_psnr > best_calibration_psnr:
            best_calibration_psnr = cal_psnr
            best_state = copy.deepcopy(controller.state_dict())
            best_epoch = epoch
        print(
            f"controller epoch {epoch:02d}/{args.epochs}: "
            f"log_ratio={row['train_mean_log_sse_ratio']:.6f} "
            f"train_gain={row['train_gain_over_fixed_beta_db']:+.4f}dB "
            f"capture={train_capture:.4f} "
            f"cal_psnr={cal_psnr:.4f} best={best_calibration_psnr:.4f}@{best_epoch}"
        )

    if best_state is None or best_train_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    controller.load_state_dict(best_train_state, strict=True)
    best_train_metrics = cache_metrics(
        controller, *train_cache, device, args.batch_size, args.fixed_beta
    )
    controller.load_state_dict(best_state, strict=True)
    return controller.eval(), history, best_epoch, best_train_metrics


def main() -> None:
    args = parse_args()
    if args.block_size <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("block-size, epochs, and batch-size must be positive")
    if not 0 <= args.fixed_beta <= 1:
        raise ValueError("fixed-beta must be in [0,1]")
    if not 0 <= args.train_headroom_gate <= 1:
        raise ValueError("train-headroom-gate must be in [0,1]")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable; enable a Kaggle GPU or pass --allow-cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = bool(args.amp and device.type == "cuda")
    seed_everything(args.seed)
    root = find_cdd11_root(args.data_root)
    restorer, restorer_checkpoint = load_restorer(args.checkpoint, device)
    train_full = CDD11Dataset(
        root, mode="train", crop_size=0, val_fraction=args.val_fraction,
        split_seed=args.seed, augment=False,
    )
    train_ids, calibration_ids = choose_scene_split(
        train_full.scene_ids, args.calibration_scenes, args.controller_split_seed
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol": "15 all-block controller-training scenes -> 5 calibration scenes -> gated 5-scene validation; test untouched",
        "git_commit": args.git_commit or None,
        "restorer_checkpoint": str(Path(args.checkpoint).resolve()),
        "restorer_epoch_index": restorer_checkpoint.get("epoch"),
        "controller_train_scene_ids": train_ids,
        "calibration_scene_ids": calibration_ids,
        "objective": "equal-image mean log(SSE/fixed-SSE), equivalent to maximizing mean image PSNR",
        "validation_policy": "load validation only if calibration-selected checkpoint captures >=20% of mean-PSNR oracle headroom on controller-training scenes",
        "validation_loaded": False,
        "test_loaded": False,
        "args": vars(args),
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in protocol.items() if key != "args"}, indent=2))

    train_cache = build_cache(
        train_full, train_ids, restorer, device, args, None
    )
    calibration_cache = build_cache(
        train_full, calibration_ids, restorer, device, args, None
    )
    cache_bytes = train_cache[0].numel() * train_cache[0].element_size()
    cache_bytes += calibration_cache[0].numel() * calibration_cache[0].element_size()
    print(
        f"all-block cache: train={len(train_cache[0])} "
        f"calibration={len(calibration_cache[0])} mib={cache_bytes / 2**20:.1f}"
    )
    controller, history, best_epoch, best_train_checkpoint_metrics = train_controller(
        train_cache, calibration_cache, device, args
    )
    selected_train_metrics = cache_metrics(
        controller, *train_cache, device, args.batch_size, args.fixed_beta
    )
    training_gate_passed = (
        selected_train_metrics["mean_psnr_headroom_captured_fraction"]
        >= args.train_headroom_gate
    )
    checkpoint = {
        "format_version": 1,
        "model": controller.state_dict(),
        "width": args.width,
        "block_size": args.block_size,
        "best_epoch": best_epoch,
        "loss": "equal-image mean log(SSE/fixed-SSE)",
        "all_training_blocks": True,
        "controller_train_scene_ids": train_ids,
        "calibration_scene_ids": calibration_ids,
    }
    torch.save(checkpoint, output_dir / "beta_controller_psnr.pt")
    write_csv(output_dir / "train_log.csv", history)

    summary = {
        "best_calibration_epoch": best_epoch,
        "selected_checkpoint_training_metrics": selected_train_metrics,
        "best_training_capture_checkpoint_metrics": best_train_checkpoint_metrics,
        "training_gate": {
            "status": "pass" if training_gate_passed else "fail",
            "threshold": args.train_headroom_gate,
            "metric": "mean_psnr_headroom_captured_fraction",
            "rule": "Validation is loaded only when the calibration-selected checkpoint captures at least 20% of training mean-PSNR oracle headroom.",
        },
        "validation": {
            "status": "not_run",
            "reason": "training gate failed" if not training_gate_passed else None,
        },
        "parameter_counts": {
            "restorer": count_parameters(restorer)["total"],
            "controller": sum(parameter.numel() for parameter in controller.parameters()),
        },
        "test_loaded": False,
    }

    if training_gate_passed:
        validation = CDD11Dataset(
            root, mode="val", crop_size=0, val_fraction=args.val_fraction,
            split_seed=args.seed, augment=False,
        )
        validation_rows, validation_summary = evaluate_validation(
            validation, restorer, controller, device, args
        )
        write_csv(output_dir / "validation_metrics.csv", validation_rows)
        summary["validation"] = validation_summary
        protocol["validation_loaded"] = True
        protocol["validation_scene_ids"] = list(validation.scene_ids)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
