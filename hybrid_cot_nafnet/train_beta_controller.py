"""Train a frozen-backbone continuous-beta controller for CDD-11 AiOIR.

The controller predicts one residual strength in [0,1] per 32x32 block from
the degraded input and frozen A3-M output.  It is trained with normalized
reconstruction regret, selected on source-disjoint calibration scenes, and
evaluated once on the existing validation split.  CDD-11_test is never loaded.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .continuous_beta_oracle import continuous_block_oracle
from .datasets import CDD11Dataset, find_cdd11_root
from .evaluate import calculate_psnr, calculate_ssim, oracle_block_output
from .model import count_parameters
from .project_config import KAGGLE_CDD11_ROOT
from .train_gain_predictor import (
    blocks_from_pair,
    choose_scene_split,
    load_restorer,
    restore,
    seed_everything,
)


SINGLE_DEGRADATIONS = {"low", "haze", "rain", "snow"}


class BetaController(nn.Module):
    """Small block CNN producing a continuous residual strength."""

    def __init__(self, width: int = 24) -> None:
        super().__init__()
        self.width = int(width)
        self.features = nn.Sequential(
            nn.Conv2d(12, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1)
        )

    def forward(self, pair: Tensor) -> Tensor:
        if pair.ndim != 4 or pair.shape[1] != 6:
            raise ValueError("Expected [N,6,H,W] input/restoration block pairs")
        degraded, restored = pair[:, :3], pair[:, 3:]
        residual = restored - degraded
        features = torch.cat((degraded, restored, residual, residual.abs()), dim=1)
        return self.head(self.features(features)).squeeze(1).sigmoid()


class ControllerBlocks(Dataset):
    def __init__(self, pairs: Tensor, coefficients: Tensor) -> None:
        if pairs.dtype != torch.float16 or pairs.ndim != 4:
            raise ValueError("pairs must be NCHW float16")
        if coefficients.shape != (len(pairs), 4):
            raise ValueError("coefficients must have shape [N,4]")
        self.pairs = pairs
        self.coefficients = coefficients.float()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.pairs[index].float(), self.coefficients[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Frozen A3-M best.pt")
    parser.add_argument("--data-root", default=str(KAGGLE_CDD11_ROOT))
    parser.add_argument("--output-dir", default="/kaggle/working/beta_controller")
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--blocks-per-image", type=int, default=64)
    parser.add_argument("--calibration-scenes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--fixed-beta", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controller-split-seed", type=int, default=31415)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def block_quadratics(
    degraded: Tensor, restored: Tensor, target: Tensor, block_size: int
) -> Tensor:
    """Return [L, a, b, c, oracle_sse] for SSE(beta)=a*beta^2+b*beta+c."""
    height, width = target.shape[-2:]
    pad_h, pad_w = (-height) % block_size, (-width) % block_size
    residual = restored - degraded
    error = degraded - target
    terms = torch.cat((residual.square(), 2 * error * residual, error.square()), dim=1)
    terms = F.pad(terms, (0, pad_w, 0, pad_h))
    unfolded = F.unfold(terms, kernel_size=block_size, stride=block_size)[0]
    area = 3 * block_size * block_size
    a = unfolded[:area].double().sum(dim=0)
    b = unfolded[area : 2 * area].double().sum(dim=0)
    c = unfolded[2 * area :].double().sum(dim=0)
    beta_star = torch.where(
        a > 1e-20, (-b / (2 * a.clamp_min(1e-20))).clamp(0, 1), torch.zeros_like(a)
    )
    oracle = (a * beta_star.square() + b * beta_star + c).clamp_min(0)
    return torch.stack((a, b, c, oracle), dim=1).float()


def normalized_regret(beta: Tensor, coefficients: Tensor) -> Tensor:
    a, b, c, oracle = coefficients.unbind(dim=1)
    predicted_sse = a * beta.square() + b * beta + c
    restored_sse = (a + b + c).clamp_min(0)
    scale = (c + restored_sse).clamp_min(1e-6)
    return ((predicted_sse - oracle).clamp_min(0) / scale).mean()


def iter_selected(dataset: CDD11Dataset, scene_ids: Iterable[str]):
    selected = set(scene_ids)
    for index, sample in enumerate(dataset.samples):
        if sample[3] in selected:
            yield dataset[index]


@torch.inference_mode()
def build_cache(
    dataset: CDD11Dataset,
    scene_ids: Sequence[str],
    restorer: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    blocks_per_image: int | None,
) -> Tuple[Tensor, Tensor, Tensor, List[int]]:
    """Cache blocks, quadratic targets, image indices and original numels."""
    pairs, coefficients, image_indices, image_numels = [], [], [], []
    generator = torch.Generator().manual_seed(args.controller_split_seed)
    selected_ids = set(scene_ids)
    sample_count = sum(sample[3] in selected_ids for sample in dataset.samples)
    for image_index, sample in enumerate(iter_selected(dataset, scene_ids)):
        degraded, target = sample["lq"].unsqueeze(0), sample["gt"].unsqueeze(0)
        restored = restore(restorer, degraded, device, args.tile, args.overlap, args.amp)
        blocks = blocks_from_pair(degraded, restored, args.block_size)
        quadratic = block_quadratics(degraded, restored, target, args.block_size)
        if blocks_per_image is not None:
            keep = min(blocks_per_image, len(blocks))
            indices = torch.randperm(len(blocks), generator=generator)[:keep]
            blocks, quadratic = blocks[indices], quadratic[indices]
        pairs.append(blocks.clamp(0, 1).half())
        coefficients.append(quadratic)
        image_indices.append(torch.full((len(blocks),), image_index, dtype=torch.long))
        image_numels.append(target.numel())
        if (image_index + 1) % 11 == 0 or image_index + 1 == sample_count:
            print(
                f"cache [{image_index + 1:03d}/{sample_count}] "
                f"blocks={sum(len(item) for item in coefficients)}"
            )
    return (
        torch.cat(pairs), torch.cat(coefficients), torch.cat(image_indices), image_numels
    )


@torch.inference_mode()
def predict_betas(
    controller: nn.Module, pairs: Tensor, device: torch.device, batch_size: int
) -> Tensor:
    predictions = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size].float().to(device, non_blocking=True)
        predictions.append(controller(batch).float().cpu())
    return torch.cat(predictions)


@torch.inference_mode()
def calibration_psnr(
    controller: nn.Module,
    pairs: Tensor,
    coefficients: Tensor,
    image_indices: Tensor,
    image_numels: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> float:
    beta = predict_betas(controller, pairs, device, batch_size)
    a, b, c, _ = coefficients.unbind(dim=1)
    sse = (a * beta.square() + b * beta + c).clamp_min(0)
    image_sse = torch.zeros(len(image_numels), dtype=torch.float64)
    image_sse.scatter_add_(0, image_indices, sse.double())
    psnr = [
        -10 * math.log10(max(value.item() / numel, 1e-12))
        for value, numel in zip(image_sse, image_numels)
    ]
    return float(np.mean(psnr))


def train_controller(
    train_pairs: Tensor,
    train_coefficients: Tensor,
    calibration_cache: Tuple[Tensor, Tensor, Tensor, List[int]],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[BetaController, List[dict], int]:
    controller = BetaController(args.width).to(device)
    loader = DataLoader(
        ControllerBlocks(train_pairs, train_coefficients),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        controller.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_state, best_psnr, best_epoch = None, -math.inf, 0
    history: List[dict] = []
    cal_pairs, cal_coefficients, cal_indices, cal_numels = calibration_cache
    for epoch in range(1, args.epochs + 1):
        controller.train()
        loss_sum, count = 0.0, 0
        for pairs, coefficients in loader:
            pairs = pairs.to(device, non_blocking=True)
            coefficients = coefficients.to(device, non_blocking=True)
            beta = controller(pairs)
            loss = normalized_regret(beta, coefficients)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(pairs)
            count += len(pairs)
        controller.eval()
        cal_psnr = calibration_psnr(
            controller, cal_pairs, cal_coefficients, cal_indices, cal_numels,
            device, args.batch_size,
        )
        row = {
            "epoch": epoch,
            "train_normalized_regret": loss_sum / count,
            "calibration_mean_psnr": cal_psnr,
        }
        history.append(row)
        if cal_psnr > best_psnr:
            best_psnr, best_epoch = cal_psnr, epoch
            best_state = copy.deepcopy(controller.state_dict())
        print(
            f"controller epoch {epoch:02d}/{args.epochs}: "
            f"regret={row['train_normalized_regret']:.6f} "
            f"cal_psnr={cal_psnr:.4f} best={best_psnr:.4f}@{best_epoch}"
        )
    if best_state is None:
        raise RuntimeError("No controller checkpoint was selected")
    controller.load_state_dict(best_state, strict=True)
    return controller.eval(), history, best_epoch


def controller_output(
    degraded: Tensor, restored: Tensor, betas: Tensor, block_size: int
) -> Tensor:
    height, width = degraded.shape[-2:]
    rows, columns = math.ceil(height / block_size), math.ceil(width / block_size)
    if len(betas) != rows * columns:
        raise ValueError("Beta count does not match image block grid")
    beta_map = betas.reshape(1, 1, rows, columns)
    beta_map = beta_map.repeat_interleave(block_size, 2).repeat_interleave(block_size, 3)
    beta_map = beta_map[..., :height, :width]
    return degraded + beta_map * (restored - degraded)


@torch.inference_mode()
def evaluate_validation(
    dataset: CDD11Dataset,
    restorer: nn.Module,
    controller: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    all_betas = []
    for index in range(len(dataset)):
        sample = dataset[index]
        degraded, target = sample["lq"].unsqueeze(0), sample["gt"].unsqueeze(0)
        restored = restore(restorer, degraded, device, args.tile, args.overlap, args.amp)
        pairs = blocks_from_pair(degraded, restored, args.block_size)
        betas = predict_betas(controller, pairs.half(), device, args.batch_size)
        controlled = controller_output(degraded, restored, betas, args.block_size)
        fixed = degraded + args.fixed_beta * (restored - degraded)
        binary, _ = oracle_block_output(degraded, restored, target, args.block_size)
        continuous, _ = continuous_block_oracle(
            degraded, restored, target, args.block_size
        )
        all_betas.append(betas.numpy())
        outputs = {
            "input": degraded,
            "beta1": restored,
            "fixed_beta": fixed,
            "controller": controlled,
            "binary_oracle": binary,
            "continuous_oracle": continuous,
        }
        row = {
            "scene_id": sample["scene_id"],
            "type": sample["degradation_type"],
            "group": (
                "single" if sample["degradation_type"] in SINGLE_DEGRADATIONS
                else "composite"
            ),
            "controller_beta_mean": float(betas.mean()),
            "controller_beta_near_zero_fraction": float((betas <= 0.05).float().mean()),
            "controller_beta_interior_fraction": float(
                ((betas > 0.05) & (betas < 0.95)).float().mean()
            ),
            "controller_beta_near_one_fraction": float((betas >= 0.95).float().mean()),
        }
        for name, output in outputs.items():
            row[f"{name}_psnr"] = calculate_psnr(output, target)
            row[f"{name}_ssim"] = calculate_ssim(output, target)
        rows.append(row)
        print(
            f"validation [{index + 1:03d}/{len(dataset)}] "
            f"{row['type']}/{row['scene_id']} fixed={row['fixed_beta_psnr']:.3f} "
            f"controller={row['controller_psnr']:.3f} "
            f"oracle={row['continuous_oracle_psnr']:.3f}"
        )

    method_names = (
        "input", "beta1", "fixed_beta", "controller", "binary_oracle",
        "continuous_oracle",
    )
    methods = {
        name: {
            "mean_psnr": float(np.mean([row[f"{name}_psnr"] for row in rows])),
            "mean_ssim": float(np.mean([row[f"{name}_ssim"] for row in rows])),
            "images_worse_than_input": int(sum(
                row[f"{name}_psnr"] < row["input_psnr"] - 1e-12 for row in rows
            )),
        }
        for name in method_names
    }
    per_type = {}
    for degradation_type in sorted({row["type"] for row in rows}):
        items = [row for row in rows if row["type"] == degradation_type]
        per_type[degradation_type] = {
            "fixed_beta_psnr": float(np.mean([row["fixed_beta_psnr"] for row in items])),
            "controller_psnr": float(np.mean([row["controller_psnr"] for row in items])),
            "gain_over_fixed_beta_db": float(np.mean([
                row["controller_psnr"] - row["fixed_beta_psnr"] for row in items
            ])),
        }
    gain_over_fixed = (
        methods["controller"]["mean_psnr"] - methods["fixed_beta"]["mean_psnr"]
    )
    headroom = (
        methods["continuous_oracle"]["mean_psnr"]
        - methods["fixed_beta"]["mean_psnr"]
    )
    captured = gain_over_fixed / headroom if headroom > 0 else None
    type_wins = sum(item["gain_over_fixed_beta_db"] > 0 for item in per_type.values())
    gates = {
        "gain_over_fixed_beta_at_least_0_2_db": gain_over_fixed >= 0.2,
        "no_mean_ssim_drop_vs_fixed_beta": (
            methods["controller"]["mean_ssim"] >= methods["fixed_beta"]["mean_ssim"]
        ),
        "fewer_harmed_images_than_fixed_beta": (
            methods["controller"]["images_worse_than_input"]
            < methods["fixed_beta"]["images_worse_than_input"]
        ),
        "captures_at_least_20_percent_of_continuous_headroom": (
            captured is not None and captured >= 0.2
        ),
        "beats_fixed_beta_on_at_least_7_of_11_types": type_wins >= 7,
    }
    beta_values = np.concatenate(all_betas)
    summary = {
        "split": "validation",
        "count": len(rows),
        "block_size": args.block_size,
        "fixed_beta": args.fixed_beta,
        "methods": methods,
        "gain_over_fixed_beta_db": gain_over_fixed,
        "continuous_oracle_headroom_over_fixed_beta_db": headroom,
        "continuous_oracle_headroom_captured_fraction": captured,
        "controller_beta_distribution": {
            "mean": float(beta_values.mean()),
            "near_zero_fraction": float(np.mean(beta_values <= 0.05)),
            "interior_fraction": float(np.mean(
                (beta_values > 0.05) & (beta_values < 0.95)
            )),
            "near_one_fraction": float(np.mean(beta_values >= 0.95)),
        },
        "per_type": per_type,
        "decision": {
            "status": "pass" if all(gates.values()) else "fail",
            "gates": gates,
            "types_beating_fixed_beta": type_wins,
            "rule": "All prespecified validation gates must pass before any test run.",
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
    if args.block_size <= 0 or args.blocks_per_image <= 0:
        raise ValueError("block-size and blocks-per-image must be positive")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if not 0 <= args.fixed_beta <= 1:
        raise ValueError("fixed-beta must be in [0,1]")
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
    validation = CDD11Dataset(
        root, mode="val", crop_size=0, val_fraction=args.val_fraction,
        split_seed=args.seed, augment=False,
    )
    train_ids, calibration_ids = choose_scene_split(
        train_full.scene_ids, args.calibration_scenes, args.controller_split_seed
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol": "15 controller-train scenes -> 5 calibration scenes -> 5 validation scenes; test untouched",
        "git_commit": args.git_commit or None,
        "restorer_checkpoint": str(Path(args.checkpoint).resolve()),
        "restorer_epoch_index": restorer_checkpoint.get("epoch"),
        "controller_train_scene_ids": train_ids,
        "calibration_scene_ids": calibration_ids,
        "validation_scene_ids": list(validation.scene_ids),
        "target": "normalized reconstruction regret for continuous block beta",
        "args": vars(args),
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in protocol.items() if key != "args"}, indent=2))

    train_pairs, train_coefficients, _, _ = build_cache(
        train_full, train_ids, restorer, device, args, args.blocks_per_image
    )
    calibration_cache = build_cache(
        train_full, calibration_ids, restorer, device, args, None
    )
    print(
        f"train blocks={len(train_pairs)} calibration blocks={len(calibration_cache[0])} "
        f"cache_mib={(train_pairs.numel() * train_pairs.element_size() + calibration_cache[0].numel() * calibration_cache[0].element_size()) / 2**20:.1f}"
    )
    controller, history, best_epoch = train_controller(
        train_pairs, train_coefficients, calibration_cache, device, args
    )
    checkpoint = {
        "format_version": 1,
        "model": controller.state_dict(),
        "width": args.width,
        "block_size": args.block_size,
        "best_epoch": best_epoch,
        "loss": "normalized reconstruction regret",
        "controller_train_scene_ids": train_ids,
        "calibration_scene_ids": calibration_ids,
    }
    torch.save(checkpoint, output_dir / "beta_controller.pt")
    write_csv(output_dir / "train_log.csv", history)
    del train_pairs, train_coefficients, calibration_cache

    rows, summary = evaluate_validation(
        validation, restorer, controller, device, args
    )
    summary["best_epoch"] = best_epoch
    summary["parameter_counts"] = {
        "restorer": count_parameters(restorer)["total"],
        "controller": sum(parameter.numel() for parameter in controller.parameters()),
    }
    write_csv(output_dir / "validation_metrics.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
