"""Train and validate a local intervention-utility predictor for frozen A3-M.

The held-out CDD-11 test split is deliberately unsupported.  The predictor is
trained on a source-disjoint subset of the restoration training scenes, its
decision threshold is selected on separate calibration scenes, and the final
report is produced on the existing scene-level validation split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .datasets import CDD11Dataset, find_cdd11_root
from .evaluate import (
    calculate_psnr,
    calculate_ssim,
    load_torch_file,
    oracle_block_output,
    tiled_inference,
)
from .model import build_model, count_parameters
from .project_config import KAGGLE_CDD11_ROOT


class LocalGainPredictor(nn.Module):
    """Small block CNN that predicts whether the frozen residual is useful."""

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
        if pair.shape[1] != 6:
            raise ValueError("Expected concatenated input/restoration with 6 channels")
        degraded, restored = pair[:, :3], pair[:, 3:]
        residual = restored - degraded
        features = torch.cat((degraded, restored, residual, residual.abs()), dim=1)
        return self.head(self.features(features)).squeeze(1)


class CachedBlocks(Dataset):
    def __init__(self, pairs_float16: Tensor, targets: Tensor) -> None:
        if pairs_float16.dtype != torch.float16 or pairs_float16.ndim != 4:
            raise ValueError("pairs_float16 must be an NCHW float16 tensor")
        self.pairs = pairs_float16
        self.targets = targets.float()

    def __len__(self) -> int:
        return self.targets.numel()

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.pairs[index].float(), self.targets[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Frozen A3-M best.pt")
    parser.add_argument("--data-root", default=str(KAGGLE_CDD11_ROOT))
    parser.add_argument("--output-dir", default="/kaggle/working/gain_predictor")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--blocks-per-image", type=int, default=64)
    parser.add_argument("--calibration-scenes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gain-split-seed", type=int, default=31415)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_restorer(path: str | Path, device: torch.device) -> Tuple[nn.Module, dict]:
    checkpoint = load_torch_file(path)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError("Expected a train_kaggle.py checkpoint containing 'model'")
    model = build_model(
        checkpoint.get("model_type", "hybrid"),
        checkpoint.get("preset", "nafnet32"),
        int(checkpoint.get("adapter_hidden", 64)),
        use_skip_gates=bool(checkpoint.get("use_skip_gates", True)),
        use_multiscale_degradation=bool(
            checkpoint.get("use_multiscale_degradation", False)
        ),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def blocks_from_pair(degraded: Tensor, restored: Tensor, block_size: int) -> Tensor:
    """Return padded [L,6,B,B] blocks from two [1,3,H,W] tensors."""
    height, width = degraded.shape[-2:]
    pad_h, pad_w = (-height) % block_size, (-width) % block_size
    pair = torch.cat((degraded, restored), dim=1)
    pair = F.pad(pair, (0, pad_w, 0, pad_h), mode="replicate")
    blocks = F.unfold(pair, kernel_size=block_size, stride=block_size)
    return blocks.transpose(1, 2).reshape(-1, 6, block_size, block_size)


def block_errors(
    degraded: Tensor, restored: Tensor, target: Tensor, block_size: int
) -> Tuple[Tensor, Tensor]:
    """Return input/restoration SSE for every block; padded error is zero."""
    height, width = target.shape[-2:]
    pad_h, pad_w = (-height) % block_size, (-width) % block_size
    input_error = (degraded - target).square()
    output_error = (restored - target).square()
    errors = torch.cat((input_error, output_error), dim=1)
    errors = F.pad(errors, (0, pad_w, 0, pad_h))
    # First three channels were flattened before the final three channels.
    channel_area = block_size * block_size
    unfolded = F.unfold(errors, kernel_size=block_size, stride=block_size)[0]
    input_sse = unfolded[: 3 * channel_area].sum(dim=0)
    output_sse = unfolded[3 * channel_area :].sum(dim=0)
    return input_sse, output_sse


def normalized_gain(input_sse: Tensor, output_sse: Tensor) -> Tensor:
    return (input_sse - output_sse) / (input_sse + output_sse).clamp_min(1e-8)


@torch.inference_mode()
def restore(
    model: nn.Module,
    degraded: Tensor,
    device: torch.device,
    tile: int,
    overlap: int,
    use_amp: bool,
) -> Tensor:
    prediction, _ = tiled_inference(model, degraded, device, tile, overlap, use_amp)
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Frozen restorer produced non-finite output")
    return prediction.clamp(0, 1)


@torch.inference_mode()
def predict_scores(
    predictor: nn.Module, blocks: Tensor, device: torch.device, batch_size: int
) -> Tensor:
    scores = []
    for start in range(0, len(blocks), batch_size):
        batch = blocks[start : start + batch_size].to(device, non_blocking=True)
        scores.append(predictor(batch).float().cpu())
    return torch.cat(scores)


def choose_scene_split(
    scene_ids: Sequence[str], calibration_count: int, seed: int
) -> Tuple[List[str], List[str]]:
    ids = sorted(scene_ids)
    random.Random(seed).shuffle(ids)
    if calibration_count <= 0 or calibration_count >= len(ids):
        raise ValueError("calibration-scenes must leave at least one gain-training scene")
    calibration = sorted(ids[:calibration_count])
    training = sorted(ids[calibration_count:])
    return training, calibration


def iter_selected(dataset: CDD11Dataset, scene_ids: Iterable[str]):
    selected = set(scene_ids)
    for index, sample in enumerate(dataset.samples):
        if sample[3] in selected:
            yield index, dataset[index]


def make_training_cache(
    dataset: CDD11Dataset,
    scene_ids: Sequence[str],
    restorer: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[Tensor, Tensor]:
    pair_parts, target_parts = [], []
    generator = torch.Generator().manual_seed(args.gain_split_seed)
    selected_count = sum(sample[3] in set(scene_ids) for sample in dataset.samples)
    for position, (_, sample) in enumerate(iter_selected(dataset, scene_ids), start=1):
        degraded, target = sample["lq"].unsqueeze(0), sample["gt"].unsqueeze(0)
        restored = restore(
            restorer, degraded, device, args.tile, args.overlap, args.amp
        )
        blocks = blocks_from_pair(degraded, restored, args.block_size)
        input_sse, output_sse = block_errors(
            degraded, restored, target, args.block_size
        )
        keep = min(args.blocks_per_image, len(blocks))
        indices = torch.randperm(len(blocks), generator=generator)[:keep]
        pair_parts.append(blocks[indices].clamp(0, 1).to(torch.float16))
        target_parts.append(normalized_gain(input_sse, output_sse)[indices])
        if position % 25 == 0 or position == selected_count:
            print(f"cache [{position:03d}/{selected_count}] blocks={sum(len(x) for x in target_parts)}")
    return torch.cat(pair_parts), torch.cat(target_parts).float()


def train_predictor(
    pairs: Tensor, targets: Tensor, device: torch.device, args: argparse.Namespace
) -> Tuple[LocalGainPredictor, List[Dict[str, float]]]:
    predictor = LocalGainPredictor(args.width).to(device)
    positive = targets > 0
    counts = torch.bincount(positive.long(), minlength=2).float().clamp_min(1)
    sample_weights = torch.where(positive, 0.5 / counts[1], 0.5 / counts[0])
    sampler = WeightedRandomSampler(
        sample_weights.double(), num_samples=len(targets), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        CachedBlocks(pairs, targets), batch_size=args.batch_size, sampler=sampler,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        predictor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        predictor.train()
        total_loss, total_count = 0.0, 0
        for pair, target in loader:
            pair = pair.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prediction = predictor(pair)
            loss = F.smooth_l1_loss(prediction, target, beta=0.1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)
            total_count += len(target)
        row = {"epoch": float(epoch), "loss": total_loss / total_count}
        history.append(row)
        print(f"gain epoch {epoch:02d}/{args.epochs}: loss={row['loss']:.6f}")
    return predictor.eval(), history


def apply_block_policy(
    degraded: Tensor, restored: Tensor, scores: Tensor, threshold: float, block_size: int
) -> Tuple[Tensor, float]:
    height, width = degraded.shape[-2:]
    rows = math.ceil(height / block_size)
    cols = math.ceil(width / block_size)
    use_restored = (scores >= threshold).reshape(1, 1, rows, cols)
    mask = use_restored.repeat_interleave(block_size, 2).repeat_interleave(block_size, 3)
    mask = mask[..., :height, :width]
    return torch.where(mask, restored, degraded), float(mask.float().mean())


@torch.inference_mode()
def collect_policy_records(
    dataset: CDD11Dataset,
    scene_ids: Sequence[str],
    restorer: nn.Module,
    predictor: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> List[dict]:
    records = []
    selected_count = sum(sample[3] in set(scene_ids) for sample in dataset.samples)
    for position, (_, sample) in enumerate(iter_selected(dataset, scene_ids), start=1):
        degraded, target = sample["lq"].unsqueeze(0), sample["gt"].unsqueeze(0)
        restored = restore(restorer, degraded, device, args.tile, args.overlap, args.amp)
        blocks = blocks_from_pair(degraded, restored, args.block_size)
        scores = predict_scores(predictor, blocks, device, args.batch_size)
        input_sse, output_sse = block_errors(degraded, restored, target, args.block_size)
        residual, base_error = restored - degraded, degraded - target
        records.append({
            "scene_id": sample["scene_id"],
            "type": sample["degradation_type"],
            "scores": scores.numpy(),
            "input_sse": input_sse.numpy(),
            "output_sse": output_sse.numpy(),
            "numel": target.numel(),
            "beta_a": residual.square().mean().item(),
            "beta_b": (2 * base_error * residual).mean().item(),
            "beta_c": base_error.square().mean().item(),
        })
        if position % 11 == 0 or position == selected_count:
            print(f"calibration [{position:03d}/{selected_count}]")
    return records


def calibrate_policy(records: Sequence[dict]) -> Tuple[float, float, dict]:
    all_scores = np.concatenate([record["scores"] for record in records])
    quantiles = np.linspace(0, 1, 101)
    thresholds = np.unique(np.quantile(all_scores, quantiles))
    thresholds = np.concatenate(([np.nextafter(all_scores.min(), -np.inf)], thresholds,
                                 [np.nextafter(all_scores.max(), np.inf)]))
    threshold_scores = []
    for threshold in thresholds:
        image_psnr = []
        for record in records:
            sse = np.where(
                record["scores"] >= threshold,
                record["output_sse"], record["input_sse"],
            ).sum()
            image_psnr.append(-10 * math.log10(max(sse / record["numel"], 1e-12)))
        threshold_scores.append(float(np.mean(image_psnr)))
    threshold = float(thresholds[int(np.argmax(threshold_scores))])

    betas = np.linspace(0, 1, 101)
    beta_scores = []
    for beta in betas:
        beta_scores.append(float(np.mean([
            -10 * math.log10(max(
                record["beta_a"] * beta * beta + record["beta_b"] * beta
                + record["beta_c"], 1e-12
            )) for record in records
        ])))
    beta = float(betas[int(np.argmax(beta_scores))])
    return threshold, beta, {
        "selected_threshold": threshold,
        "selected_threshold_mean_psnr": max(threshold_scores),
        "selected_fixed_beta": beta,
        "selected_fixed_beta_mean_psnr": max(beta_scores),
        "candidate_thresholds": int(len(thresholds)),
    }


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(values), dtype=np.float64)
        result[order] = np.arange(len(values), dtype=np.float64)
        return result
    x_rank, y_rank = ranks(x), ranks(y)
    if x_rank.std() == 0 or y_rank.std() == 0:
        return 0.0
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


@torch.inference_mode()
def evaluate_validation(
    dataset: CDD11Dataset,
    restorer: nn.Module,
    predictor: nn.Module,
    threshold: float,
    beta: float,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[List[dict], dict]:
    rows, all_scores, all_gains = [], [], []
    for index in range(len(dataset)):
        sample = dataset[index]
        degraded, target = sample["lq"].unsqueeze(0), sample["gt"].unsqueeze(0)
        started = time.perf_counter()
        restored = restore(restorer, degraded, device, args.tile, args.overlap, args.amp)
        blocks = blocks_from_pair(degraded, restored, args.block_size)
        scores = predict_scores(predictor, blocks, device, args.batch_size)
        gated, restored_fraction = apply_block_policy(
            degraded, restored, scores, threshold, args.block_size
        )
        blended = (degraded + beta * (restored - degraded)).clamp(0, 1)
        oracle, oracle_fraction = oracle_block_output(
            degraded, restored, target, args.block_size
        )
        input_sse, output_sse = block_errors(degraded, restored, target, args.block_size)
        gains = normalized_gain(input_sse, output_sse).numpy()
        all_scores.append(scores.numpy())
        all_gains.append(gains)
        row = {
            "scene_id": sample["scene_id"], "type": sample["degradation_type"],
            "input_psnr": calculate_psnr(degraded, target),
            "input_ssim": calculate_ssim(degraded, target),
            "restored_psnr": calculate_psnr(restored, target),
            "restored_ssim": calculate_ssim(restored, target),
            "fixed_beta_psnr": calculate_psnr(blended, target),
            "fixed_beta_ssim": calculate_ssim(blended, target),
            "gain_policy_psnr": calculate_psnr(gated, target),
            "gain_policy_ssim": calculate_ssim(gated, target),
            "oracle_psnr": calculate_psnr(oracle, target),
            "oracle_ssim": calculate_ssim(oracle, target),
            "gain_restored_fraction": restored_fraction,
            "oracle_restored_fraction": oracle_fraction,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
        rows.append(row)
        print(
            f"validation [{index + 1:03d}/{len(dataset)}] {row['type']}/{row['scene_id']} "
            f"restored={row['restored_psnr']:.3f} beta={row['fixed_beta_psnr']:.3f} "
            f"gain={row['gain_policy_psnr']:.3f} oracle={row['oracle_psnr']:.3f}"
        )

    scores, gains = np.concatenate(all_scores), np.concatenate(all_gains)
    predicted, helpful = scores >= threshold, gains > 0
    tp = int(np.count_nonzero(predicted & helpful))
    fp = int(np.count_nonzero(predicted & ~helpful))
    fn = int(np.count_nonzero(~predicted & helpful))
    methods = {}
    for name in ("input", "restored", "fixed_beta", "gain_policy", "oracle"):
        psnr = float(np.mean([row[f"{name}_psnr"] for row in rows]))
        ssim = float(np.mean([row[f"{name}_ssim"] for row in rows]))
        methods[name] = {
            "mean_psnr": psnr, "mean_ssim": ssim,
            "images_worse_than_input": int(sum(
                row[f"{name}_psnr"] < row["input_psnr"] - 1e-12 for row in rows
            )),
        }
    restored_psnr = methods["restored"]["mean_psnr"]
    oracle_headroom = methods["oracle"]["mean_psnr"] - restored_psnr
    summary = {
        "split": "validation",
        "count": len(rows),
        "block_size": args.block_size,
        "threshold": threshold,
        "fixed_beta": beta,
        "methods": methods,
        "gain_over_restored_db": methods["gain_policy"]["mean_psnr"] - restored_psnr,
        "gain_over_fixed_beta_db": (
            methods["gain_policy"]["mean_psnr"] - methods["fixed_beta"]["mean_psnr"]
        ),
        "oracle_headroom_over_restored_db": oracle_headroom,
        "oracle_headroom_captured_fraction": (
            (methods["gain_policy"]["mean_psnr"] - restored_psnr) / oracle_headroom
            if oracle_headroom > 0 else None
        ),
        "mean_gain_restored_fraction": float(np.mean([
            row["gain_restored_fraction"] for row in rows
        ])),
        "block_sign_precision": tp / max(1, tp + fp),
        "block_sign_recall": tp / max(1, tp + fn),
        "block_rank_correlation": rank_correlation(scores, gains),
    }
    per_type: Dict[str, dict] = {}
    for degradation_type in sorted({row["type"] for row in rows}):
        items = [row for row in rows if row["type"] == degradation_type]
        per_type[degradation_type] = {
            f"{name}_{metric}": float(np.mean([row[f"{name}_{metric}"] for row in items]))
            for name in ("input", "restored", "fixed_beta", "gain_policy", "oracle")
            for metric in ("psnr", "ssim")
        }
    summary["per_type"] = per_type
    positive_types = sum(
        values["gain_policy_psnr"] > values["fixed_beta_psnr"]
        for values in per_type.values()
    )
    capture = summary["oracle_headroom_captured_fraction"]
    gates = {
        "gain_over_fixed_beta_at_least_0_2_db": (
            summary["gain_over_fixed_beta_db"] >= 0.2
        ),
        "no_mean_ssim_drop_vs_fixed_beta": (
            methods["gain_policy"]["mean_ssim"] >= methods["fixed_beta"]["mean_ssim"]
        ),
        "fewer_harmed_images_than_restoration": (
            methods["gain_policy"]["images_worse_than_input"]
            < methods["restored"]["images_worse_than_input"]
        ),
        "captures_at_least_20_percent_of_oracle_headroom": (
            capture is not None and capture >= 0.2
        ),
        "beats_fixed_beta_on_at_least_7_of_11_types": positive_types >= 7,
    }
    summary["decision"] = {
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "types_beating_fixed_beta": positive_types,
        "rule": "All prespecified gates must pass; a failed scientific result is not rerun unchanged.",
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
    if args.tile > 0 and args.tile % 16:
        raise ValueError("tile must be divisible by 16")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable; enable a Kaggle GPU or pass --allow-cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = bool(args.amp and device.type == "cuda")
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = find_cdd11_root(args.data_root)
    restorer, checkpoint = load_restorer(args.checkpoint, device)
    train_full = CDD11Dataset(
        root, mode="train", crop_size=0, val_fraction=args.val_fraction,
        split_seed=args.seed, augment=False,
    )
    validation = CDD11Dataset(
        root, mode="val", crop_size=0, val_fraction=args.val_fraction,
        split_seed=args.seed, augment=False,
    )
    gain_train_ids, calibration_ids = choose_scene_split(
        train_full.scene_ids, args.calibration_scenes, args.gain_split_seed
    )
    manifest = {
        "protocol": "train scenes -> source-disjoint calibration scenes -> validation; test untouched",
        "restorer_checkpoint": str(Path(args.checkpoint).resolve()),
        "restorer_epoch": checkpoint.get("epoch"),
        "restorer_model_type": checkpoint.get("model_type"),
        "restorer_preset": checkpoint.get("preset"),
        "gain_train_scene_ids": gain_train_ids,
        "calibration_scene_ids": calibration_ids,
        "validation_scene_ids": list(validation.scene_ids),
        "args": vars(args),
    }
    (output_dir / "protocol.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "args"}, indent=2))

    pairs, targets = make_training_cache(
        train_full, gain_train_ids, restorer, device, args
    )
    print(
        f"cached {len(targets)} blocks; helpful={(targets > 0).float().mean().item():.4f}; "
        f"memory={pairs.numel() * pairs.element_size() / 2**20:.1f} MiB"
    )
    predictor, history = train_predictor(pairs, targets, device, args)
    predictor_payload = {
        "format_version": 1, "model": predictor.state_dict(), "width": args.width,
        "block_size": args.block_size, "target": "(input_sse-output_sse)/(input_sse+output_sse)",
        "gain_train_scene_ids": gain_train_ids,
    }
    torch.save(predictor_payload, output_dir / "gain_predictor.pt")
    write_csv(output_dir / "train_log.csv", history)
    del pairs, targets

    calibration_records = collect_policy_records(
        train_full, calibration_ids, restorer, predictor, device, args
    )
    threshold, beta, calibration = calibrate_policy(calibration_records)
    calibration["scene_ids"] = calibration_ids
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    print("calibration:", json.dumps(calibration, indent=2))

    rows, summary = evaluate_validation(
        validation, restorer, predictor, threshold, beta, device, args
    )
    summary["parameter_counts"] = {
        "restorer": count_parameters(restorer)["total"],
        "gain_predictor": sum(parameter.numel() for parameter in predictor.parameters()),
    }
    write_csv(output_dir / "validation_metrics.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
