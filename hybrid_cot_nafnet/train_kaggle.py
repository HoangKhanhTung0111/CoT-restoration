"""Pure-PyTorch CDD-11 training entry point designed for Kaggle GPUs."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_cot_nafnet.datasets import CDD11Dataset, DEGRADATIONS, find_cdd11_root
    from hybrid_cot_nafnet.datasets.order_controls import OrderControlDataset
    from hybrid_cot_nafnet.degradation_metrics import (
        flattened_degradation_metrics,
        multilabel_degradation_metrics,
    )
    from hybrid_cot_nafnet.model import build_model, count_parameters
    from hybrid_cot_nafnet.project_config import (
        KAGGLE_CDD11_ROOT,
        pretrained_path_for_preset,
    )
else:
    from .datasets import CDD11Dataset, DEGRADATIONS, find_cdd11_root
    from .datasets.order_controls import OrderControlDataset
    from .degradation_metrics import (
        flattened_degradation_metrics,
        multilabel_degradation_metrics,
    )
    from .model import build_model, count_parameters
    from .project_config import KAGGLE_CDD11_ROOT, pretrained_path_for_preset


DEGRADATION_DETAIL_METRICS = (
    "precision",
    "recall",
    "f1",
    "auroc",
    "average_precision",
)
DEGRADATION_VALIDATION_KEYS = (
    "degradation_micro_f1",
    "degradation_macro_f1",
    "degradation_exact_match",
    *(
        f"degradation_{label}_{metric}"
        for label in DEGRADATIONS
        for metric in DEGRADATION_DETAIL_METRICS
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=str(KAGGLE_CDD11_ROOT),
        help="Directory containing CDD-11_train and CDD-11_test.",
    )
    parser.add_argument(
        "--training-data",
        choices=("cdd11", "synthetic_order"),
        default="cdd11",
        help="Use paired CDD-11 or the preregistered synthetic low+haze order control.",
    )
    parser.add_argument(
        "--order-policy",
        choices=("fixed_a", "fixed_b", "balanced"),
        default="balanced",
        help="Formation-order policy used only with --training-data synthetic_order.",
    )
    parser.add_argument("--synthetic-train-realizations", type=int, default=16)
    parser.add_argument("--synthetic-val-realizations", type=int, default=3)
    parser.add_argument("--generation-seed", type=int, default=20260920)
    parser.add_argument(
        "--group-dro",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use epoch-level exponentiated-gradient weights over formation orders.",
    )
    parser.add_argument(
        "--group-dro-eta",
        type=float,
        default=0.1,
        help="Exponentiated-gradient step for the two order-group weights.",
    )
    parser.add_argument("--output-dir", default="/kaggle/working/cot_nafnet_output")
    parser.add_argument("--model", choices=("hybrid", "baseline"), default="hybrid")
    parser.add_argument(
        "--preset",
        choices=("gopro32", "gopro64", "sidd32", "sidd64", "nafnet32", "compact"),
        default="gopro32",
    )
    parser.add_argument("--adapter-hidden", type=int, default=64)
    parser.add_argument(
        "--pretrained",
        default="auto",
        help="Checkpoint path, 'auto' for the matching Kaggle pretrained, or 'none'.",
    )
    parser.add_argument(
        "--allow-partial-pretrained",
        action="store_true",
        help="Allow missing/mismatched NAFNet backbone tensors (unsafe for final runs).",
    )
    parser.add_argument("--resume", default="", help="Resume a full training checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=0.0,
        help="Optional wall-clock safety limit; zero means no time limit.",
    )
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument(
        "--val-crop-size",
        type=int,
        default=0,
        help=(
            "Validation crop size; zero evaluates complete images and matches the "
            "default final-evaluation protocol."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=1,
        help="Peak activation memory follows this value; gradients are accumulated.",
    )
    parser.add_argument("--patches-per-image", type=int, default=2)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="CDD-11-30 is small; zero avoids worker forks retaining model memory.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pin CPU batches; disabled by default to bound Kaggle host RAM.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--fft-weight", type=float, default=0.05)
    parser.add_argument("--degradation-weight", type=float, default=0.05)
    parser.add_argument("--content-weight", type=float, default=0.02)
    parser.add_argument("--decorrelation-weight", type=float, default=0.005)
    parser.add_argument("--gate-weight", type=float, default=0.001)
    parser.add_argument(
        "--skip-gates", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--multi-scale-degradation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build the degradation descriptor from all encoder scales.",
    )
    parser.add_argument(
        "--balanced-degradation-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Weight each degradation label by its training-set negative/positive ratio.",
    )
    parser.add_argument(
        "--multi-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torchrun DistributedDataParallel when WORLD_SIZE is greater than one.",
    )
    parser.add_argument("--freeze-backbone-epochs", type=int, default=3)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save full periodic resume checkpoints every N epochs; zero disables them.",
    )
    parser.add_argument(
        "--save-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include optimizer/scaler state in last.pt for exact resume.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distributed_context(args: argparse.Namespace):
    """Initialize torchrun state and bind one process to one device."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not args.multi_gpu:
        raise RuntimeError("torchrun launched multiple ranks but --no-multi-gpu was set")
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if distributed else 0)
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError(
            "CUDA is unavailable. Enable a Kaggle GPU or pass --allow-cpu for testing."
        )
    if (
        args.multi_gpu
        and not distributed
        and device.type == "cuda"
        and torch.cuda.device_count() > 1
    ):
        raise RuntimeError(
            "Multiple GPUs are visible, but training was not launched with torchrun. "
            "Use `python -m torch.distributed.run --standalone "
            "--nproc_per_node=2 -m hybrid_cot_nafnet.train_kaggle ...` or pass "
            "--no-multi-gpu for a single-GPU fallback."
        )
    return distributed, rank, local_rank, world_size, device


def load_torch_file(path: str | Path, device: str = "cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint) -> Dict[str, Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint)!r}")
    for key in ("model", "state_dict", "params_ema", "params"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
        if isinstance(value, Tensor)
    }


def load_compatible_weights(model: nn.Module, path: str | Path) -> Dict[str, int]:
    source = extract_state_dict(load_torch_file(path))
    target = model.state_dict()
    compatible = {
        key: value for key, value in source.items() if key in target and target[key].shape == value.shape
    }
    mismatched = sum(1 for key, value in source.items() if key in target and target[key].shape != value.shape)
    model.load_state_dict(compatible, strict=False)
    backbone_keys = {key for key in target if not key.startswith("cot_adapter.")}
    loaded_backbone = backbone_keys.intersection(compatible)
    return {
        "loaded": len(compatible),
        "available": len(target),
        "backbone_loaded": len(loaded_backbone),
        "backbone_available": len(backbone_keys),
        "backbone_missing": len(backbone_keys - loaded_backbone),
        "unexpected": sum(key not in target for key in source),
        "mismatched": mismatched,
    }


def _git_info(project_root: Path) -> Dict[str, object]:
    def query(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(project_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "commit": query("rev-parse", "HEAD"),
        "branch": query("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(query("status", "--porcelain")),
    }


def _environment_info(device: torch.device) -> Dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "gpus": (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if device.type == "cuda"
            else []
        ),
    }


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


class PSNRLoss(nn.Module):
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        mse = (prediction.float() - target.float()).square().mean(dim=(1, 2, 3))
        return (10.0 / math.log(10.0)) * torch.log(mse + 1e-8).mean()


def fft_loss(prediction: Tensor, target: Tensor) -> Tensor:
    pred_fft = torch.fft.rfft2(prediction.float(), norm="ortho")
    target_fft = torch.fft.rfft2(target.float(), norm="ortho")
    return F.l1_loss(torch.view_as_real(pred_fft), torch.view_as_real(target_fft))


def restoration_loss_per_sample(
    prediction: Tensor, target: Tensor, fft_weight: float
) -> Tensor:
    """PSNR-style plus FFT loss without reducing the batch dimension."""
    mse = (prediction.float() - target.float()).square().mean(dim=(1, 2, 3))
    losses = (10.0 / math.log(10.0)) * torch.log(mse + 1e-8)
    if fft_weight:
        pred_fft = torch.view_as_real(
            torch.fft.rfft2(prediction.float(), norm="ortho")
        )
        target_fft = torch.view_as_real(
            torch.fft.rfft2(target.float(), norm="ortho")
        )
        fft_per_sample = (pred_fft - target_fft).abs().mean(dim=(1, 2, 3, 4))
        losses = losses + fft_weight * fft_per_sample
    return losses


def amp_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_loader(
    dataset,
    batch_size: int,
    workers: int,
    sampler=None,
    shuffle=False,
    pin_memory: bool = False,
):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=workers,
        pin_memory=bool(pin_memory and torch.cuda.is_available()),
        drop_last=False,
    )
    if workers > 0:
        kwargs.update(persistent_workers=False, prefetch_factor=2)
    return DataLoader(**kwargs)


@torch.no_grad()
def validate(
    model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool
) -> Dict[str, float]:
    validation_model = unwrap_model(model)
    validation_model.eval()
    psnr_sum = 0.0
    bce_sum = 0.0
    sample_count = 0
    label_count = 0
    gate_sum = 0.0
    degradation_targets = []
    degradation_probabilities = []
    order_psnr_sums = {"low>haze": 0.0, "haze>low": 0.0}
    order_counts = {"low>haze": 0, "haze>low": 0}
    for batch in loader:
        lq = batch["lq"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with amp_context(device, use_amp):
            if has_cot_adapter(validation_model):
                prediction, auxiliary = validation_model(lq, return_aux=True)
                bce = F.binary_cross_entropy_with_logits(
                    auxiliary["degradation_logits"].float(), labels.float()
                )
                degradation_probabilities.append(
                    auxiliary["degradation_logits"].sigmoid().float().cpu().numpy()
                )
                degradation_targets.append(labels.float().cpu().numpy())
                label_count += labels.numel()
                gate_sum += auxiliary["gate_mean_abs"].float().mean().item() * lq.shape[0]
            else:
                prediction, bce = validation_model(lq), torch.zeros((), device=device)
        mse = (prediction.clamp(0, 1).float() - gt.float()).square().mean(dim=(1, 2, 3))
        sample_psnr = -10.0 * torch.log10(mse + 1e-8)
        psnr_sum += sample_psnr.sum().item()
        if "formation_order" in batch:
            for order, value in zip(batch["formation_order"], sample_psnr.tolist()):
                if order not in order_psnr_sums:
                    raise RuntimeError(f"Unexpected validation formation order: {order}")
                order_psnr_sums[order] += float(value)
                order_counts[order] += 1
        bce_sum += bce.item() * lq.shape[0]
        sample_count += lq.shape[0]
    model.train()
    result = {
        "psnr": psnr_sum / sample_count,
        "bce": bce_sum / sample_count,
        "gate_mean_abs": gate_sum / sample_count if label_count else 0.0,
    }
    if all(order_counts.values()):
        result["order_a_psnr"] = order_psnr_sums["low>haze"] / order_counts["low>haze"]
        result["order_b_psnr"] = order_psnr_sums["haze>low"] / order_counts["haze>low"]
        result["worst_order_psnr"] = min(
            result["order_a_psnr"], result["order_b_psnr"]
        )
    else:
        result["order_a_psnr"] = result["psnr"]
        result["order_b_psnr"] = result["psnr"]
        result["worst_order_psnr"] = result["psnr"]
    if degradation_targets:
        detailed = multilabel_degradation_metrics(
            np.concatenate(degradation_targets),
            np.concatenate(degradation_probabilities),
            DEGRADATIONS,
        )
        result.update(flattened_degradation_metrics(detailed))
    else:
        result.update({key: 0.0 for key in DEGRADATION_VALIDATION_KEYS})
    return result


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_psnr,
    best_degradation_macro_f1,
    args,
):
    return {
        "format_version": 2,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_psnr": best_psnr,
        "best_degradation_macro_f1": best_degradation_macro_f1,
        "model_type": args.model,
        "preset": args.preset,
        "adapter_hidden": args.adapter_hidden,
        "use_skip_gates": args.skip_gates,
        "use_multiscale_degradation": args.multi_scale_degradation,
        "args": vars(args),
    }


def model_only_checkpoint_payload(
    model, epoch, best_psnr, best_degradation_macro_f1, args, selection_metric
):
    """Small evaluation checkpoint; use `last.pt` when optimizer resume is needed."""
    return {
        "format_version": 2,
        "model": unwrap_model(model).state_dict(),
        "epoch": epoch,
        "best_psnr": best_psnr,
        "best_degradation_macro_f1": best_degradation_macro_f1,
        "selection_metric": selection_metric,
        "model_type": args.model,
        "preset": args.preset,
        "adapter_hidden": args.adapter_hidden,
        "use_skip_gates": args.skip_gates,
        "use_multiscale_degradation": args.multi_scale_degradation,
        "args": vars(args),
    }


def memory_snapshot(device: torch.device) -> Dict[str, object]:
    """Return lightweight Linux/Kaggle host and CUDA memory diagnostics."""
    result: Dict[str, object] = {
        "rss_mb": None,
        "host_available_mb": None,
        "gpu_allocated_mb": {},
        "gpu_reserved_mb": {},
    }
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                result["rss_mb"] = float(line.split()[1]) / 1024.0
                break
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.is_file():
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                result["host_available_mb"] = float(line.split()[1]) / 1024.0
                break
    if device.type == "cuda":
        result["gpu_allocated_mb"] = {
            str(index): torch.cuda.memory_allocated(index) / 1024**2
            for index in range(torch.cuda.device_count())
        }
        result["gpu_reserved_mb"] = {
            str(index): torch.cuda.memory_reserved(index) / 1024**2
            for index in range(torch.cuda.device_count())
        }
    return result


def append_memory_log(
    path: Path,
    epoch: int,
    stage: str,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    snapshot = memory_snapshot(device)
    local_rss = float(snapshot["rss_mb"] or 0.0)
    local_gpu_allocated = (
        float(torch.cuda.memory_allocated(device) / 1024**2)
        if device.type == "cuda"
        else 0.0
    )
    local_gpu_reserved = (
        float(torch.cuda.memory_reserved(device) / 1024**2)
        if device.type == "cuda"
        else 0.0
    )
    local_values = torch.tensor(
        [local_rss, local_gpu_allocated, local_gpu_reserved],
        dtype=torch.float64,
        device=device,
    )
    gathered = [torch.zeros_like(local_values) for _ in range(world_size)]
    if world_size > 1:
        dist.all_gather(gathered, local_values)
    else:
        gathered[0].copy_(local_values)
    if rank != 0:
        return
    rank_values = [item.cpu().tolist() for item in gathered]
    row = {
        "epoch": epoch,
        "stage": stage,
        "rss_total_mb": sum(item[0] for item in rank_values),
        "rss_by_rank_mb": json.dumps(
            {str(index): item[0] for index, item in enumerate(rank_values)}
        ),
        "host_available_mb": snapshot["host_available_mb"],
        "gpu_allocated_by_rank_mb": json.dumps(
            {str(index): item[1] for index, item in enumerate(rank_values)}
        ),
        "gpu_reserved_by_rank_mb": json.dumps(
            {str(index): item[2] for index, item in enumerate(rank_values)}
        ),
    }
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(
        f"memory[{stage}] rss_total={row['rss_total_mb']:.1f}MB "
        f"rss_by_rank={row['rss_by_rank_mb']} "
        f"host_available={snapshot['host_available_mb']}MB "
        f"gpu_allocated={row['gpu_allocated_by_rank_mb']}",
        flush=True,
    )


def content_consistency_loss(embedding: Tensor, pair_size: int) -> Tensor:
    first = F.normalize(embedding[:pair_size].float(), dim=1)
    second = F.normalize(embedding[pair_size:].float(), dim=1)
    return (1.0 - (first * second).sum(dim=1)).mean()


def embedding_decorrelation_loss(content: Tensor, degradation: Tensor) -> Tensor:
    content = F.normalize(content.float(), dim=1)
    degradation = F.normalize(degradation.float(), dim=1)
    return (content * degradation).sum(dim=1).square().mean()


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in unwrap_model(model).named_parameters():
        if not name.startswith("cot_adapter."):
            parameter.requires_grad_(trainable)


def clear_backbone_gradients(model: nn.Module) -> None:
    """Freeze DDP backbone updates without rebuilding reducer hooks."""
    for name, parameter in unwrap_model(model).named_parameters():
        if not name.startswith("cot_adapter."):
            parameter.grad = None


def unwrap_model(model: nn.Module) -> nn.Module:
    return (
        model.module
        if isinstance(model, (nn.DataParallel, DistributedDataParallel))
        else model
    )


def has_cot_adapter(model: nn.Module) -> bool:
    return hasattr(unwrap_model(model), "cot_adapter")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.microbatch_size < 1:
        raise ValueError("batch-size and microbatch-size must be positive")
    if args.microbatch_size > args.batch_size:
        args.microbatch_size = args.batch_size
    if args.freeze_backbone_epochs < 0:
        raise ValueError("freeze-backbone-epochs must be non-negative")
    if args.num_workers < 0 or args.save_every < 0:
        raise ValueError("num-workers and save-every must be non-negative")
    if args.synthetic_train_realizations <= 0 or args.synthetic_val_realizations <= 0:
        raise ValueError("synthetic realization counts must be positive")
    if args.group_dro_eta <= 0:
        raise ValueError("group-dro-eta must be positive")
    if args.group_dro and (
        args.training_data != "synthetic_order"
        or args.order_policy != "balanced"
        or args.model != "baseline"
    ):
        raise ValueError(
            "The preregistered Group DRO control requires synthetic_order, "
            "order_policy=balanced, and model=baseline"
        )
    if args.group_dro and args.resume:
        raise ValueError(
            "The preregistered one-shot Group DRO control does not support resume, "
            "because its group-weight trajectory must start from [0.5, 0.5]."
        )
    if (
        args.training_data == "synthetic_order"
        and args.order_policy == "balanced"
        and args.synthetic_train_realizations % 2
    ):
        raise ValueError("balanced order training requires an even realization count")
    if args.training_data == "synthetic_order" and args.patches_per_image != 1:
        raise ValueError(
            "synthetic_order materializes its update budget via realizations; "
            "set --patches-per-image 1"
        )
    if not 0.0 < args.backbone_lr_scale <= 1.0:
        raise ValueError("backbone-lr-scale must be in (0, 1]")
    if args.crop_size % 16:
        raise ValueError("crop-size must be divisible by 16 for the four-level NAFNet")
    if args.val_crop_size < 0 or (
        args.val_crop_size > 0 and args.val_crop_size % 16
    ):
        raise ValueError("val-crop-size must be zero or divisible by 16")

    if args.training_data == "synthetic_order":
        # The three controls run in separate processes. Make their shared seed
        # meaningful by disabling benchmark-driven kernel selection and asking
        # PyTorch to prefer deterministic implementations when available.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    distributed, rank, local_rank, world_size, device = distributed_context(args)
    is_main = rank == 0
    if args.batch_size % world_size or args.microbatch_size % world_size:
        raise ValueError(
            "Global batch-size and microbatch-size must be divisible by WORLD_SIZE"
        )
    local_batch_size = args.batch_size // world_size
    local_microbatch_size = args.microbatch_size // world_size
    if local_microbatch_size < 1:
        raise ValueError("Per-rank microbatch size must be at least one")
    seed_everything(args.seed + rank)
    requested_amp = bool(args.amp)
    use_amp = bool(requested_amp and device.type == "cuda")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    data_root = find_cdd11_root(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        _write_json(
            output_dir / "run_config.json",
            {
                **vars(args),
                "resolved_data_root": str(data_root),
                "distributed": distributed,
                "world_size": world_size,
                "per_rank_batch_size": local_batch_size,
                "per_rank_microbatch_size": local_microbatch_size,
            },
        )
        _write_json(output_dir / "environment.json", _environment_info(device))
        _write_json(
            output_dir / "git_info.json",
            _git_info(Path(__file__).resolve().parents[1]),
        )

    if args.training_data == "synthetic_order":
        if args.model != "baseline":
            raise ValueError(
                "The preregistered order controls use the baseline model only; "
                "do not add an adapter before the causal control is resolved."
            )
        train_set = OrderControlDataset(
            data_root,
            mode="train",
            order_policy=args.order_policy,
            realizations=args.synthetic_train_realizations,
            crop_size=args.crop_size,
            val_fraction=args.val_fraction,
            split_seed=args.seed,
            generation_seed=args.generation_seed,
        )
        val_set = OrderControlDataset(
            data_root,
            mode="val",
            order_policy="both",
            realizations=args.synthetic_val_realizations,
            crop_size=0,
            val_fraction=args.val_fraction,
            split_seed=args.seed,
            generation_seed=args.generation_seed,
        )
        test_probe = None
    else:
        train_set = CDD11Dataset(
            data_root,
            mode="train",
            crop_size=args.crop_size,
            val_fraction=args.val_fraction,
            split_seed=args.seed,
            augment=True,
            paired_view=args.model == "hybrid" and args.content_weight > 0,
        )
        val_set = CDD11Dataset(
            data_root,
            mode="val",
            crop_size=args.val_crop_size,
            val_fraction=args.val_fraction,
            split_seed=args.seed,
            augment=False,
        )
        test_probe = CDD11Dataset(data_root, mode="test", crop_size=0, augment=False)
    degradation_pos_weight = None
    if args.balanced_degradation_loss:
        label_matrix = torch.stack([sample[2] for sample in train_set.samples]).float()
        positives = label_matrix.sum(dim=0)
        negatives = label_matrix.shape[0] - positives
        if torch.any(positives == 0):
            raise RuntimeError("Balanced degradation loss requires every label in training")
        degradation_pos_weight = (negatives / positives).to(device)
    if set(train_set.scene_ids).intersection(val_set.scene_ids):
        raise RuntimeError("Scene-level leakage detected between train and validation")
    development_ids = set(train_set.scene_ids).union(val_set.scene_ids)
    if test_probe is not None:
        overlap_with_test = development_ids.intersection(test_probe.scene_ids)
        if overlap_with_test:
            raise RuntimeError(
                f"Scene IDs overlap between development and test: {sorted(overlap_with_test)}"
            )
    if is_main:
        _write_json(
            output_dir / "dataset_manifest.json",
            {
                "root": str(data_root),
                "train_scene_ids": list(train_set.scene_ids),
                "validation_scene_ids": list(val_set.scene_ids),
                "test_scene_ids": list(test_probe.scene_ids) if test_probe else [],
                "test_split_loaded": test_probe is not None,
                "train_samples": len(train_set),
                "validation_samples": len(val_set),
                "test_samples": len(test_probe) if test_probe else 0,
                "train_crop_size": args.crop_size,
                "validation_crop_size": args.val_crop_size,
                "degradation_pos_weight": (
                    degradation_pos_weight.cpu().tolist()
                    if degradation_pos_weight is not None
                    else None
                ),
            },
        )
    global_sample_count = len(train_set) * max(1, args.patches_per_image)
    if global_sample_count % world_size:
        raise ValueError("Sample count must be divisible by WORLD_SIZE")
    if args.training_data == "synthetic_order" and distributed:
        sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
    elif args.training_data == "synthetic_order":
        sampler = RandomSampler(
            train_set,
            replacement=False,
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        sampler = RandomSampler(
            train_set,
            replacement=True,
            num_samples=global_sample_count // world_size,
            generator=torch.Generator().manual_seed(args.seed + rank),
        )
    train_loader = make_loader(
        train_set,
        local_batch_size,
        args.num_workers,
        sampler=sampler,
        pin_memory=args.pin_memory,
    )
    val_loader = (
        make_loader(val_set, 1, args.num_workers, pin_memory=args.pin_memory)
        if is_main
        else None
    )

    model = build_model(
        args.model,
        args.preset,
        args.adapter_hidden,
        use_skip_gates=args.skip_gates,
        use_multiscale_degradation=args.multi_scale_degradation,
    ).to(device)
    parameter_counts = count_parameters(model)
    if is_main:
        print(train_set.summary())
        print(val_set.summary())
        if test_probe is not None:
            print(test_probe.summary())
        else:
            print("CDD-11_test not loaded for synthetic order controls")
        print(
            f"Model={args.model}/{args.preset} "
            f"total={parameter_counts['total']/1e6:.3f}M "
            f"adapter={parameter_counts['adapter']/1e6:.3f}M "
            f"world_size={world_size} global_batch={args.batch_size} "
            f"per_rank_batch={local_batch_size} device={device} amp={use_amp}"
        )
    if parameter_counts["adapter"] >= 500_000:
        raise RuntimeError("Adapter exceeds the 0.5M parameter budget")

    pretrained_path = None
    if args.pretrained.lower() == "auto":
        pretrained_path = pretrained_path_for_preset(args.preset)
    elif args.pretrained.lower() not in {"", "none"}:
        pretrained_path = Path(args.pretrained)
    if pretrained_path is not None:
        if not pretrained_path.is_file():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {pretrained_path}. "
                "Attach the nafnetmodel Kaggle input or pass --pretrained none."
            )
        report = load_compatible_weights(model, pretrained_path)
        if is_main:
            print(f"Loaded pretrained weights: {report}")
            _write_json(
                output_dir / "pretrained_report.json",
                {"path": str(pretrained_path), **report},
            )
        if report["backbone_missing"] and not args.allow_partial_pretrained:
            raise RuntimeError(
                "Pretrained checkpoint does not fully match the selected backbone: "
                f"{report}. Select the matching preset or explicitly pass "
                "--allow-partial-pretrained for a diagnostic run."
            )

    if has_cot_adapter(model):
        backbone_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("cot_adapter.")
        ]
        adapter_parameters = list(unwrap_model(model).cot_adapter.parameters())
        optimizer_groups = [
            {
                "params": backbone_parameters,
                "lr": args.learning_rate * args.backbone_lr_scale,
                "name": "backbone",
            },
            {
                "params": adapter_parameters,
                "lr": args.learning_rate,
                "name": "adapter",
            },
        ]
    else:
        optimizer_groups = [
            {"params": list(model.parameters()), "lr": args.learning_rate, "name": "backbone"}
        ]
    optimizer = torch.optim.AdamW(
        optimizer_groups, betas=(0.9, 0.9), weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=1e-7
    )
    scaler = make_grad_scaler(use_amp)
    start_epoch = 0
    best_psnr = -float("inf")
    best_degradation_macro_f1 = -float("inf")
    best_psnr_epoch = 0
    best_reasoning_epoch = 0
    if args.resume:
        resume = load_torch_file(args.resume)
        model.load_state_dict(extract_state_dict(resume), strict=True)
        if "optimizer" in resume and "scheduler" in resume:
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            if resume.get("scaler"):
                scaler.load_state_dict(resume["scaler"])
        else:
            if is_main:
                print(
                    "Resuming model/epoch only; optimizer and scheduler start fresh.",
                    flush=True,
                )
        start_epoch = int(resume["epoch"]) + 1
        best_psnr = float(resume.get("best_psnr", best_psnr))
        best_degradation_macro_f1 = float(
            resume.get(
                "best_degradation_macro_f1", best_degradation_macro_f1
            )
        )
        if is_main:
            print(f"Resumed from epoch {start_epoch}")

    current_backbone_trainable = not (
        has_cot_adapter(model) and start_epoch < args.freeze_backbone_epochs
    )
    if not distributed:
        set_backbone_trainable(model, current_backbone_trainable)
    if distributed:
        if device.type == "cuda":
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
            )
        else:
            model = DistributedDataParallel(model, broadcast_buffers=False)
        if is_main:
            print(
                f"Multi-GPU enabled with DistributedDataParallel on {world_size} "
                "processes (one fixed model replica per device)."
            )
    elif is_main:
        print("Single-process training enabled")

    # Cheap preflight catches tensor/channel mistakes before a long Kaggle run.
    sample = train_set[0]["lq"][None].to(device)
    with torch.no_grad(), amp_context(device, use_amp):
        probe = model(sample)
    finite_flag = torch.tensor(
        [int(torch.isfinite(probe).all())], dtype=torch.int32, device=device
    )
    if distributed:
        dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
    if not finite_flag.item() and use_amp:
        if is_main:
            print(
                "WARNING: pretrained model produced non-finite output with AMP; "
                "falling back to FP32 for this run."
            )
        use_amp = False
        args.amp = False
        scaler = make_grad_scaler(False)
        with torch.no_grad(), amp_context(device, False):
            probe = model(sample)
    valid_probe = probe.shape == sample.shape and bool(torch.isfinite(probe).all())
    valid_probe_flag = torch.tensor([int(valid_probe)], dtype=torch.int32, device=device)
    if distributed:
        dist.all_reduce(valid_probe_flag, op=dist.ReduceOp.MIN)
    if not valid_probe_flag.item():
        raise RuntimeError(
            f"Preflight failed even after numerical fallback: "
            f"input={sample.shape}, output={probe.shape}, "
            f"finite={bool(torch.isfinite(probe).all())}"
        )
    if is_main:
        _write_json(
            output_dir / "runtime_resolution.json",
            {
                "amp_requested": requested_amp,
                "amp_used": use_amp,
                "multi_gpu": distributed,
                "parallel_strategy": "ddp" if distributed else "single_process",
                "world_size": world_size,
                "visible_cuda_devices": list(range(torch.cuda.device_count())),
                "deterministic_order_control": args.training_data == "synthetic_order",
            },
        )
    del sample, probe
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        if is_main:
            print(f"Rank 0 GPU: {torch.cuda.get_device_name(device)}")

    psnr_loss = PSNRLoss()
    group_weights = torch.tensor([0.5, 0.5], dtype=torch.float64, device=device)
    last_group_losses = torch.tensor(
        [float("nan"), float("nan")], dtype=torch.float64, device=device
    )
    log_path = output_dir / "train_log.csv"
    memory_log_path = output_dir / "memory_log.csv"
    if is_main and (not log_path.exists() or start_epoch == 0):
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    "epoch", "train_total", "train_restoration",
                    "train_degradation", "train_content", "train_decorrelation",
                    "train_gate", "val_psnr", "val_bce",
                    *[f"val_{key}" for key in DEGRADATION_VALIDATION_KEYS],
                    "val_gate_mean_abs", "val_order_a_psnr", "val_order_b_psnr",
                    "val_worst_order_psnr", "checkpoint_selection_score",
                    "next_group_weight_a", "next_group_weight_b",
                    "group_train_loss_a", "group_train_loss_b",
                    "backbone_lr", "adapter_lr", "minutes",
                ]
            )
    if is_main and start_epoch == 0 and memory_log_path.exists():
        memory_log_path.unlink()
    if distributed:
        dist.barrier()
    append_memory_log(
        memory_log_path, start_epoch, "setup", device, rank, world_size
    )

    started_at = time.monotonic()
    stop_for_time = False
    last_validation: Dict[str, float] = {}
    reasoning_checkpoint_enabled = (
        has_cot_adapter(model) and args.degradation_weight > 0
    )
    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_set, "set_epoch"):
            train_set.set_epoch(epoch)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        backbone_trainable = not (
            has_cot_adapter(model) and epoch < args.freeze_backbone_epochs
        )
        if backbone_trainable != current_backbone_trainable:
            if not distributed:
                set_backbone_trainable(model, backbone_trainable)
            current_backbone_trainable = backbone_trainable
        if is_main and (epoch == start_epoch or epoch == args.freeze_backbone_epochs):
            print(
                f"Backbone {'trainable' if backbone_trainable else 'frozen'} "
                f"at epoch {epoch + 1}"
            )
        model.train()
        totals = {
            "total": 0.0,
            "restoration": 0.0,
            "degradation": 0.0,
            "content": 0.0,
            "decorrelation": 0.0,
            "gate": 0.0,
        }
        epoch_group_loss_sums = torch.zeros(2, dtype=torch.float64, device=device)
        epoch_group_counts = torch.zeros(2, dtype=torch.float64, device=device)
        weights_used_this_epoch = group_weights.clone()
        optimizer_steps = 0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            current_batch = batch["lq"].shape[0]
            microbatch_count = math.ceil(current_batch / local_microbatch_size)
            step_values = {key: 0.0 for key in totals}
            try:
                for begin in range(0, current_batch, local_microbatch_size):
                    end = min(current_batch, begin + local_microbatch_size)
                    lq = batch["lq"][begin:end].to(device, non_blocking=True)
                    gt = batch["gt"][begin:end].to(device, non_blocking=True)
                    labels = batch["label"][begin:end].to(device, non_blocking=True)
                    pair_size = lq.shape[0]
                    has_paired_view = "lq_view2" in batch
                    if has_paired_view:
                        lq = torch.cat(
                            [lq, batch["lq_view2"][begin:end].to(device, non_blocking=True)]
                        )
                        gt = torch.cat([gt, gt], dim=0)
                        labels = torch.cat(
                            [
                                labels,
                                batch["label_view2"][begin:end].to(
                                    device, non_blocking=True
                                ),
                            ]
                        )
                    synchronize = end >= current_batch
                    sync_context = (
                        nullcontext()
                        if not distributed or synchronize
                        else model.no_sync()
                    )
                    with sync_context:
                        with amp_context(device, use_amp):
                            if has_cot_adapter(model):
                                prediction, auxiliary = model(lq, return_aux=True)
                                degradation = F.binary_cross_entropy_with_logits(
                                    auxiliary["degradation_logits"].float(),
                                    labels.float(),
                                    pos_weight=degradation_pos_weight,
                                )
                            else:
                                prediction = model(lq)
                                degradation = torch.zeros((), device=device)
                                auxiliary = {}
                            if args.group_dro:
                                per_sample_restoration = restoration_loss_per_sample(
                                    prediction, gt, args.fft_weight
                                )
                                order_names = list(batch["formation_order"])[begin:end]
                                group_ids = torch.tensor(
                                    [0 if name == "low>haze" else 1 for name in order_names],
                                    dtype=torch.long,
                                    device=device,
                                )
                                if any(
                                    name not in {"low>haze", "haze>low"}
                                    for name in order_names
                                ):
                                    raise RuntimeError(
                                        f"Unexpected training formation order: {order_names}"
                                    )
                                # The full epoch is exactly 50/50 A/B, so 2*q_g
                                # gives an unbiased estimate of sum_g q_g E[L|g].
                                restoration = (
                                    per_sample_restoration
                                    * (
                                        2.0
                                        * weights_used_this_epoch[group_ids].to(
                                            per_sample_restoration.dtype
                                        )
                                    )
                                ).mean()
                                epoch_group_loss_sums.scatter_add_(
                                    0,
                                    group_ids,
                                    per_sample_restoration.detach().double(),
                                )
                                epoch_group_counts.scatter_add_(
                                    0,
                                    group_ids,
                                    torch.ones_like(group_ids, dtype=torch.float64),
                                )
                            else:
                                restoration = psnr_loss(prediction, gt)
                                if args.fft_weight:
                                    restoration = restoration + args.fft_weight * fft_loss(
                                        prediction, gt
                                    )
                            if has_paired_view:
                                content = content_consistency_loss(
                                    auxiliary["content_embedding"], pair_size
                                )
                            else:
                                content = torch.zeros((), device=device)
                            if has_cot_adapter(model):
                                decorrelation = embedding_decorrelation_loss(
                                    auxiliary["content_embedding"],
                                    auxiliary["degradation_embedding"],
                                )
                                gate = auxiliary["gate_regularization"].mean()
                            else:
                                decorrelation = torch.zeros((), device=device)
                                gate = torch.zeros((), device=device)
                            loss = (
                                restoration
                                + args.degradation_weight * degradation
                                + args.content_weight * content
                                + args.decorrelation_weight * decorrelation
                                + args.gate_weight * gate
                            )
                            scaled_loss = loss / microbatch_count
                        scaler.scale(scaled_loss).backward()
                    step_values["total"] += loss.detach().item() / microbatch_count
                    step_values["restoration"] += restoration.detach().item() / microbatch_count
                    step_values["degradation"] += degradation.detach().item() / microbatch_count
                    step_values["content"] += content.detach().item() / microbatch_count
                    step_values["decorrelation"] += decorrelation.detach().item() / microbatch_count
                    step_values["gate"] += gate.detach().item() / microbatch_count
                    del (
                        lq,
                        gt,
                        labels,
                        prediction,
                        loss,
                        scaled_loss,
                        auxiliary,
                        restoration,
                        degradation,
                        content,
                        decorrelation,
                        gate,
                    )
                    if args.group_dro:
                        del per_sample_restoration, group_ids
                scaler.unscale_(optimizer)
                if not backbone_trainable:
                    clear_backbone_gradients(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            except torch.cuda.OutOfMemoryError as error:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA OOM despite microbatching. Re-run with --crop-size 192 "
                    "or --preset compact; the last completed checkpoint is intact."
                ) from error

            optimizer_steps += 1
            for key in totals:
                totals[key] += step_values[key]
            if is_main and (step % args.log_every == 0 or step == len(train_loader)):
                elapsed = (time.monotonic() - started_at) / 60.0
                print(
                    f"epoch {epoch+1:03d}/{args.epochs} step {step:04d}/{len(train_loader)} "
                    f"loss={step_values['total']:.4f} deg={step_values['degradation']:.4f} "
                    f"elapsed={elapsed:.1f}m"
                )
            if args.max_minutes > 0:
                reached_limit = int(
                    (time.monotonic() - started_at) / 60.0 >= args.max_minutes
                )
                limit_flag = torch.tensor(
                    [reached_limit], dtype=torch.int32, device=device
                )
                if distributed:
                    dist.all_reduce(limit_flag, op=dist.ReduceOp.MAX)
                if limit_flag.item():
                    stop_for_time = True
                    break

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if "batch" in locals():
            del batch
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        aggregate = torch.tensor(
            [*[totals[key] for key in totals], float(optimizer_steps)],
            dtype=torch.float64,
            device=device,
        )
        if distributed:
            dist.all_reduce(aggregate, op=dist.ReduceOp.SUM)
        reduced = aggregate.cpu().tolist()
        reduced_totals = dict(zip(totals, reduced[:-1]))
        reduced_steps = max(1.0, reduced[-1])
        if args.group_dro:
            group_statistics = torch.cat(
                [epoch_group_loss_sums, epoch_group_counts]
            )
            if distributed:
                dist.all_reduce(group_statistics, op=dist.ReduceOp.SUM)
            group_sums = group_statistics[:2]
            group_counts = group_statistics[2:]
            if torch.any(group_counts <= 0):
                raise RuntimeError(
                    f"Group DRO epoch is missing an order group: {group_counts.tolist()}"
                )
            last_group_losses = group_sums / group_counts
            centered_losses = last_group_losses - last_group_losses.mean()
            group_weights = group_weights * torch.exp(
                args.group_dro_eta * centered_losses
            )
            group_weights = group_weights / group_weights.sum()
        append_memory_log(
            memory_log_path,
            epoch + 1,
            "after_train",
            device,
            rank,
            world_size,
        )
        if is_main:
            print(
                f"epoch {epoch+1:03d}: training pass complete; starting validation",
                flush=True,
            )
            assert val_loader is not None
            last_validation = validate(model, val_loader, device, use_amp)
        metric_keys = (
            "psnr",
            "bce",
            *DEGRADATION_VALIDATION_KEYS,
            "gate_mean_abs",
            "order_a_psnr",
            "order_b_psnr",
            "worst_order_psnr",
        )
        metric_tensor = torch.tensor(
            [last_validation.get(key, 0.0) for key in metric_keys],
            dtype=torch.float64,
            device=device,
        )
        if distributed:
            dist.broadcast(metric_tensor, src=0)
        last_validation = dict(zip(metric_keys, metric_tensor.cpu().tolist()))
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        append_memory_log(
            memory_log_path,
            epoch + 1,
            "after_validation",
            device,
            rank,
            world_size,
        )
        val_psnr = last_validation["psnr"]
        checkpoint_selection_score = (
            last_validation["worst_order_psnr"]
            if args.group_dro
            else val_psnr
        )
        val_macro_f1 = last_validation["degradation_macro_f1"]
        psnr_improved = checkpoint_selection_score > best_psnr
        reasoning_improved = (
            reasoning_checkpoint_enabled
            and val_macro_f1 > best_degradation_macro_f1
        )
        next_best_psnr = max(best_psnr, checkpoint_selection_score)
        next_best_macro_f1 = max(best_degradation_macro_f1, val_macro_f1)
        averages = {key: value / reduced_steps for key, value in reduced_totals.items()}
        elapsed_minutes = (time.monotonic() - started_at) / 60.0
        if is_main:
            print(
                f"epoch {epoch+1:03d}: val_psnr={val_psnr:.4f} "
                f"worst_order={last_validation['worst_order_psnr']:.4f} "
                f"val_bce={last_validation['bce']:.4f} "
                f"val_f1={last_validation['degradation_micro_f1']:.4f} "
                f"macro_f1={last_validation['degradation_macro_f1']:.4f} "
                f"exact={last_validation['degradation_exact_match']:.4f}"
            )
            if args.save_optimizer:
                payload = checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    next_best_psnr,
                    next_best_macro_f1,
                    args,
                )
            else:
                payload = model_only_checkpoint_payload(
                    model,
                    epoch,
                    next_best_psnr,
                    next_best_macro_f1,
                    args,
                    "last",
                )
            atomic_save(payload, output_dir / "last.pt")
            if psnr_improved:
                atomic_save(
                    model_only_checkpoint_payload(
                        model,
                        epoch,
                        next_best_psnr,
                        next_best_macro_f1,
                        args,
                        (
                            "validation_worst_order_psnr"
                            if args.group_dro
                            else "validation_psnr"
                        ),
                    ),
                    output_dir / "best.pt",
                )
            if reasoning_improved:
                atomic_save(
                    model_only_checkpoint_payload(
                        model,
                        epoch,
                        next_best_psnr,
                        next_best_macro_f1,
                        args,
                        "degradation_macro_f1",
                    ),
                    output_dir / "best_reasoning.pt",
                )
            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                atomic_save(payload, output_dir / f"epoch_{epoch+1:03d}.pt")
            del payload
        if psnr_improved:
            best_psnr = checkpoint_selection_score
            best_psnr_epoch = epoch + 1
        if reasoning_improved:
            best_degradation_macro_f1 = val_macro_f1
            best_reasoning_epoch = epoch + 1
        gc.collect()
        if distributed:
            dist.barrier()
        append_memory_log(
            memory_log_path,
            epoch + 1,
            "after_checkpoint",
            device,
            rank,
            world_size,
        )
        if is_main:
            with log_path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(
                    [
                        epoch + 1,
                        averages["total"],
                        averages["restoration"],
                        averages["degradation"],
                        averages["content"],
                        averages["decorrelation"],
                        averages["gate"],
                        last_validation["psnr"],
                        last_validation["bce"],
                        *[
                            last_validation.get(key, 0.0)
                            for key in DEGRADATION_VALIDATION_KEYS
                        ],
                        last_validation["gate_mean_abs"],
                        last_validation["order_a_psnr"],
                        last_validation["order_b_psnr"],
                        last_validation["worst_order_psnr"],
                        checkpoint_selection_score,
                        float(group_weights[0].item()),
                        float(group_weights[1].item()),
                        float(last_group_losses[0].item()),
                        float(last_group_losses[1].item()),
                        optimizer.param_groups[0]["lr"],
                        optimizer.param_groups[-1]["lr"],
                        elapsed_minutes,
                    ]
                )
        if stop_for_time:
            if is_main:
                print(
                    f"Reached the {args.max_minutes:.1f}-minute safety limit; "
                    "checkpoint saved."
                )
            break

    local_peak = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    peak_tensor = torch.tensor([local_peak], dtype=torch.float64, device=device)
    gathered_peaks = [torch.zeros_like(peak_tensor) for _ in range(world_size)]
    if distributed:
        dist.all_gather(gathered_peaks, peak_tensor)
    else:
        gathered_peaks[0].copy_(peak_tensor)
    completed_epochs = epoch + 1 if "epoch" in locals() else start_epoch
    if is_main:
        _write_json(
            output_dir / "run_summary.json",
            {
                "status": "time_limit" if stop_for_time else "completed",
                "completed_epochs": completed_epochs,
                "best_validation_psnr": best_psnr,
                "best_validation_psnr_epoch": best_psnr_epoch,
                "checkpoint_selection_metric": (
                    "validation_worst_order_psnr"
                    if args.group_dro
                    else "validation_psnr"
                ),
                "best_checkpoint_selection_score": best_psnr,
                "group_dro": args.group_dro,
                "group_dro_eta": args.group_dro_eta if args.group_dro else None,
                "final_group_weights": (
                    group_weights.cpu().tolist() if args.group_dro else None
                ),
                "final_group_train_losses": (
                    last_group_losses.cpu().tolist() if args.group_dro else None
                ),
                "best_degradation_macro_f1": (
                    best_degradation_macro_f1
                    if reasoning_checkpoint_enabled
                    else None
                ),
                "best_degradation_macro_f1_epoch": (
                    best_reasoning_epoch if reasoning_checkpoint_enabled else None
                ),
                "elapsed_minutes": (time.monotonic() - started_at) / 60.0,
                "model_type": args.model,
                "preset": args.preset,
                "parameters": parameter_counts,
                "validation_crop_size": args.val_crop_size,
                "validation_protocol": (
                    "full_frame" if args.val_crop_size == 0 else "center_crop"
                ),
                "last_validation": last_validation,
                "amp_requested": requested_amp,
                "amp_used": use_amp,
                "multi_gpu": distributed,
                "parallel_strategy": "ddp" if distributed else "single_process",
                "world_size": world_size,
                "gpu_peak_memory_mb_by_rank": {
                    str(index): value.item()
                    for index, value in enumerate(gathered_peaks)
                },
            },
        )
        selection_label = (
            "worst-order validation PSNR" if args.group_dro else "validation PSNR"
        )
        print(f"Training complete. Best {selection_label}: {best_psnr:.4f} dB")
        if reasoning_checkpoint_enabled:
            print(
                "Best degradation macro-F1: "
                f"{best_degradation_macro_f1:.4f}"
            )
        print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
