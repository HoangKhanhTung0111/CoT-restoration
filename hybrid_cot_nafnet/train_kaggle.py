"""Pure-PyTorch CDD-11 training entry point designed for Kaggle GPUs."""

from __future__ import annotations

import argparse
import csv
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
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, RandomSampler

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_cot_nafnet.datasets import CDD11Dataset, find_cdd11_root
    from hybrid_cot_nafnet.model import build_model, count_parameters
    from hybrid_cot_nafnet.project_config import (
        KAGGLE_CDD11_ROOT,
        pretrained_path_for_preset,
    )
else:
    from .datasets import CDD11Dataset, find_cdd11_root
    from .model import build_model, count_parameters
    from .project_config import KAGGLE_CDD11_ROOT, pretrained_path_for_preset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=str(KAGGLE_CDD11_ROOT),
        help="Directory containing CDD-11_train and CDD-11_test.",
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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=1,
        help="Peak activation memory follows this value; gradients are accumulated.",
    )
    parser.add_argument("--patches-per-image", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
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
        "--multi-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use all visible CUDA devices through DataParallel when more than one exists.",
    )
    parser.add_argument("--freeze-backbone-epochs", type=int, default=3)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def amp_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_loader(dataset, batch_size: int, workers: int, sampler=None, shuffle=False):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


@torch.no_grad()
def validate(
    model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool
) -> Dict[str, float]:
    model.eval()
    psnr_sum = 0.0
    bce_sum = 0.0
    sample_count = 0
    true_positive = false_positive = false_negative = exact = 0
    label_count = 0
    gate_sum = 0.0
    for batch in loader:
        lq = batch["lq"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with amp_context(device, use_amp):
            if has_cot_adapter(model):
                prediction, auxiliary = model(lq, return_aux=True)
                bce = F.binary_cross_entropy_with_logits(
                    auxiliary["degradation_logits"].float(), labels.float()
                )
                predicted_labels = auxiliary["degradation_logits"].sigmoid() >= 0.5
                target_labels = labels >= 0.5
                true_positive += (predicted_labels & target_labels).sum().item()
                false_positive += (predicted_labels & ~target_labels).sum().item()
                false_negative += (~predicted_labels & target_labels).sum().item()
                exact += predicted_labels.eq(target_labels).all(dim=1).sum().item()
                label_count += labels.numel()
                gate_sum += auxiliary["gate_mean_abs"].float().mean().item() * lq.shape[0]
            else:
                prediction, bce = model(lq), torch.zeros((), device=device)
        mse = (prediction.clamp(0, 1).float() - gt.float()).square().mean(dim=(1, 2, 3))
        psnr_sum += (-10.0 * torch.log10(mse + 1e-8)).sum().item()
        bce_sum += bce.item() * lq.shape[0]
        sample_count += lq.shape[0]
    model.train()
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "psnr": psnr_sum / sample_count,
        "bce": bce_sum / sample_count,
        "degradation_micro_f1": (
            2 * true_positive / denominator if denominator else 0.0
        ),
        "degradation_exact_match": exact / sample_count if label_count else 0.0,
        "gate_mean_abs": gate_sum / sample_count if label_count else 0.0,
    }


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best_psnr, args):
    return {
        "format_version": 2,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_psnr": best_psnr,
        "model_type": args.model,
        "preset": args.preset,
        "adapter_hidden": args.adapter_hidden,
        "use_skip_gates": args.skip_gates,
        "args": vars(args),
    }


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


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


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
    if not 0.0 < args.backbone_lr_scale <= 1.0:
        raise ValueError("backbone-lr-scale must be in (0, 1]")
    if args.crop_size % 16:
        raise ValueError("crop-size must be divisible by 16 for the four-level NAFNet")

    seed_everything(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA is unavailable. Enable a Kaggle GPU or pass --allow-cpu for testing.")
    requested_amp = bool(args.amp)
    use_amp = bool(requested_amp and device.type == "cuda")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    data_root = find_cdd11_root(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "run_config.json",
        {**vars(args), "resolved_data_root": str(data_root)},
    )
    _write_json(output_dir / "environment.json", _environment_info(device))
    _write_json(output_dir / "git_info.json", _git_info(Path(__file__).resolve().parents[1]))

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
        crop_size=args.crop_size,
        val_fraction=args.val_fraction,
        split_seed=args.seed,
        augment=False,
    )
    if set(train_set.scene_ids).intersection(val_set.scene_ids):
        raise RuntimeError("Scene-level leakage detected between train and validation")
    test_probe = CDD11Dataset(data_root, mode="test", crop_size=0, augment=False)
    development_ids = set(train_set.scene_ids).union(val_set.scene_ids)
    overlap_with_test = development_ids.intersection(test_probe.scene_ids)
    if overlap_with_test:
        raise RuntimeError(
            f"Scene IDs overlap between development and test: {sorted(overlap_with_test)}"
        )
    _write_json(
        output_dir / "dataset_manifest.json",
        {
            "root": str(data_root),
            "train_scene_ids": list(train_set.scene_ids),
            "validation_scene_ids": list(val_set.scene_ids),
            "test_scene_ids": list(test_probe.scene_ids),
            "train_samples": len(train_set),
            "validation_samples": len(val_set),
            "test_samples": len(test_probe),
        },
    )
    sampler = RandomSampler(
        train_set,
        replacement=True,
        num_samples=len(train_set) * max(1, args.patches_per_image),
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = make_loader(train_set, args.batch_size, args.num_workers, sampler=sampler)
    val_loader = make_loader(val_set, 1, args.num_workers)

    model = build_model(
        args.model,
        args.preset,
        args.adapter_hidden,
        use_skip_gates=args.skip_gates,
    ).to(device)
    parameter_counts = count_parameters(model)
    print(train_set.summary())
    print(val_set.summary())
    print(test_probe.summary())
    print(
        f"Model={args.model}/{args.preset} total={parameter_counts['total']/1e6:.3f}M "
        f"adapter={parameter_counts['adapter']/1e6:.3f}M device={device} amp={use_amp}"
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
    start_epoch, best_psnr = 0, -float("inf")
    if args.resume:
        resume = load_torch_file(args.resume)
        model.load_state_dict(extract_state_dict(resume), strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        if resume.get("scaler"):
            scaler.load_state_dict(resume["scaler"])
        start_epoch = int(resume["epoch"]) + 1
        best_psnr = float(resume.get("best_psnr", best_psnr))
        print(f"Resumed from epoch {start_epoch}")

    gpu_ids = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    if args.multi_gpu and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
        print(
            f"Multi-GPU enabled with DataParallel on {len(gpu_ids)} devices: "
            + ", ".join(torch.cuda.get_device_name(index) for index in gpu_ids)
        )
    else:
        print("Multi-GPU disabled or fewer than two CUDA devices are visible")

    # Cheap preflight catches tensor/channel mistakes before a long Kaggle run.
    sample = train_set[0]["lq"][None].to(device)
    with torch.no_grad(), amp_context(device, use_amp):
        probe = model(sample)
    if not torch.isfinite(probe).all() and use_amp:
        print(
            "WARNING: pretrained model produced non-finite output with AMP; "
            "falling back to FP32 for this run."
        )
        use_amp = False
        args.amp = False
        scaler = make_grad_scaler(False)
        with torch.no_grad(), amp_context(device, False):
            probe = model(sample)
    if probe.shape != sample.shape or not torch.isfinite(probe).all():
        raise RuntimeError(
            f"Preflight failed even after numerical fallback: "
            f"input={sample.shape}, output={probe.shape}, "
            f"finite={bool(torch.isfinite(probe).all())}"
        )
    _write_json(
        output_dir / "runtime_resolution.json",
        {
            "amp_requested": requested_amp,
            "amp_used": use_amp,
            "multi_gpu": isinstance(model, nn.DataParallel),
            "visible_cuda_devices": gpu_ids,
        },
    )
    del sample, probe
    if device.type == "cuda":
        torch.cuda.empty_cache()
        for index in gpu_ids:
            torch.cuda.reset_peak_memory_stats(index)
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    psnr_loss = PSNRLoss()
    log_path = output_dir / "train_log.csv"
    if not log_path.exists() or start_epoch == 0:
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    "epoch", "train_total", "train_restoration",
                    "train_degradation", "train_content", "train_decorrelation",
                    "train_gate", "val_psnr", "val_bce", "val_degradation_micro_f1",
                    "val_degradation_exact_match", "val_gate_mean_abs",
                    "backbone_lr", "adapter_lr", "minutes",
                ]
            )

    started_at = time.monotonic()
    stop_for_time = False
    last_validation: Dict[str, float] = {}
    for epoch in range(start_epoch, args.epochs):
        backbone_trainable = not (
            has_cot_adapter(model) and epoch < args.freeze_backbone_epochs
        )
        set_backbone_trainable(model, backbone_trainable)
        if epoch == start_epoch or epoch == args.freeze_backbone_epochs:
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
        optimizer_steps = 0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            current_batch = batch["lq"].shape[0]
            microbatch_count = math.ceil(current_batch / args.microbatch_size)
            step_values = {key: 0.0 for key in totals}
            try:
                for begin in range(0, current_batch, args.microbatch_size):
                    end = min(current_batch, begin + args.microbatch_size)
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
                    with amp_context(device, use_amp):
                        if has_cot_adapter(model):
                            prediction, auxiliary = model(lq, return_aux=True)
                            degradation = F.binary_cross_entropy_with_logits(
                                auxiliary["degradation_logits"].float(), labels.float()
                            )
                        else:
                            prediction = model(lq)
                            degradation = torch.zeros((), device=device)
                            auxiliary = {}
                        restoration = psnr_loss(prediction, gt)
                        if args.fft_weight:
                            restoration = restoration + args.fft_weight * fft_loss(prediction, gt)
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
                    del lq, gt, labels, prediction, loss, scaled_loss, auxiliary
                scaler.unscale_(optimizer)
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
            if step % args.log_every == 0 or step == len(train_loader):
                elapsed = (time.monotonic() - started_at) / 60.0
                print(
                    f"epoch {epoch+1:03d}/{args.epochs} step {step:04d}/{len(train_loader)} "
                    f"loss={step_values['total']:.4f} deg={step_values['degradation']:.4f} "
                    f"elapsed={elapsed:.1f}m"
                )
            if args.max_minutes > 0 and (time.monotonic() - started_at) / 60.0 >= args.max_minutes:
                stop_for_time = True
                break

        scheduler.step()
        last_validation = validate(model, val_loader, device, use_amp)
        val_psnr = last_validation["psnr"]
        averages = {key: value / max(1, optimizer_steps) for key, value in totals.items()}
        elapsed_minutes = (time.monotonic() - started_at) / 60.0
        print(
            f"epoch {epoch+1:03d}: val_psnr={val_psnr:.4f} "
            f"val_bce={last_validation['bce']:.4f} "
            f"val_f1={last_validation['degradation_micro_f1']:.4f} "
            f"exact={last_validation['degradation_exact_match']:.4f}"
        )
        payload = checkpoint_payload(
            model, optimizer, scheduler, scaler, epoch, max(best_psnr, val_psnr), args
        )
        atomic_save(payload, output_dir / "last.pt")
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            atomic_save(payload, output_dir / "best.pt")
        if (epoch + 1) % args.save_every == 0:
            atomic_save(payload, output_dir / f"epoch_{epoch+1:03d}.pt")
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
                    last_validation["degradation_micro_f1"],
                    last_validation["degradation_exact_match"],
                    last_validation["gate_mean_abs"],
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[-1]["lr"],
                    elapsed_minutes,
                ]
            )
        if stop_for_time:
            print(f"Reached the {args.max_minutes:.1f}-minute safety limit; checkpoint saved.")
            break

    completed_epochs = epoch + 1 if "epoch" in locals() else start_epoch
    _write_json(
        output_dir / "run_summary.json",
        {
            "status": "time_limit" if stop_for_time else "completed",
            "completed_epochs": completed_epochs,
            "best_validation_psnr": best_psnr,
            "elapsed_minutes": (time.monotonic() - started_at) / 60.0,
            "model_type": args.model,
            "preset": args.preset,
            "parameters": parameter_counts,
            "last_validation": last_validation,
            "amp_requested": requested_amp,
            "amp_used": use_amp,
            "multi_gpu": isinstance(model, nn.DataParallel),
            "gpu_peak_memory_mb": (
                {
                    str(index): torch.cuda.max_memory_allocated(index) / 1024**2
                    for index in gpu_ids
                }
                if device.type == "cuda"
                else {}
            ),
        },
    )
    print(f"Training complete. Best validation PSNR: {best_psnr:.4f} dB")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
