"""Evaluate a project checkpoint on all 11 CDD-11 degradation categories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_cot_nafnet.datasets import CDD11Dataset, DEGRADATIONS, find_cdd11_root
    from hybrid_cot_nafnet.model import build_model, count_parameters
    from hybrid_cot_nafnet.project_config import KAGGLE_CDD11_ROOT
else:
    from .datasets import CDD11Dataset, DEGRADATIONS, find_cdd11_root
    from .model import build_model, count_parameters
    from .project_config import KAGGLE_CDD11_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data-root", default=str(KAGGLE_CDD11_ROOT)
    )
    parser.add_argument("--output-dir", default="/kaggle/working/cot_nafnet_evaluation")
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def load_torch_file(path: str | Path, device: str = "cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def amp_context(enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else nullcontext()


def tile_starts(length: int, tile: int, overlap: int) -> List[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile")
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.inference_mode()
def tiled_inference(
    model: nn.Module,
    image_cpu: Tensor,
    device: torch.device,
    tile: int,
    overlap: int,
    use_amp: bool,
) -> Tensor:
    """Run one image while keeping the full-resolution buffers on CPU."""
    if image_cpu.ndim != 4 or image_cpu.shape[0] != 1:
        raise ValueError("tiled_inference expects a [1,C,H,W] tensor")
    _, _, height, width = image_cpu.shape
    if tile <= 0:
        with amp_context(use_amp):
            return model(image_cpu.to(device)).float().cpu()
    y_starts = tile_starts(height, tile, overlap)
    x_starts = tile_starts(width, tile, overlap)
    output = torch.zeros_like(image_cpu, dtype=torch.float32, device="cpu")
    weights = torch.zeros((1, 1, height, width), dtype=torch.float32, device="cpu")
    for top in y_starts:
        for left in x_starts:
            patch_cpu = image_cpu[..., top : min(top + tile, height), left : min(left + tile, width)]
            with amp_context(use_amp):
                prediction = model(patch_cpu.to(device, non_blocking=True))
            prediction = prediction.float().cpu()
            patch_h, patch_w = prediction.shape[-2:]
            output[..., top : top + patch_h, left : left + patch_w] += prediction
            weights[..., top : top + patch_h, left : left + patch_w] += 1.0
            del prediction
    return output / weights.clamp_min(1.0)


def calculate_psnr(prediction: Tensor, target: Tensor) -> float:
    mse = (prediction.float() - target.float()).square().mean().item()
    return -10.0 * math.log10(max(mse, 1e-8))


def _gaussian_window(channels: int, size: int, sigma: float = 1.5) -> Tensor:
    coordinates = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, size, size).contiguous()


def calculate_ssim(prediction: Tensor, target: Tensor) -> float:
    """RGB SSIM with the standard 11x11 Gaussian window and valid convolution."""
    prediction, target = prediction.float(), target.float()
    channels = prediction.shape[1]
    smallest_side = min(prediction.shape[-2:])
    window_size = min(11, smallest_side if smallest_side % 2 else smallest_side - 1)
    window_size = max(1, window_size)
    window = _gaussian_window(channels, window_size)
    mu_pred = F.conv2d(prediction, window, groups=channels)
    mu_target = F.conv2d(target, window, groups=channels)
    mu_pred_sq, mu_target_sq = mu_pred.square(), mu_target.square()
    mu_cross = mu_pred * mu_target
    variance_pred = F.conv2d(prediction.square(), window, groups=channels) - mu_pred_sq
    variance_target = F.conv2d(target.square(), window, groups=channels) - mu_target_sq
    covariance = F.conv2d(prediction * target, window, groups=channels) - mu_cross
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mu_cross + c1) * (2 * covariance + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (
        variance_pred + variance_target + c2
    )
    return (numerator / denominator.clamp_min(1e-12)).mean().item()


def save_image(tensor: Tensor, path: Path) -> None:
    array = (
        tensor[0]
        .clamp(0, 1)
        .mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


@torch.inference_mode()
def estimate_conv_linear_macs(
    model: nn.Module, device: torch.device, use_amp: bool, size: int = 256
) -> int:
    total = 0
    hooks = []

    def hook(module: nn.Module, inputs: Sequence[Tensor], output: Tensor):
        nonlocal total
        if isinstance(module, nn.Conv2d):
            kernel_ops = module.kernel_size[0] * module.kernel_size[1]
            kernel_ops *= module.in_channels // module.groups
            total += output.numel() * kernel_ops
        elif isinstance(module, nn.Linear):
            total += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    dummy = torch.zeros(1, 3, size, size, device=device)
    try:
        with amp_context(use_amp):
            model(dummy)
    finally:
        for item in hooks:
            item.remove()
    return total


def main() -> None:
    args = parse_args()
    if args.tile > 0 and args.tile % 16:
        raise ValueError("tile must be divisible by 16")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA is unavailable. Enable a Kaggle GPU or pass --allow-cpu.")
    use_amp = bool(args.amp and device.type == "cuda")
    checkpoint = load_torch_file(args.checkpoint)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError("Expected a checkpoint created by train_kaggle.py")
    model_type = checkpoint.get("model_type", "hybrid")
    preset = checkpoint.get("preset", "nafnet32")
    adapter_hidden = int(checkpoint.get("adapter_hidden", 64))
    model = build_model(model_type, preset, adapter_hidden)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    parameter_counts = count_parameters(model)
    macs = estimate_conv_linear_macs(model, device, use_amp)
    print(
        f"Model={model_type}/{preset} params={parameter_counts['total']/1e6:.3f}M "
        f"adapter={parameter_counts['adapter']/1e6:.3f}M MACs@256={macs/1e9:.3f}G"
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    dataset = CDD11Dataset(
        find_cdd11_root(args.data_root), mode="test", crop_size=0, augment=False
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )
    print(dataset.summary())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    grouped: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    for index, batch in enumerate(loader, start=1):
        lq_cpu, gt_cpu = batch["lq"].float(), batch["gt"].float()
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = tiled_inference(
            model, lq_cpu, device, args.tile, args.overlap, use_amp
        ).clamp(0, 1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        degradation_type = batch["degradation_type"][0]
        scene_id = batch["scene_id"][0]
        psnr = calculate_psnr(prediction, gt_cpu)
        ssim = calculate_ssim(prediction, gt_cpu)
        input_psnr = calculate_psnr(lq_cpu, gt_cpu)
        grouped[degradation_type].append((psnr, ssim))
        rows.append(
            {
                "type": degradation_type,
                "scene_id": scene_id,
                "psnr": psnr,
                "ssim": ssim,
                "input_psnr": input_psnr,
                "latency_ms": latency_ms,
            }
        )
        if args.save_images:
            save_image(prediction, output_dir / "images" / degradation_type / f"{scene_id}.png")
        print(
            f"[{index:03d}/{len(dataset)}] {degradation_type}/{scene_id} "
            f"PSNR={psnr:.3f} SSIM={ssim:.4f} time={latency_ms:.1f}ms"
        )

    per_type = {
        degradation_type: {
            "psnr": float(np.mean([item[0] for item in values])),
            "ssim": float(np.mean([item[1] for item in values])),
            "count": len(values),
        }
        for degradation_type, values in sorted(grouped.items())
    }
    macro_psnr = float(np.mean([value["psnr"] for value in per_type.values()]))
    macro_ssim = float(np.mean([value["ssim"] for value in per_type.values()]))
    summary = {
        "model_type": model_type,
        "preset": preset,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "samples": len(rows),
        "macro_psnr": macro_psnr,
        "macro_ssim": macro_ssim,
        "mean_latency_ms": float(np.mean([row["latency_ms"] for row in rows])),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
        ),
        "parameters": parameter_counts,
        "macs_256": macs,
        "per_type": per_type,
    }
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nPer-type results")
    for degradation_type, values in per_type.items():
        print(
            f"{degradation_type:16s} PSNR={values['psnr']:.4f} "
            f"SSIM={values['ssim']:.4f} n={values['count']}"
        )
    print(f"MACRO             PSNR={macro_psnr:.4f} SSIM={macro_ssim:.4f}")
    print(f"Saved evaluation to {output_dir}")


if __name__ == "__main__":
    main()
