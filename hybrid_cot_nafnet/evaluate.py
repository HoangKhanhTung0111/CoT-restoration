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
from PIL import Image, ImageDraw
from torch import Tensor, nn
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_cot_nafnet.datasets import CDD11Dataset, DEGRADATIONS, find_cdd11_root
    from hybrid_cot_nafnet.degradation_metrics import (
        multilabel_degradation_metrics,
    )
    from hybrid_cot_nafnet.model import build_model, count_parameters
    from hybrid_cot_nafnet.project_config import KAGGLE_CDD11_ROOT
else:
    from .datasets import CDD11Dataset, DEGRADATIONS, find_cdd11_root
    from .degradation_metrics import multilabel_degradation_metrics
    from .model import build_model, count_parameters
    from .project_config import KAGGLE_CDD11_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data-root", default=str(KAGGLE_CDD11_ROOT)
    )
    parser.add_argument("--output-dir", default="/kaggle/working/cot_nafnet_evaluation")
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Use validation while developing; reserve test for a locked final run.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tile",
        type=int,
        default=0,
        help="Tile size; zero uses artifact-free full-frame inference (CDD-11 default).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=32,
        help="Minimum feathered overlap used only when --tile is positive.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-comparisons",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save labelled Input | Restored | Ground truth contact sheets.",
    )
    parser.add_argument("--max-saved-per-type", type=int, default=1)
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
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile")
    extent = length - tile
    interval_count = math.ceil(extent / stride)
    # Even spacing avoids a nearly duplicated final tile and guarantees at
    # least the requested overlap everywhere.
    return [
        round(index * extent / interval_count)
        for index in range(interval_count + 1)
    ]


def tile_blend_window(
    patch_height: int,
    patch_width: int,
    overlap: int,
    top: int,
    left: int,
    image_height: int,
    image_width: int,
) -> Tensor:
    """Cosine feather weights that suppress context-poor internal tile edges."""

    def axis_weights(size: int, start: int, full_size: int) -> Tensor:
        weights = torch.ones(size, dtype=torch.float32)
        blend = min(overlap, size)
        if blend == 0:
            return weights
        phase = (torch.arange(blend, dtype=torch.float32) + 0.5) / blend
        ramp = torch.sin(phase * (math.pi / 2.0)).square()
        if start > 0:
            weights[:blend] = ramp
        if start + size < full_size:
            weights[-blend:] = ramp.flip(0)
        return weights

    vertical = axis_weights(patch_height, top, image_height)
    horizontal = axis_weights(patch_width, left, image_width)
    return (vertical[:, None] * horizontal[None, :])[None, None]


@torch.inference_mode()
def tiled_inference(
    model: nn.Module,
    image_cpu: Tensor,
    device: torch.device,
    tile: int,
    overlap: int,
    use_amp: bool,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Run one image while keeping the full-resolution buffers on CPU."""
    if image_cpu.ndim != 4 or image_cpu.shape[0] != 1:
        raise ValueError("tiled_inference expects a [1,C,H,W] tensor")
    _, _, height, width = image_cpu.shape
    if tile <= 0:
        with amp_context(use_amp):
            if hasattr(model, "cot_adapter"):
                prediction, auxiliary = model(
                    image_cpu.to(device), return_aux=True
                )
                return prediction.float().cpu(), {
                    "degradation_logits": auxiliary["degradation_logits"].float().cpu(),
                    "gate_mean_abs": auxiliary["gate_mean_abs"].float().cpu(),
                }
            return model(image_cpu.to(device)).float().cpu(), {}
    y_starts = tile_starts(height, tile, overlap)
    x_starts = tile_starts(width, tile, overlap)
    output = torch.zeros_like(image_cpu, dtype=torch.float32, device="cpu")
    weights = torch.zeros((1, 1, height, width), dtype=torch.float32, device="cpu")
    logits_sum = None
    gate_sum = 0.0
    tile_count = 0
    for top in y_starts:
        for left in x_starts:
            patch_cpu = image_cpu[..., top : min(top + tile, height), left : min(left + tile, width)]
            with amp_context(use_amp):
                if hasattr(model, "cot_adapter"):
                    prediction, auxiliary = model(
                        patch_cpu.to(device, non_blocking=True), return_aux=True
                    )
                    logits = auxiliary["degradation_logits"].float().cpu()
                    logits_sum = logits if logits_sum is None else logits_sum + logits
                    gate_sum += auxiliary["gate_mean_abs"].item()
                else:
                    prediction = model(patch_cpu.to(device, non_blocking=True))
            prediction = prediction.float().cpu()
            patch_h, patch_w = prediction.shape[-2:]
            blend = tile_blend_window(
                patch_h,
                patch_w,
                overlap,
                top,
                left,
                height,
                width,
            )
            output[..., top : top + patch_h, left : left + patch_w] += (
                prediction * blend
            )
            weights[..., top : top + patch_h, left : left + patch_w] += blend
            tile_count += 1
            del prediction, blend
    auxiliary_output: Dict[str, Tensor] = {}
    if logits_sum is not None:
        auxiliary_output = {
            "degradation_logits": logits_sum / tile_count,
            "gate_mean_abs": torch.tensor(gate_sum / tile_count),
        }
    return output / weights.clamp_min(1.0), auxiliary_output


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
    Image.fromarray(array).save(path)


def tensor_to_pil(tensor: Tensor) -> Image.Image:
    array = (
        tensor[0]
        .clamp(0, 1)
        .mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array)


def save_comparison(
    lq: Tensor,
    prediction: Tensor,
    gt: Tensor,
    path: Path,
    title: str,
) -> None:
    """Save an explicitly labelled input/output/target contact sheet."""
    panels = [tensor_to_pil(item) for item in (lq, prediction, gt)]
    labels = ("Input (degraded)", "Restored", "Ground truth")
    header_height = 44
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), panels[0].height + header_height),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), title, fill="black")
    left = 0
    for panel, label in zip(panels, labels):
        canvas.paste(panel, (left, header_height))
        draw.text((left + 6, 25), label, fill="black")
        left += panel.width
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


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
    if args.max_saved_per_type < 0:
        raise ValueError("max-saved-per-type must be non-negative")
    if args.tile > 0 and args.tile % 16:
        raise ValueError("tile must be divisible by 16")
    if args.overlap < 0 or (args.tile > 0 and args.overlap >= args.tile):
        raise ValueError("overlap must be non-negative and smaller than tile")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA is unavailable. Enable a Kaggle GPU or pass --allow-cpu.")
    requested_amp = bool(args.amp and device.type == "cuda")
    use_amp = requested_amp
    checkpoint = load_torch_file(args.checkpoint)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError("Expected a checkpoint created by train_kaggle.py")
    model_type = checkpoint.get("model_type", "hybrid")
    preset = checkpoint.get("preset", "nafnet32")
    adapter_hidden = int(checkpoint.get("adapter_hidden", 64))
    use_skip_gates = bool(checkpoint.get("use_skip_gates", True))
    use_multiscale_degradation = bool(
        checkpoint.get("use_multiscale_degradation", False)
    )
    model = build_model(
        model_type,
        preset,
        adapter_hidden,
        use_skip_gates=use_skip_gates,
        use_multiscale_degradation=use_multiscale_degradation,
    )
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

    dataset_mode = "val" if args.split == "validation" else "test"
    if args.split == "test":
        print(
            "WARNING: evaluating the held-out test split. Do not use this run "
            "for model selection or tuning.",
            flush=True,
        )
    dataset = CDD11Dataset(
        find_cdd11_root(args.data_root),
        mode=dataset_mode,
        crop_size=0,
        val_fraction=args.val_fraction,
        split_seed=args.seed,
        augment=False,
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
    grouped: Dict[str, List[Tuple[float, float, float, float]]] = defaultdict(list)
    degradation_targets = []
    degradation_probabilities = []
    fp32_fallback = False
    saved_per_type: Dict[str, int] = defaultdict(int)

    for index, batch in enumerate(loader, start=1):
        lq_cpu, gt_cpu = batch["lq"].float(), batch["gt"].float()
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        prediction, auxiliary = tiled_inference(
            model, lq_cpu, device, args.tile, args.overlap, use_amp
        )
        if not torch.isfinite(prediction).all() and use_amp:
            print(
                "WARNING: non-finite AMP output; retrying this and subsequent "
                "samples in FP32.",
                flush=True,
            )
            use_amp = False
            fp32_fallback = True
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            prediction, auxiliary = tiled_inference(
                model, lq_cpu, device, args.tile, args.overlap, False
            )
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(
                f"Non-finite output for {batch['degradation_type'][0]}/"
                f"{batch['scene_id'][0]} even in FP32"
            )
        prediction = prediction.clamp(0, 1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        degradation_type = batch["degradation_type"][0]
        scene_id = batch["scene_id"][0]
        psnr = calculate_psnr(prediction, gt_cpu)
        ssim = calculate_ssim(prediction, gt_cpu)
        input_psnr = calculate_psnr(lq_cpu, gt_cpu)
        input_ssim = calculate_ssim(lq_cpu, gt_cpu)
        metric_values = (input_psnr, input_ssim, psnr, ssim)
        if not all(math.isfinite(value) for value in metric_values):
            raise FloatingPointError(
                f"Non-finite metric for {degradation_type}/{scene_id}: "
                f"{metric_values}"
            )
        grouped[degradation_type].append(metric_values)
        row: Dict[str, object] = {
            "split": args.split,
            "type": degradation_type,
            "scene_id": scene_id,
            "input_psnr": input_psnr,
            "input_ssim": input_ssim,
            "psnr": psnr,
            "ssim": ssim,
            "delta_psnr": psnr - input_psnr,
            "delta_ssim": ssim - input_ssim,
            "latency_ms": latency_ms,
        }
        if "degradation_logits" in auxiliary:
            probabilities = auxiliary["degradation_logits"].sigmoid()[0]
            predicted_labels = probabilities >= 0.5
            target_labels = batch["label"][0] >= 0.5
            degradation_probabilities.append(probabilities.numpy())
            degradation_targets.append(target_labels.numpy())
            is_exact = predicted_labels.eq(target_labels).all().item()
            for name, probability in zip(DEGRADATIONS, probabilities.tolist()):
                row[f"prob_{name}"] = probability
            row["degradation_exact_match"] = bool(is_exact)
            row["gate_mean_abs"] = auxiliary["gate_mean_abs"].item()
        rows.append(row)
        if args.save_images:
            save_image(
                prediction,
                output_dir / "images" / degradation_type / f"{scene_id}.png",
            )
        if (
            args.save_comparisons
            and saved_per_type[degradation_type] < args.max_saved_per_type
        ):
            save_comparison(
                lq_cpu,
                prediction,
                gt_cpu,
                output_dir
                / "comparisons"
                / f"{degradation_type}_{scene_id}.png",
                f"{model_type}/{preset} | {degradation_type}/{scene_id}",
            )
            saved_per_type[degradation_type] += 1
        print(
            f"[{index:03d}/{len(dataset)}] {degradation_type}/{scene_id} "
            f"PSNR={psnr:.3f} SSIM={ssim:.4f} time={latency_ms:.1f}ms"
        )

    per_type = {
        degradation_type: {
            "input_psnr": float(np.mean([item[0] for item in values])),
            "input_ssim": float(np.mean([item[1] for item in values])),
            "psnr": float(np.mean([item[2] for item in values])),
            "ssim": float(np.mean([item[3] for item in values])),
            "count": len(values),
        }
        for degradation_type, values in sorted(grouped.items())
    }
    macro_psnr = float(np.mean([value["psnr"] for value in per_type.values()]))
    macro_ssim = float(np.mean([value["ssim"] for value in per_type.values()]))
    macro_input_psnr = float(
        np.mean([value["input_psnr"] for value in per_type.values()])
    )
    macro_input_ssim = float(
        np.mean([value["input_ssim"] for value in per_type.values()])
    )
    degradation_metrics = (
        multilabel_degradation_metrics(
            np.stack(degradation_targets),
            np.stack(degradation_probabilities),
            DEGRADATIONS,
        )
        if degradation_targets
        else None
    )
    summary = {
        "model_type": model_type,
        "preset": preset,
        "split": args.split,
        "scene_ids": list(dataset.scene_ids),
        "val_fraction": args.val_fraction if args.split == "validation" else None,
        "split_seed": args.seed if args.split == "validation" else None,
        "use_skip_gates": use_skip_gates,
        "use_multiscale_degradation": use_multiscale_degradation,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "samples": len(rows),
        "amp_requested": requested_amp,
        "amp_used": use_amp,
        "fp32_fallback": fp32_fallback,
        "inference_mode": "full_frame" if args.tile <= 0 else "feathered_tiles",
        "tile": args.tile,
        "overlap": args.overlap if args.tile > 0 else None,
        "macro_input_psnr": macro_input_psnr,
        "macro_input_ssim": macro_input_ssim,
        "macro_psnr": macro_psnr,
        "macro_ssim": macro_ssim,
        "macro_delta_psnr": macro_psnr - macro_input_psnr,
        "macro_delta_ssim": macro_ssim - macro_input_ssim,
        "saved_comparisons": int(sum(saved_per_type.values())),
        "degradation_micro_f1": (
            degradation_metrics["micro_f1"] if degradation_metrics else None
        ),
        "degradation_macro_f1": (
            degradation_metrics["macro_f1"] if degradation_metrics else None
        ),
        "degradation_exact_match": (
            degradation_metrics["exact_match"] if degradation_metrics else None
        ),
        "degradation_per_label": (
            degradation_metrics["per_label"] if degradation_metrics else None
        ),
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
