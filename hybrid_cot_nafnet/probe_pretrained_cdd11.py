"""Measure official NAFNet checkpoints zero-shot on a CDD-11-30 split."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import CDD11Dataset, find_cdd11_root
from .evaluate import calculate_psnr, calculate_ssim, tiled_inference
from .model import build_model, count_parameters
from .project_config import (
    KAGGLE_CDD11_ROOT,
    KAGGLE_PRETRAINED_ROOT,
    PRETRAINED_FILENAMES,
)
from .train_kaggle import load_compatible_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(KAGGLE_CDD11_ROOT))
    parser.add_argument("--pretrained-root", default=str(KAGGLE_PRETRAINED_ROOT))
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/pretrained_cdd11_validation_probe",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Use validation for model selection; reserve test for final reporting.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=("gopro32", "gopro64", "sidd32", "sidd64"),
        default=("gopro32", "gopro64", "sidd32", "sidd64"),
    )
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tile > 0 and args.tile % 16:
        raise ValueError("tile must be divisible by 16")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("Enable a Kaggle GPU or pass --allow-cpu for diagnostics")
    use_amp = bool(args.amp and device.type == "cuda")

    dataset_mode = "val" if args.split == "validation" else "test"
    if args.split == "test":
        print(
            "WARNING: probing the held-out test split. Do not use these metrics "
            "to select a checkpoint or tune hyperparameters.",
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
        persistent_workers=args.num_workers > 0,
    )
    pretrained_root = Path(args.pretrained_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    summaries: Dict[str, object] = {}

    for preset in args.presets:
        checkpoint_path = pretrained_root / PRETRAINED_FILENAMES[preset]
        model = build_model("baseline", preset).to(device).eval()
        load_report = load_compatible_weights(model, checkpoint_path)
        if load_report["backbone_missing"]:
            raise RuntimeError(f"Incompatible {preset} checkpoint: {load_report}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        grouped: Dict[str, List[Tuple[float, float, float, float]]] = defaultdict(list)
        latencies: List[float] = []
        seen = 0
        preset_use_amp = use_amp
        fp32_fallback = False
        for batch in loader:
            if args.max_samples > 0 and seen >= args.max_samples:
                break
            lq, gt = batch["lq"].float(), batch["gt"].float()
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            prediction, _ = tiled_inference(
                model, lq, device, args.tile, args.overlap, preset_use_amp
            )
            if not torch.isfinite(prediction).all() and preset_use_amp:
                print(
                    f"WARNING: {preset} produced non-finite output with AMP; "
                    "retrying this and subsequent samples in FP32.",
                    flush=True,
                )
                preset_use_amp = False
                fp32_fallback = True
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                prediction, _ = tiled_inference(
                    model, lq, device, args.tile, args.overlap, False
                )
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(
                    f"{preset} produced non-finite output even in FP32"
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000.0
            prediction = prediction.clamp(0, 1)
            degradation_type = batch["degradation_type"][0]
            input_psnr = calculate_psnr(lq, gt)
            input_ssim = calculate_ssim(lq, gt)
            psnr = calculate_psnr(prediction, gt)
            ssim = calculate_ssim(prediction, gt)
            metrics = (input_psnr, input_ssim, psnr, ssim)
            if not all(math.isfinite(value) for value in metrics):
                raise FloatingPointError(
                    f"Non-finite metric for {preset}/{degradation_type}/"
                    f"{batch['scene_id'][0]}: {metrics}"
                )
            grouped[degradation_type].append(metrics)
            latencies.append(latency_ms)
            rows.append(
                {
                    "split": args.split,
                    "preset": preset,
                    "type": degradation_type,
                    "scene_id": batch["scene_id"][0],
                    "input_psnr": input_psnr,
                    "input_ssim": input_ssim,
                    "psnr": psnr,
                    "ssim": ssim,
                    "delta_psnr": psnr - input_psnr,
                    "delta_ssim": ssim - input_ssim,
                    "latency_ms": latency_ms,
                }
            )
            seen += 1
            print(
                f"{preset} [{seen:03d}/{len(dataset)}] {degradation_type} "
                f"PSNR={psnr:.3f} SSIM={ssim:.4f}",
                flush=True,
            )
        per_type = {
            name: {
                "input_psnr": float(np.mean([value[0] for value in values])),
                "input_ssim": float(np.mean([value[1] for value in values])),
                "psnr": float(np.mean([value[2] for value in values])),
                "ssim": float(np.mean([value[3] for value in values])),
                "count": len(values),
            }
            for name, values in sorted(grouped.items())
        }
        macro_input_psnr = float(
            np.mean([value["input_psnr"] for value in per_type.values()])
        )
        macro_input_ssim = float(
            np.mean([value["input_ssim"] for value in per_type.values()])
        )
        macro_psnr = float(np.mean([value["psnr"] for value in per_type.values()]))
        macro_ssim = float(np.mean([value["ssim"] for value in per_type.values()]))
        summaries[preset] = {
            "split": args.split,
            "scene_ids": list(dataset.scene_ids),
            "val_fraction": args.val_fraction if args.split == "validation" else None,
            "split_seed": args.seed if args.split == "validation" else None,
            "checkpoint": str(checkpoint_path),
            "load_report": load_report,
            "parameters": count_parameters(model),
            "samples": seen,
            "amp_requested": use_amp,
            "amp_used": preset_use_amp,
            "fp32_fallback": fp32_fallback,
            "macro_input_psnr": macro_input_psnr,
            "macro_input_ssim": macro_input_ssim,
            "macro_psnr": macro_psnr,
            "macro_ssim": macro_ssim,
            "macro_delta_psnr": macro_psnr - macro_input_psnr,
            "macro_delta_ssim": macro_ssim - macro_input_ssim,
            "mean_latency_ms": float(np.mean(latencies)),
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated() / 1024**2
                if device.type == "cuda"
                else 0.0
            ),
            "per_type": per_type,
        }
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, ensure_ascii=False)
    print(f"Saved pretrained probe to {output_dir}")


if __name__ == "__main__":
    main()
