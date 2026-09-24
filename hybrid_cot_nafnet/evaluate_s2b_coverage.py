"""Evaluate an R0/R1 checkpoint on locked S2b original or generated probes."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets.cdd11 import CDD11_TYPES
from .datasets.s2b_coverage import (
    A_PROBE_TYPES,
    GENERATED_TYPES,
    S2BCoverageDataset,
    sha256_file,
)
from .evaluate import (
    calculate_psnr,
    calculate_ssim,
    estimate_conv_linear_macs,
    load_torch_file,
    tiled_inference,
)
from .model import build_model, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probe", choices=("original", "generator_b", "generator_a"), required=True)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def probe_types(name: str):
    return {
        "original": CDD11_TYPES,
        "generator_b": GENERATED_TYPES,
        "generator_a": A_PROBE_TYPES,
    }[name]


def mean(values):
    return float(np.mean(values, dtype=np.float64))


def main() -> None:
    args = parse_args()
    if args.tile > 0 and args.tile % 16:
        raise ValueError("tile must be divisible by 16")
    if args.overlap < 0 or (args.tile > 0 and args.overlap >= args.tile):
        raise ValueError("invalid overlap")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA unavailable; pass --allow-cpu only for a smoke test")
    use_amp = bool(args.amp and device.type == "cuda")
    checkpoint = load_torch_file(args.checkpoint)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError("Expected a train_kaggle checkpoint")
    model = build_model(
        checkpoint.get("model_type", "baseline"),
        checkpoint.get("preset", "sidd32"),
        int(checkpoint.get("adapter_hidden", 64)),
        use_skip_gates=bool(checkpoint.get("use_skip_gates", False)),
        use_multiscale_degradation=bool(checkpoint.get("use_multiscale_degradation", False)),
        order_hidden=int(checkpoint.get("order_hidden", 32)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    types = probe_types(args.probe)
    dataset = S2BCoverageDataset(
        args.data_root,
        args.manifest,
        args.cache_root,
        fold=args.fold,
        split="validation",
        arm="r0" if args.probe == "original" else "r1",
        crop_size=0,
        augment=False,
        degradation_types=types,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        lq, gt = batch["lq"], batch["gt"]
        started = time.perf_counter()
        prediction, _ = tiled_inference(
            model, lq, device, args.tile, args.overlap, use_amp
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        prediction = prediction.clamp(0, 1)
        if not torch.isfinite(prediction).all():
            raise RuntimeError("Non-finite S2b prediction")
        inner = 32 if min(gt.shape[-2:]) > 64 else 0
        pred_inner = prediction[..., inner:-inner, inner:-inner] if inner else prediction
        gt_inner = gt[..., inner:-inner, inner:-inner] if inner else gt
        lq_inner = lq[..., inner:-inner, inner:-inner] if inner else lq
        rows.append(
            {
                "fold": args.fold,
                "scene": batch["scene_id"][0],
                "degradation_type": batch["degradation_type"][0],
                "psnr": calculate_psnr(prediction, gt),
                "ssim": calculate_ssim(prediction, gt),
                "input_psnr": calculate_psnr(lq, gt),
                "input_ssim": calculate_ssim(lq, gt),
                "inner32_psnr": calculate_psnr(pred_inner, gt_inner),
                "inner32_ssim": calculate_ssim(pred_inner, gt_inner),
                "inner32_input_psnr": calculate_psnr(lq_inner, gt_inner),
                "seconds": seconds,
            }
        )
    fieldnames = list(rows[0])
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["degradation_type"]].append(row)
    per_type = {
        name: {
            "count": len(items),
            **{
                key: mean([float(item[key]) for item in items])
                for key in ("psnr", "ssim", "input_psnr", "input_ssim", "inner32_psnr")
            },
        }
        for name, items in sorted(by_type.items())
    }
    counts = count_parameters(model)
    summary = {
        "status": "complete",
        "protocol": "s2b-coverage-null-cv5-v2",
        "fold": args.fold,
        "probe": args.probe,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "split_manifest_sha256": sha256_file(Path(args.manifest)),
        "scene_count": len(dataset.scene_ids),
        "sample_count": len(rows),
        "types": list(types),
        "macro_psnr": mean([item["psnr"] for item in per_type.values()]),
        "macro_ssim": mean([item["ssim"] for item in per_type.values()]),
        "per_type": per_type,
        "parameters": counts,
        "macs_256": estimate_conv_linear_macs(model, device, use_amp),
        "mean_latency_seconds": mean([row["seconds"] for row in rows]),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        ),
        "amp_used": use_amp,
        "tile": args.tile,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
