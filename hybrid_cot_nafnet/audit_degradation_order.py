"""Audit a frozen A0-L restorer under counterfactual degradation orders.

The command uses only clean images from the existing scene-disjoint validation
split.  For every scene and realization it replays exactly the same factor
parameters and spatial masks across all permutations of a degradation set.
It never loads CDD-11_test and never updates the model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor, nn

from .datasets import CDD11Dataset, find_cdd11_root
from .evaluate import calculate_psnr, calculate_ssim, load_torch_file, tiled_inference
from .model import build_model, count_parameters
from .project_config import KAGGLE_CDD11_ROOT


CANONICAL_FACTOR_SETS: Mapping[str, Tuple[str, ...]] = {
    "low_haze": ("low", "haze"),
    "low_rain": ("low", "rain"),
    "low_snow": ("low", "snow"),
    "haze_rain": ("rain", "haze"),
    "haze_snow": ("snow", "haze"),
    "low_haze_rain": ("low", "rain", "haze"),
    "low_haze_snow": ("low", "snow", "haze"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Frozen A0-L best.pt")
    parser.add_argument("--data-root", default=str(KAGGLE_CDD11_ROOT))
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/degradation_order_audit",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=20260920)
    parser.add_argument("--realizations", type=int, default=3)
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="Zero uses every validation scene; positive values are for smoke tests.",
    )
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--allow-non-a0-smoke",
        action="store_true",
        help="Allow a baseline other than the exact 20-epoch SIDD32 A0-L for local smoke tests.",
    )
    parser.add_argument(
        "--save-panels",
        type=int,
        default=1,
        help="Number of validation scenes for which one panel per factor set is saved.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--material-delta-db", type=float, default=0.2)
    parser.add_argument("--sign-consistency", type=float, default=0.7)
    return parser.parse_args()


def _stable_seed(base_seed: int, scene_id: str, realization: int) -> int:
    payload = f"{base_seed}:{scene_id}:{realization}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def _read_rgb(path: Path) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _random_scalar(generator: torch.Generator, lower: float, upper: float) -> float:
    value = torch.rand((), generator=generator).item()
    return float(lower + (upper - lower) * value)


def _coarse_field(
    height: int,
    width: int,
    generator: torch.Generator,
    cells: int = 8,
) -> Tensor:
    coarse_height = max(2, math.ceil(height / max(1, width) * cells))
    coarse = torch.rand((1, 1, coarse_height, cells), generator=generator)
    return F.interpolate(
        coarse,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 1.0)[0]


def _rain_layer(
    height: int,
    width: int,
    generator: torch.Generator,
) -> Tuple[Tensor, Dict[str, float]]:
    density = _random_scalar(generator, 0.0010, 0.0025)
    strength = _random_scalar(generator, 0.18, 0.34)
    length = int(2 * round(_random_scalar(generator, 5.0, 9.0)) + 1)
    slope = _random_scalar(generator, -0.32, 0.32)
    impulses = (
        torch.rand((1, 1, height, width), generator=generator) < density
    ).float()
    impulses *= torch.rand((1, 1, height, width), generator=generator) * 0.5 + 0.5
    kernel = torch.zeros((1, 1, length, length), dtype=torch.float32)
    center = length // 2
    for row in range(length):
        column = center + round((row - center) * slope)
        kernel[0, 0, row, max(0, min(length - 1, column))] = 1.0
    streaks = F.conv2d(impulses, kernel, padding=length // 2)
    streaks = F.avg_pool2d(streaks, kernel_size=3, stride=1, padding=1)
    color = torch.tensor([0.85, 0.92, 1.0]).view(3, 1, 1)
    layer = (streaks[0] * strength * color).clamp(0.0, 0.45)
    return layer, {
        "density": density,
        "strength": strength,
        "length": float(length),
        "slope": slope,
    }


def _snow_alpha(
    height: int,
    width: int,
    generator: torch.Generator,
) -> Tuple[Tensor, Dict[str, float]]:
    density = _random_scalar(generator, 0.0008, 0.0020)
    opacity = _random_scalar(generator, 0.45, 0.75)
    seeds = (
        torch.rand((1, 1, height, width), generator=generator) < density
    ).float()
    small = F.max_pool2d(seeds, kernel_size=3, stride=1, padding=1)
    large_seeds = seeds * (
        torch.rand((1, 1, height, width), generator=generator) < 0.18
    ).float()
    large = F.max_pool2d(large_seeds, kernel_size=7, stride=1, padding=3)
    alpha = (0.55 * small + large).clamp(0.0, 1.0)
    alpha = F.avg_pool2d(alpha, kernel_size=3, stride=1, padding=1)
    return (alpha[0] * opacity).clamp(0.0, 0.9), {
        "density": density,
        "opacity": opacity,
    }


def build_realization(clean: Tensor, seed: int) -> Dict[str, object]:
    """Create one replayable set of tensors for all factor permutations."""
    generator = torch.Generator().manual_seed(seed)
    _, height, width = clean.shape
    gray = clean.mean(dim=0, keepdim=True).unsqueeze(0)
    illumination = gray
    for _ in range(3):
        illumination = F.avg_pool2d(
            illumination,
            kernel_size=31,
            stride=1,
            padding=15,
        )
    illumination = illumination[0].clamp(0.12, 1.0)
    gamma = _random_scalar(generator, 2.0, 2.8)
    noise_std = _random_scalar(generator, 0.015, 0.035)
    low_noise = torch.randn(clean.shape, generator=generator) * noise_std

    transmission_center = _random_scalar(generator, 0.48, 0.72)
    transmission_variation = _random_scalar(generator, 0.10, 0.20)
    transmission = (
        transmission_center
        + (_coarse_field(height, width, generator) - 0.5)
        * 2.0
        * transmission_variation
    ).clamp(0.28, 0.90)
    atmospheric_light = torch.tensor(
        [
            _random_scalar(generator, 0.72, 0.92),
            _random_scalar(generator, 0.72, 0.92),
            _random_scalar(generator, 0.72, 0.92),
        ],
        dtype=torch.float32,
    ).view(3, 1, 1)
    rain, rain_parameters = _rain_layer(height, width, generator)
    snow, snow_parameters = _snow_alpha(height, width, generator)
    return {
        "low_illumination": illumination,
        "low_gamma": gamma,
        "low_noise": low_noise,
        "low_noise_std": noise_std,
        "haze_transmission": transmission,
        "haze_transmission_center": transmission_center,
        "haze_transmission_variation": transmission_variation,
        "haze_atmospheric_light": atmospheric_light,
        "rain_layer": rain,
        "rain_parameters": rain_parameters,
        "snow_alpha": snow,
        "snow_parameters": snow_parameters,
    }


def apply_factor(image: Tensor, factor: str, manifest: Mapping[str, object]) -> Tensor:
    if factor == "low":
        illumination = manifest["low_illumination"]
        gamma = float(manifest["low_gamma"])
        noise = manifest["low_noise"]
        assert isinstance(illumination, Tensor) and isinstance(noise, Tensor)
        return (image * illumination.pow(gamma - 1.0) + noise).clamp(0.0, 1.0)
    if factor == "haze":
        transmission = manifest["haze_transmission"]
        atmospheric_light = manifest["haze_atmospheric_light"]
        assert isinstance(transmission, Tensor)
        assert isinstance(atmospheric_light, Tensor)
        return (
            image * transmission + atmospheric_light * (1.0 - transmission)
        ).clamp(0.0, 1.0)
    if factor == "rain":
        layer = manifest["rain_layer"]
        assert isinstance(layer, Tensor)
        return (image + layer).clamp(0.0, 1.0)
    if factor == "snow":
        alpha = manifest["snow_alpha"]
        assert isinstance(alpha, Tensor)
        return (image * (1.0 - alpha) + alpha).clamp(0.0, 1.0)
    raise KeyError(f"Unknown factor: {factor}")


def apply_order(
    clean: Tensor,
    order: Sequence[str],
    manifest: Mapping[str, object],
) -> Tensor:
    result = clean
    for factor in order:
        result = apply_factor(result, factor, manifest)
    return result


def _mse(left: Tensor, right: Tensor) -> float:
    return float((left.float() - right.float()).square().mean().item())


def severity_match(image: Tensor, clean: Tensor, target_mse: float) -> Tuple[Tensor, float]:
    """Scale the degraded residual to match a target MSE after clipping."""
    if target_mse <= 0.0:
        return clean.clone(), 0.0
    residual = image - clean
    if _mse(image, clean) <= 0.0:
        return image.clone(), 1.0

    def candidate(scale: float) -> Tensor:
        return (clean + residual * scale).clamp(0.0, 1.0)

    lower, upper = 0.0, 1.0
    while _mse(candidate(upper), clean) < target_mse and upper < 32.0:
        upper *= 2.0
    for _ in range(40):
        middle = (lower + upper) / 2.0
        if _mse(candidate(middle), clean) < target_mse:
            lower = middle
        else:
            upper = middle
    scale = (lower + upper) / 2.0
    return candidate(scale), scale


def load_a0_model(
    checkpoint_path: str | Path,
    device: torch.device,
    allow_non_a0_smoke: bool = False,
) -> Tuple[nn.Module, dict]:
    checkpoint = load_torch_file(checkpoint_path)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError("Expected a checkpoint created by train_kaggle.py")
    if checkpoint.get("model_type") != "baseline":
        raise RuntimeError(
            "This preregistered audit requires the A0-L baseline checkpoint; "
            f"found model_type={checkpoint.get('model_type')!r}."
        )
    training_args = checkpoint.get("args", {})
    trained_epochs = training_args.get("epochs") if isinstance(training_args, dict) else None
    if not allow_non_a0_smoke and (
        checkpoint.get("preset") != "sidd32" or trained_epochs != 20
    ):
        raise RuntimeError(
            "The scientific audit requires the exact SIDD32 20-epoch A0-L "
            f"configuration; found preset={checkpoint.get('preset')!r}, "
            f"epochs={trained_epochs!r}. Use --allow-non-a0-smoke only for a "
            "non-scientific local smoke test."
        )
    model = build_model(
        "baseline",
        checkpoint.get("preset", "sidd32"),
        int(checkpoint.get("adapter_hidden", 64)),
        use_skip_gates=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def evaluate_image(
    model: nn.Module,
    degraded: Tensor,
    clean: Tensor,
    device: torch.device,
    tile: int,
    overlap: int,
    use_amp: bool,
) -> Tuple[Dict[str, float], Tensor, bool]:
    input_batch = degraded.unsqueeze(0).float()
    clean_batch = clean.unsqueeze(0).float()
    started = time.perf_counter()
    prediction, _ = tiled_inference(
        model,
        input_batch,
        device,
        tile,
        overlap,
        use_amp,
    )
    fallback = False
    if not torch.isfinite(prediction).all() and use_amp:
        prediction, _ = tiled_inference(
            model,
            input_batch,
            device,
            tile,
            overlap,
            False,
        )
        fallback = True
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("A0-L produced non-finite values even in FP32")
    prediction = prediction.clamp(0.0, 1.0)
    latency_ms = (time.perf_counter() - started) * 1000.0
    input_psnr = calculate_psnr(input_batch, clean_batch)
    input_ssim = calculate_ssim(input_batch, clean_batch)
    output_psnr = calculate_psnr(prediction, clean_batch)
    output_ssim = calculate_ssim(prediction, clean_batch)
    return {
        "input_psnr": input_psnr,
        "input_ssim": input_ssim,
        "output_psnr": output_psnr,
        "output_ssim": output_ssim,
        "gain_psnr": output_psnr - input_psnr,
        "gain_ssim": output_ssim - input_ssim,
        "harm": int(output_psnr < input_psnr),
        "latency_ms": latency_ms,
    }, prediction[0], fallback


def _mean(values: Iterable[float]) -> float:
    sequence = list(values)
    return float(np.mean(sequence)) if sequence else float("nan")


def _bootstrap_interval(
    values: Sequence[float],
    samples: int,
    seed: int,
) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or samples <= 0:
        return float("nan"), float("nan")
    if len(array) == 1:
        return float(array[0]), float(array[0])
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_rows(rows: List[Dict[str, object]], args: argparse.Namespace):
    lookup = {
        (
            str(row["scene_id"]),
            int(row["realization"]),
            str(row["factor_set"]),
            str(row["mode"]),
            str(row["order"]),
        ): row
        for row in rows
    }
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["factor_set"]), str(row["mode"]), str(row["order"]))].append(row)

    order_rows: List[Dict[str, object]] = []
    gate_factor_sets = set()
    for (factor_set, mode, order), items in sorted(grouped.items()):
        canonical = ">".join(CANONICAL_FACTOR_SETS[factor_set])
        deltas_output: List[float] = []
        deltas_gain: List[float] = []
        deltas_input: List[float] = []
        for item in items:
            reference = lookup[
                (
                    str(item["scene_id"]),
                    int(item["realization"]),
                    factor_set,
                    mode,
                    canonical,
                )
            ]
            deltas_output.append(
                float(item["output_psnr"]) - float(reference["output_psnr"])
            )
            deltas_gain.append(float(item["gain_psnr"]) - float(reference["gain_psnr"]))
            deltas_input.append(
                float(item["input_psnr"]) - float(reference["input_psnr"])
            )
        positive_fraction = float(np.mean(np.asarray(deltas_gain) > 0.0))
        negative_fraction = float(np.mean(np.asarray(deltas_gain) < 0.0))
        sign_consistency = max(positive_fraction, negative_fraction)
        ci_low, ci_high = _bootstrap_interval(
            deltas_gain,
            args.bootstrap_samples,
            _stable_seed(args.generation_seed, f"{factor_set}:{mode}:{order}", 0),
        )
        noncanonical = order != canonical
        ci_excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)
        material_consistent = bool(
            mode == "severity_matched"
            and noncanonical
            and abs(_mean(deltas_gain)) >= args.material_delta_db
            and sign_consistency >= args.sign_consistency
            and ci_excludes_zero
        )
        if material_consistent:
            gate_factor_sets.add(factor_set)
        order_rows.append(
            {
                "factor_set": factor_set,
                "mode": mode,
                "order": order,
                "canonical_order": canonical,
                "is_canonical": int(not noncanonical),
                "n": len(items),
                "mean_input_psnr": _mean(float(item["input_psnr"]) for item in items),
                "mean_output_psnr": _mean(float(item["output_psnr"]) for item in items),
                "mean_gain_psnr": _mean(float(item["gain_psnr"]) for item in items),
                "harm_rate": _mean(float(item["harm"]) for item in items),
                "mean_delta_input_psnr_vs_canonical": _mean(deltas_input),
                "mean_delta_output_psnr_vs_canonical": _mean(deltas_output),
                "mean_delta_gain_psnr_vs_canonical": _mean(deltas_gain),
                "delta_gain_ci95_low": ci_low,
                "delta_gain_ci95_high": ci_high,
                "delta_gain_ci95_excludes_zero": int(ci_excludes_zero),
                "delta_gain_sign_consistency": sign_consistency,
                "material_consistent_order_effect": int(material_consistent),
            }
        )

    ranges: Dict[str, List[float]] = defaultdict(list)
    sample_groups: Dict[Tuple[str, int, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        sample_groups[
            (
                str(row["scene_id"]),
                int(row["realization"]),
                str(row["factor_set"]),
                str(row["mode"]),
            )
        ].append(row)
    for (_, _, _, mode), items in sample_groups.items():
        output_values = [float(item["output_psnr"]) for item in items]
        gain_values = [float(item["gain_psnr"]) for item in items]
        ranges[f"{mode}_output_psnr_range"].append(max(output_values) - min(output_values))
        ranges[f"{mode}_gain_psnr_range"].append(max(gain_values) - min(gain_values))

    return order_rows, {
        "prespecified_gate": {
            "material_delta_db": args.material_delta_db,
            "minimum_sign_consistency": args.sign_consistency,
            "requires_bootstrap_ci95_to_exclude_zero": True,
            "minimum_affected_factor_sets": 2,
            "affected_factor_sets": sorted(gate_factor_sets),
            "passes": len(gate_factor_sets) >= 2,
            "interpretation": (
                "Passing supports a systematic order-sensitivity limitation; it does "
                "not validate a new method or establish novelty."
            ),
        },
        "mean_permutation_ranges": {
            name: _mean(values) for name, values in sorted(ranges.items())
        },
    }


def _tensor_to_pil(tensor: Tensor, width: int = 320) -> Image.Image:
    array = (
        tensor.detach().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)
    image = Image.fromarray(array)
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def save_panel(
    path: Path,
    factor_set: str,
    clean: Tensor,
    examples: Sequence[Tuple[str, Tensor, Tensor]],
) -> None:
    if not examples:
        return
    clean_image = _tensor_to_pil(clean)
    width, image_height = clean_image.size
    header = 24
    canvas = Image.new(
        "RGB",
        (width * 3, (image_height + header) * len(examples)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row, (order, degraded, restored) in enumerate(examples):
        top = row * (image_height + header)
        draw.text((5, top + 5), f"{factor_set}: {order}", fill="black")
        for column, (label, image) in enumerate(
            (("Input", degraded), ("Restored", restored), ("Clean", clean))
        ):
            left = column * width
            draw.text((left + width - 70, top + 5), label, fill="black")
            canvas.paste(_tensor_to_pil(image, width), (left, top + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.realizations <= 0:
        raise ValueError("--realizations must be positive")
    if args.max_scenes < 0 or args.save_panels < 0:
        raise ValueError("--max-scenes and --save-panels must be non-negative")
    if args.tile > 0 and args.tile % 16:
        raise ValueError("--tile must be zero or divisible by 16")
    if not 0.5 <= args.sign_consistency <= 1.0:
        raise ValueError("--sign-consistency must be in [0.5, 1]")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cpu" and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; use a Kaggle GPU or pass --allow-cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    model, checkpoint = load_a0_model(
        args.checkpoint,
        device,
        allow_non_a0_smoke=args.allow_non_a0_smoke,
    )
    root = find_cdd11_root(args.data_root)
    dataset = CDD11Dataset(
        root,
        mode="val",
        crop_size=0,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        augment=False,
        degradation_types=("low",),
    )
    scene_ids = list(dataset.scene_ids)
    if args.max_scenes:
        scene_ids = scene_ids[: args.max_scenes]
    clear_dir = root / "CDD-11_train" / "clear"
    clear_paths = {
        path.stem: path
        for path in clear_dir.iterdir()
        if path.is_file() and path.stem in scene_ids
    }
    missing = sorted(set(scene_ids).difference(clear_paths))
    if missing:
        raise RuntimeError(f"Missing clean validation images: {missing}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    fp32_fallback = False
    total_started = time.perf_counter()
    for scene_index, scene_id in enumerate(scene_ids):
        clean = _read_rgb(clear_paths[scene_id])
        for realization in range(args.realizations):
            sample_seed = _stable_seed(args.generation_seed, scene_id, realization)
            manifest = build_realization(clean, sample_seed)
            for factor_set, canonical_order in CANONICAL_FACTOR_SETS.items():
                permutations = [canonical_order] + sorted(
                    order
                    for order in itertools.permutations(canonical_order)
                    if order != canonical_order
                )
                canonical_image = apply_order(clean, canonical_order, manifest)
                target_mse = _mse(canonical_image, clean)
                raw_cache: Dict[Tuple[str, ...], Tuple[Dict[str, float], Tensor]] = {}
                panel_examples: List[Tuple[str, Tensor, Tensor]] = []
                for order in permutations:
                    degraded = apply_order(clean, order, manifest)
                    metrics, prediction, fallback = evaluate_image(
                        model,
                        degraded,
                        clean,
                        device,
                        args.tile,
                        args.overlap,
                        use_amp and not fp32_fallback,
                    )
                    fp32_fallback = fp32_fallback or fallback
                    raw_cache[order] = (metrics, prediction)
                    order_name = ">".join(order)
                    rows.append(
                        {
                            "scene_id": scene_id,
                            "realization": realization,
                            "generation_seed": sample_seed,
                            "factor_set": factor_set,
                            "mode": "raw",
                            "order": order_name,
                            "is_canonical": int(order == canonical_order),
                            "severity_scale": 1.0,
                            "target_mse": target_mse,
                            **metrics,
                        }
                    )
                    if scene_index < args.save_panels and realization == 0:
                        panel_examples.append((order_name, degraded, prediction))

                canonical_metrics, _ = raw_cache[canonical_order]
                rows.append(
                    {
                        "scene_id": scene_id,
                        "realization": realization,
                        "generation_seed": sample_seed,
                        "factor_set": factor_set,
                        "mode": "severity_matched",
                        "order": ">".join(canonical_order),
                        "is_canonical": 1,
                        "severity_scale": 1.0,
                        "target_mse": target_mse,
                        **canonical_metrics,
                    }
                )
                for order in permutations[1:]:
                    raw_image = apply_order(clean, order, manifest)
                    matched_image, scale = severity_match(raw_image, clean, target_mse)
                    metrics, _, fallback = evaluate_image(
                        model,
                        matched_image,
                        clean,
                        device,
                        args.tile,
                        args.overlap,
                        use_amp and not fp32_fallback,
                    )
                    fp32_fallback = fp32_fallback or fallback
                    rows.append(
                        {
                            "scene_id": scene_id,
                            "realization": realization,
                            "generation_seed": sample_seed,
                            "factor_set": factor_set,
                            "mode": "severity_matched",
                            "order": ">".join(order),
                            "is_canonical": 0,
                            "severity_scale": scale,
                            "target_mse": target_mse,
                            **metrics,
                        }
                    )
                if panel_examples:
                    save_panel(
                        output_dir
                        / "panels"
                        / f"{scene_id}_{factor_set}_realization{realization}.jpg",
                        factor_set,
                        clean,
                        panel_examples,
                    )
        print(
            f"Completed scene {scene_index + 1}/{len(scene_ids)}: {scene_id}",
            flush=True,
        )

    order_rows, aggregate = summarize_rows(rows, args)
    _write_csv(output_dir / "per_sample.csv", rows)
    _write_csv(output_dir / "per_order.csv", order_rows)
    elapsed_seconds = time.perf_counter() - total_started
    protocol = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": _file_sha256(args.checkpoint),
        "checkpoint_epoch_index": checkpoint.get("epoch"),
        "model_type": checkpoint.get("model_type"),
        "preset": checkpoint.get("preset"),
        "parameters": count_parameters(model),
        "data_root": str(root.resolve()),
        "split": "CDD-11_train scene-disjoint validation subset only",
        "test_split_loaded": False,
        "exploratory_split_warning": (
            "A0-L checkpoint selection used this validation split. This audit is for "
            "hypothesis discovery, not an unbiased final performance claim."
        ),
        "scene_ids": scene_ids,
        "realizations_per_scene": args.realizations,
        "generation_seed": args.generation_seed,
        "canonical_orders": {
            name: list(order) for name, order in CANONICAL_FACTOR_SETS.items()
        },
        "controlled_operator_ranges": {
            "low": {"gamma": [2.0, 2.8], "noise_std": [0.015, 0.035]},
            "haze": {
                "transmission_center": [0.48, 0.72],
                "transmission_variation": [0.10, 0.20],
                "atmospheric_light_per_channel": [0.72, 0.92],
            },
            "rain": {"density": [0.0010, 0.0025], "strength": [0.18, 0.34]},
            "snow": {"density": [0.0008, 0.0020], "opacity": [0.45, 0.75]},
        },
        "same_realization_policy": (
            "Every permutation for one scene/realization reuses identical low noise, "
            "illumination, haze transmission/airlight, rain layer, and snow alpha."
        ),
        "severity_control": (
            "Secondary rows scale each noncanonical degraded residual around the clean "
            "image until its post-clipping MSE matches the canonical-order input MSE."
        ),
        "synthetic_scope_warning": (
            "The controlled operators approximate OneRestore factors but are not the "
            "original CDD-11 generator. Canonical and counterfactual orders therefore "
            "share an out-of-generator distribution shift; conclusions concern relative "
            "order sensitivity, not absolute CDD-11 performance."
        ),
        "amp_requested": bool(args.amp),
        "fp32_fallback": fp32_fallback,
        "tile": args.tile,
        "overlap": args.overlap,
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "elapsed_seconds": elapsed_seconds,
        "args": vars(args),
    }
    summary = {
        **aggregate,
        "sample_rows": len(rows),
        "order_summary_rows": len(order_rows),
        "mean_latency_ms": _mean(float(row["latency_ms"]) for row in rows),
        "protocol_file": "protocol.json",
        "per_sample_file": "per_sample.csv",
        "per_order_file": "per_order.csv",
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
