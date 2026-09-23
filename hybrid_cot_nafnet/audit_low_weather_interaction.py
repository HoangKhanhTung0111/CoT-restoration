"""Controlled low-light × weather diagnostic on locked CDD-11 discovery scenes.

The four views reuse a single realization manifest. This diagnoses a failure
mode of frozen restorers; it does not train a new method or establish novelty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .audit_degradation_order import _stable_seed, apply_order, build_realization
from .common_failure_audit import load_model, save_json, sha256


WEATHERS = {"rain_haze": ("rain", "haze"), "snow_haze": ("snow", "haze")}
LOW_LEVELS = (0.6, 1.0)


def apply_low(clean: torch.Tensor, manifest: dict, level: float) -> torch.Tensor:
    """Interpolate the existing low operator to a predeclared strength in [0,1]."""
    if not 0 <= level <= 1:
        raise ValueError("Low level must be within [0,1]")
    gain = manifest["low_illumination"].pow(float(manifest["low_gamma"]) - 1)
    return (clean * (1 - level * (1 - gain)) + level * manifest["low_noise"]).clamp(0, 1)


def make_views(clean: torch.Tensor, manifest: dict, weather: str, low_level: float):
    if weather not in WEATHERS:
        raise ValueError(weather)
    order = WEATHERS[weather]
    low = apply_low(clean, manifest, low_level)
    return {
        "clean": clean,
        "low": low,
        "weather": apply_order(clean, order, manifest),
        "combined": apply_order(low, order, manifest),
    }


def quantize_view(view: torch.Tensor):
    """Match the prior CDD-11 8-bit input protocol for both image encoder and restorer."""
    return (view.clamp(0, 1) * 255).round().to(torch.uint8)


def interaction(errors: dict[str, float]):
    return errors["combined"] - errors["low"] - errors["weather"] + errors["clean"]


def relative_error(input_errors: dict[str, float], output_errors: dict[str, float]):
    """Fraction of input MSE remaining after restoration for each degraded view.

    Composite excess is only a descriptive efficiency comparison. It does not
    establish that the restorer causes the loss of recoverable information.
    """
    fractions = {}
    for name in ("low", "weather", "combined"):
        denominator = input_errors[name]
        fractions[name] = (output_errors[name] / denominator if denominator > 1e-6 else None)
    valid = all(value is not None for value in fractions.values())
    return fractions, (fractions["combined"] - max(fractions["low"], fractions["weather"])
                       if valid else None)


def mean_error(pred: np.ndarray, clean: np.ndarray, border: int = 0):
    if border:
        if min(clean.shape[:2]) <= 2 * border:
            raise ValueError("Image too small for border trim")
        pred = pred[border:-border, border:-border]
        clean = clean[border:-border, border:-border]
    return float(np.mean((pred - clean) ** 2, dtype=np.float64))


def _forward(model, view: torch.Tensor, device):
    rgb = quantize_view(view).permute(1, 2, 0).contiguous().cpu().numpy()
    image = Image.fromarray(rgb)
    inp = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255
    height, width = inp.shape[-2:]
    padded = F.pad(inp, (0, (-width) % 8, 0, (-height) % 8), mode="replicate")
    output = model(padded, image)
    if output.shape != padded.shape or not torch.isfinite(output).all():
        raise RuntimeError("Model produced an invalid output")
    raw = output[0, :, :height, :width].permute(1, 2, 0).cpu().numpy()
    return rgb.astype(np.float32) / 255, np.clip(raw, 0, 1)


def evaluate(work: Path, model_name: str, max_scenes: int):
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen-model evaluation requires Kaggle GPU")
    if max_scenes < 0:
        raise ValueError("max-scenes cannot be negative")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20260923)
    manifest_path = work / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = [r for r in manifest["scenes"] if r["partition"] == "discovery"]
    if max_scenes:
        scenes = scenes[:max_scenes]
    if len(scenes) != 78 and not max_scenes:
        raise RuntimeError("Expected exactly 78 locked discovery scenes")
    root = Path(manifest["root"])
    output = work / "interaction" / model_name
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    save_json(output / "status.json", {"status": "running", "model": model_name})
    device = torch.device("cuda:0")
    model = load_model(model_name, work, device)
    with torch.inference_mode():
        for scene_index, record in enumerate(scenes):
            path = root / "clear" / record["file"]
            with Image.open(path) as im:
                clean_u8 = np.asarray(im.convert("RGB")).copy()
            clean = torch.from_numpy(clean_u8).permute(2, 0, 1).float() / 255
            clean_np = clean_u8.astype(np.float32) / 255
            seed = _stable_seed(20260923, record["scene"], 0)
            realization = build_realization(clean, seed)
            # Cache clean and low results; all weather variants share this exact state.
            prediction = {}
            image_error = {}
            gt_error = {}
            for level in LOW_LEVELS:
                for weather in WEATHERS:
                    views = make_views(clean, realization, weather, level)
                    for view_name, view in views.items():
                        cache_key = (view_name, level if view_name in ("low", "combined") else None,
                                     weather if view_name in ("weather", "combined") else None)
                        if cache_key not in prediction:
                            arr, restored = _forward(model, view, device)
                            prediction[cache_key] = restored
                            image_error[cache_key] = {
                                "full": mean_error(arr, clean_np),
                                "inner32": mean_error(arr, clean_np, border=32),
                            }
                            gt_error[cache_key] = {
                                "full": mean_error(restored, clean_np),
                                "inner32": mean_error(restored, clean_np, border=32),
                            }
                            if scene_index == 0:
                                panel_dir = output / "panels"
                                panel_dir.mkdir(exist_ok=True)
                                panel = np.concatenate((arr, restored, clean_np), axis=1)
                                suffix = "_".join(str(x) for x in cache_key if x is not None)
                                Image.fromarray((panel.clip(0, 1) * 255).round().astype(np.uint8)).save(
                                    panel_dir / f"{record['scene']}_{suffix}.png")
                    result = {"scene": record["scene"], "weather": weather,
                              "low_level": level, "seed": seed}
                    keys = {name: (name, level if name in ("low", "combined") else None,
                                   weather if name in ("weather", "combined") else None)
                            for name in views}
                    for region in ("full", "inner32"):
                        input_errors = {name: image_error[key][region] for name, key in keys.items()}
                        output_errors = {name: gt_error[key][region] for name, key in keys.items()}
                        result[region] = {
                            "input_mse": input_errors,
                            "output_mse": output_errors,
                            "input_interaction": interaction(input_errors),
                            "output_interaction": interaction(output_errors),
                        }
                        fractions, excess = relative_error(input_errors, output_errors)
                        result[region]["remaining_error_fraction"] = fractions
                        result[region]["composite_fraction_excess"] = excess
                    rows.append(result)
                    save_json(output / "metrics.json", rows)
                    print(model_name, record["scene"], weather, level,
                          result["full"]["composite_fraction_excess"], flush=True)
    save_json(output / "status.json", {
        "status": "complete", "model": model_name, "scenes": len(scenes),
        "rows": len(rows), "manifest_sha256": sha256(manifest_path),
        "generator": "audit_degradation_order.build_realization + fixed low-before-weather order",
        "low_levels": LOW_LEVELS, "weathers": WEATHERS,
        "protocol": "8bit_rgb_fullframe_clamp01_mse_full_and_inner32_no_training; primary_remaining_error_fraction_descriptive",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--model", choices=("onerestore", "mirage"), required=True)
    parser.add_argument("--max-scenes", type=int, default=0,
                        help="0 = all 78 discovery scenes; small positive value is technical smoke only")
    args = parser.parse_args()
    evaluate(args.work.resolve(), args.model, args.max_scenes)


if __name__ == "__main__":
    main()
