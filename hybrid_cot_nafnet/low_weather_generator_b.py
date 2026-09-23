"""Independent, deterministic low-light × weather stress-test generator.

This is a synthetic diagnostic, not a physically calibrated night-image model.
All random fields are sampled once per scene and replayed across views/models.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


GENERATOR_VERSION = "cvpr2027-generator-b-v1"
STRENGTHS = {"weak": 0.5, "strong": 0.2}
WEATHERS = ("rain_haze", "snow_haze")
LOW_K = 0.001
READ_NOISE = 0.002
PRECIPITATION_COLOR = 0.9
MAX_ALPHA = 0.45


def stable_seed(scene_id: str) -> int:
    digest = hashlib.sha256(f"{GENERATOR_VERSION}:{scene_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    value = np.asarray(srgb, dtype=np.float32)
    if not np.isfinite(value).all() or value.min(initial=0) < 0 or value.max(initial=0) > 1:
        raise ValueError("sRGB values must be finite and within [0, 1]")
    return np.where(value <= 0.04045, value / 12.92,
                    ((value + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    value = np.asarray(linear, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("Linear RGB values must be finite")
    value = np.clip(value, 0, 1)
    return np.where(value <= 0.0031308, 12.92 * value,
                    1.055 * np.power(value, 1 / 2.4) - 0.055).astype(np.float32)


def encode_u8(linear: np.ndarray) -> np.ndarray:
    """Encode linear RGB and quantize once using ties-to-even rounding."""
    return np.rint(linear_to_srgb(linear) * 255.0).astype(np.uint8)


def _field_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    prefix = json.dumps({"shape": contiguous.shape, "dtype": str(contiguous.dtype)},
                        separators=(",", ":")).encode()
    return hashlib.sha256(prefix + contiguous.tobytes()).hexdigest()


@dataclass(frozen=True)
class RealizationB:
    scene_id: str
    seed: int
    epsilon: np.ndarray
    haze_airlight: float
    haze_grid: np.ndarray
    haze_transmission: np.ndarray
    rain_angle_degrees: float
    rain_uniforms: np.ndarray
    snow_uniforms: np.ndarray
    rain_kernel: np.ndarray
    snow_kernel: np.ndarray
    rain_alpha: np.ndarray
    snow_alpha: np.ndarray
    field_hashes: dict[str, str]


def _rain_kernel(height: int, width: int, angle_degrees: float) -> np.ndarray:
    import cv2

    length = max(3, round(0.04 * min(height, width)))
    side = 2 * math.ceil(length / 2) + 1
    center = side // 2
    half = (length - 1) / 2
    angle = math.radians(angle_degrees)
    dx, dy = math.sin(angle), math.cos(angle)
    start = (int(np.rint(center - half * dx)), int(np.rint(center - half * dy)))
    end = (int(np.rint(center + half * dx)), int(np.rint(center + half * dy)))
    kernel = np.zeros((side, side), dtype=np.float32)
    cv2.line(kernel, start, end, color=1.0, thickness=1, lineType=cv2.LINE_8)
    return kernel


def _snow_kernel(height: int, width: int) -> np.ndarray:
    radius = max(1, round(0.006 * min(height, width)))
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return ((xx * xx + yy * yy <= radius * radius).astype(np.float32))


def _filter_impulses(uniforms: np.ndarray, density: float, kernel: np.ndarray) -> np.ndarray:
    import cv2

    impulses = (uniforms < density).astype(np.float32)
    response = cv2.filter2D(impulses, ddepth=cv2.CV_32F, kernel=kernel,
                            borderType=cv2.BORDER_CONSTANT)
    return (MAX_ALPHA * np.clip(response, 0, 1)).astype(np.float32)


def build_realization(clean_rgb_u8: np.ndarray, scene_id: str) -> RealizationB:
    """Sample one NumPy-PCG64 realization with the locked draw order."""
    if clean_rgb_u8.dtype != np.uint8 or clean_rgb_u8.ndim != 3 or clean_rgb_u8.shape[-1] != 3:
        raise ValueError("clean_rgb_u8 must have shape H×W×3 and dtype uint8")
    height, width = clean_rgb_u8.shape[:2]
    if min(height, width) <= 0:
        raise ValueError("Empty image")
    seed = stable_seed(scene_id)
    rng = np.random.Generator(np.random.PCG64(seed))
    # Draw order is part of the protocol. Keep each field independent of apply order.
    epsilon64 = rng.standard_normal((height, width, 3))
    airlight = float(rng.uniform(0.75, 0.95))
    grid64 = rng.uniform(0.0, 1.0, size=(1, 1, 8, 8))
    rain_angle = float(rng.uniform(-20.0, 20.0))
    rain_uniforms64 = rng.uniform(0.0, 1.0, size=(height, width))
    snow_uniforms64 = rng.uniform(0.0, 1.0, size=(height, width))

    rain_kernel = _rain_kernel(height, width, rain_angle)
    snow_kernel = _snow_kernel(height, width)
    rain_uniforms = rain_uniforms64.astype(np.float32)
    snow_uniforms = snow_uniforms64.astype(np.float32)
    rain_alpha = _filter_impulses(rain_uniforms64, 0.002, rain_kernel)
    snow_alpha = _filter_impulses(snow_uniforms64, 0.0005, snow_kernel)

    # Match torch.interpolate's float32 bilinear, half-pixel convention.
    grid = torch.from_numpy(grid64.astype(np.float32))
    t = 0.55 + 0.35 * F.interpolate(grid, size=(height, width), mode="bilinear",
                                    align_corners=False)[0, 0].numpy()
    epsilon = epsilon64.astype(np.float32)
    haze_grid = grid64.astype(np.float32)[0, 0]
    fields: dict[str, np.ndarray] = {
        "epsilon": epsilon, "haze_grid": haze_grid, "transmission": t.astype(np.float32),
        "rain_uniforms": rain_uniforms, "snow_uniforms": snow_uniforms,
        "rain_kernel": rain_kernel, "snow_kernel": snow_kernel,
        "rain_alpha": rain_alpha, "snow_alpha": snow_alpha,
    }
    return RealizationB(
        scene_id=scene_id, seed=seed, epsilon=epsilon, haze_airlight=airlight,
        haze_grid=haze_grid, haze_transmission=t.astype(np.float32),
        rain_angle_degrees=rain_angle,
        rain_uniforms=rain_uniforms, snow_uniforms=snow_uniforms,
        rain_kernel=rain_kernel, snow_kernel=snow_kernel,
        rain_alpha=rain_alpha, snow_alpha=snow_alpha,
        field_hashes={name: _field_hash(value) for name, value in fields.items()},
    )


def transmission(realization: RealizationB, height: int, width: int) -> np.ndarray:
    grid = torch.from_numpy(realization.haze_grid.copy()).view(1, 1, 8, 8)
    return (0.55 + 0.35 * F.interpolate(grid, size=(height, width), mode="bilinear",
                                       align_corners=False)[0, 0].numpy()).astype(np.float32)


def apply_low(linear: np.ndarray, realization: RealizationB, exposure: float,
              *, k: float = LOW_K, sigma_r: float = READ_NOISE) -> np.ndarray:
    if exposure < 0 or k < 0 or sigma_r < 0:
        raise ValueError("Exposure and noise parameters must be nonnegative")
    if linear.shape != realization.epsilon.shape:
        raise ValueError("Linear image and realization shape differ")
    signal = exposure * linear
    std = np.sqrt(k * signal + sigma_r * sigma_r)
    return np.clip(signal + std * realization.epsilon, 0, 1).astype(np.float32)


def apply_weather(linear: np.ndarray, realization: RealizationB, weather: str,
                  *, t_override: float | np.ndarray | None = None,
                  alpha_override: float | np.ndarray | None = None) -> np.ndarray:
    if weather not in WEATHERS:
        raise ValueError(f"Unknown weather {weather!r}")
    height, width = linear.shape[:2]
    t = realization.haze_transmission if t_override is None else np.asarray(t_override)
    alpha = (realization.rain_alpha if weather == "rain_haze" else realization.snow_alpha
             ) if alpha_override is None else np.asarray(alpha_override)
    if np.ndim(t) == 0:
        t = np.full((height, width), t, dtype=np.float32)
    if np.ndim(alpha) == 0:
        alpha = np.full((height, width), alpha, dtype=np.float32)
    if t.shape != (height, width) or alpha.shape != (height, width):
        raise ValueError("Weather fields must match image height and width")
    precipitated = ((1.0 - alpha[..., None]) * linear
                    + alpha[..., None] * PRECIPITATION_COLOR)
    return (t[..., None] * precipitated
            + (1.0 - t[..., None]) * realization.haze_airlight).clip(0, 1).astype(np.float32)


def make_views(clean_rgb_u8: np.ndarray, realization: RealizationB,
               weather: str) -> dict[str, np.ndarray]:
    clean_linear = srgb_to_linear(clean_rgb_u8.astype(np.float32) / 255.0)
    views: dict[str, np.ndarray] = {"clean": clean_rgb_u8.copy()}
    for strength, exposure in STRENGTHS.items():
        low_linear = apply_low(clean_linear, realization, exposure)
        views[f"low_{strength}"] = encode_u8(low_linear)
        views[f"combined_{strength}"] = encode_u8(apply_weather(low_linear, realization, weather))
    views["weather"] = encode_u8(apply_weather(clean_linear, realization, weather))
    return views


def realization_metadata(realization: RealizationB) -> dict[str, Any]:
    return {
        "version": GENERATOR_VERSION, "scene": realization.scene_id, "seed": realization.seed,
        "bit_generator": "numpy.random.PCG64", "draw_order": [
            "epsilon_HxWx3_normal", "airlight_scalar_uniform", "haze_grid_8x8_uniform",
            "rain_angle_uniform", "rain_uniforms_HxW", "snow_uniforms_HxW"],
        "haze_airlight": realization.haze_airlight,
        "rain_angle_degrees": realization.rain_angle_degrees,
        "rain_coverage": float(np.mean(realization.rain_alpha > 0)),
        "snow_coverage": float(np.mean(realization.snow_alpha > 0)),
        "rain_alpha_mean": float(realization.rain_alpha.mean()),
        "snow_alpha_mean": float(realization.snow_alpha.mean()),
        "transmission_min": float(realization.haze_transmission.min()),
        "transmission_max": float(realization.haze_transmission.max()),
        "transmission_mean": float(realization.haze_transmission.mean()),
        "field_sha256": realization.field_hashes,
    }
