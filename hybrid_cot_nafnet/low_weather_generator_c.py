"""Pinned Albumentations renderer for the S2d mechanism-holdout audit.

The operator order is fixed to low-light -> precipitation -> fog.  A weather
realization is sampled once per scene and replayed on the clean, weak-low and
strong-low inputs so that the factorial comparisons do not mix different
weather draws.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import albumentations as A
except ImportError:  # Keep protocol-only tooling usable without the optional pin.
    A = None


GENERATOR_VERSION = "cvpr2027-generator-c-albumentations-v1"
ALBUMENTATIONS_VERSION = "2.0.8"
UPSTREAM_REVISION = "4d2cf04b6635663275a747333754410ef255e54c"
CONDITIONS = (
    "low_weak",
    "low_strong",
    "rain_haze",
    "snow_haze",
    "low_weak_rain_haze",
    "low_strong_rain_haze",
    "low_weak_snow_haze",
    "low_strong_snow_haze",
)


@dataclass(frozen=True)
class ParametersC:
    """Pre-calibration F1 parameters; F2 must freeze calibrated replacements."""

    weak_gamma: float = 140.0
    strong_gamma: float = 220.0
    rain_slant_min: float = -12.0
    rain_slant_max: float = 12.0
    rain_blur: int = 3
    rain_brightness: float = 0.90
    snow_brightness: float = 1.8
    snow_point_min: float = 0.05
    snow_point_max: float = 0.15
    fog_alpha: float = 0.06
    fog_min: float = 0.20
    fog_max: float = 0.40


DEFAULT_PARAMETERS = ParametersC()


@dataclass(frozen=True)
class RealizationC:
    scene_id: str
    image_shape: tuple[int, int, int]
    seeds: dict[str, int]
    rain_haze_replay: dict[str, Any]
    snow_haze_replay: dict[str, Any]
    replay_sha256: dict[str, str]


def _require_upstream() -> None:
    if A is None:
        raise RuntimeError(
            "Mechanism C requires albumentations==2.0.8; install "
            "requirements_s2d_full_audit.txt"
        )
    if A.__version__ != ALBUMENTATIONS_VERSION:
        raise RuntimeError(
            f"Mechanism C is pinned to albumentations=={ALBUMENTATIONS_VERSION}, "
            f"found {A.__version__}"
        )


def _validate_clean(clean_rgb_u8: np.ndarray) -> None:
    if (
        not isinstance(clean_rgb_u8, np.ndarray)
        or clean_rgb_u8.dtype != np.uint8
        or clean_rgb_u8.ndim != 3
        or clean_rgb_u8.shape[-1] != 3
        or min(clean_rgb_u8.shape[:2], default=0) <= 0
    ):
        raise ValueError("clean_rgb_u8 must be a non-empty HxWx3 uint8 array")


def stable_seed(scene_id: str, stream: str) -> int:
    payload = f"{GENERATOR_VERSION}:{scene_id}:{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "array_shape": list(contiguous.shape),
            "array_dtype": str(contiguous.dtype),
            "array_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"object_type": type(value).__qualname__, "repr": repr(value)}


def _replay_hash(replay: dict[str, Any]) -> str:
    payload = json.dumps(
        _jsonable(replay), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _weather_transform(weather: str):
    _require_upstream()
    p = DEFAULT_PARAMETERS
    fog = A.RandomFog(
        alpha_coef=p.fog_alpha,
        fog_coef_range=(p.fog_min, p.fog_max),
        p=1.0,
    )
    if weather == "rain_haze":
        precipitation = A.RandomRain(
            slant_range=(p.rain_slant_min, p.rain_slant_max),
            drop_length=None,
            drop_width=1,
            drop_color=(210, 210, 210),
            blur_value=p.rain_blur,
            brightness_coefficient=p.rain_brightness,
            rain_type="default",
            p=1.0,
        )
    elif weather == "snow_haze":
        precipitation = A.RandomSnow(
            brightness_coeff=p.snow_brightness,
            snow_point_range=(p.snow_point_min, p.snow_point_max),
            method="texture",
            p=1.0,
        )
    else:
        raise ValueError(f"Unknown weather {weather!r}")
    return A.ReplayCompose([precipitation, fog], p=1.0)


def _sample_weather(clean_rgb_u8: np.ndarray, scene_id: str, weather: str):
    transform = _weather_transform(weather)
    seed = stable_seed(scene_id, weather)
    transform.set_random_seed(seed)
    sampled = transform(image=clean_rgb_u8)
    return sampled["replay"], seed


def build_realization(clean_rgb_u8: np.ndarray, scene_id: str) -> RealizationC:
    """Sample the two weather streams exactly once for a clean scene."""
    _require_upstream()
    _validate_clean(clean_rgb_u8)
    if not scene_id:
        raise ValueError("scene_id must be non-empty")
    rain_replay, rain_seed = _sample_weather(clean_rgb_u8, scene_id, "rain_haze")
    snow_replay, snow_seed = _sample_weather(clean_rgb_u8, scene_id, "snow_haze")
    return RealizationC(
        scene_id=scene_id,
        image_shape=tuple(clean_rgb_u8.shape),
        seeds={"rain_haze": rain_seed, "snow_haze": snow_seed},
        rain_haze_replay=rain_replay,
        snow_haze_replay=snow_replay,
        replay_sha256={
            "rain_haze": _replay_hash(rain_replay),
            "snow_haze": _replay_hash(snow_replay),
        },
    )


def _apply_low(clean_rgb_u8: np.ndarray, gamma: float, seed: int) -> np.ndarray:
    transform = A.Compose(
        [A.RandomGamma(gamma_limit=(gamma, gamma), p=1.0)],
        p=1.0,
        seed=seed,
        strict=True,
    )
    return transform(image=clean_rgb_u8)["image"]


def _replay_weather(replay: dict[str, Any], image: np.ndarray) -> np.ndarray:
    # ReplayCompose annotates reconstructed transforms, so isolate every replay.
    return A.ReplayCompose.replay(copy.deepcopy(replay), image=image)["image"]


def make_views(
    clean_rgb_u8: np.ndarray,
    realization: RealizationC,
) -> dict[str, np.ndarray]:
    """Return the locked eight-condition factorial grid in canonical order."""
    _require_upstream()
    _validate_clean(clean_rgb_u8)
    if tuple(clean_rgb_u8.shape) != realization.image_shape:
        raise ValueError("Clean image shape does not match the sampled realization")

    p = DEFAULT_PARAMETERS
    low_weak = _apply_low(
        clean_rgb_u8, p.weak_gamma, stable_seed(realization.scene_id, "low_weak")
    )
    low_strong = _apply_low(
        clean_rgb_u8, p.strong_gamma, stable_seed(realization.scene_id, "low_strong")
    )
    rain = realization.rain_haze_replay
    snow = realization.snow_haze_replay
    views = {
        "low_weak": low_weak,
        "low_strong": low_strong,
        "rain_haze": _replay_weather(rain, clean_rgb_u8),
        "snow_haze": _replay_weather(snow, clean_rgb_u8),
        "low_weak_rain_haze": _replay_weather(rain, low_weak),
        "low_strong_rain_haze": _replay_weather(rain, low_strong),
        "low_weak_snow_haze": _replay_weather(snow, low_weak),
        "low_strong_snow_haze": _replay_weather(snow, low_strong),
    }
    if tuple(views) != CONDITIONS:
        raise AssertionError("Mechanism-C condition order changed")
    return views


def realization_metadata(realization: RealizationC) -> dict[str, Any]:
    return {
        "version": GENERATOR_VERSION,
        "albumentations_version": ALBUMENTATIONS_VERSION,
        "upstream_revision": UPSTREAM_REVISION,
        "scene": realization.scene_id,
        "image_shape": list(realization.image_shape),
        "seeds": realization.seeds,
        "replay_sha256": realization.replay_sha256,
        "operator_order": ["low", "precipitation", "fog"],
        "parameters": DEFAULT_PARAMETERS.__dict__,
        "severity_status": "UNCALIBRATED_F1_ONLY",
    }
