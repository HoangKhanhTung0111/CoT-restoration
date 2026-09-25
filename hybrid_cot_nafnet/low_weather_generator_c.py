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
    weak_noise_std: float = 0.0
    strong_noise_std: float = 0.0
    rain_slant_min: float = -12.0
    rain_slant_max: float = 12.0
    rain_drop_length: int | None = None
    rain_drop_color: tuple[int, int, int] = (210, 210, 210)
    rain_blur: int = 3
    rain_brightness: float = 0.90
    rain_type: str = "default"
    snow_brightness: float = 1.8
    snow_point_min: float = 0.05
    snow_point_max: float = 0.15
    snow_method: str = "texture"
    fog_alpha: float = 0.06
    fog_min: float = 0.20
    fog_max: float = 0.40
    rain_fog_alpha: float | None = None
    rain_fog_min: float | None = None
    rain_fog_max: float | None = None
    snow_fog_alpha: float | None = None
    snow_fog_min: float | None = None
    snow_fog_max: float | None = None


DEFAULT_PARAMETERS = ParametersC()


@dataclass(frozen=True)
class RealizationC:
    scene_id: str
    image_shape: tuple[int, int, int]
    seeds: dict[str, int]
    rain_haze_replay: dict[str, Any]
    snow_haze_replay: dict[str, Any]
    replay_sha256: dict[str, str]
    parameters: ParametersC
    seed_variant: int


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


def stable_seed(scene_id: str, stream: str, seed_variant: int = 0) -> int:
    if seed_variant < 0:
        raise ValueError("seed_variant must be nonnegative")
    suffix = "" if seed_variant == 0 else f":candidate-{seed_variant}"
    payload = f"{GENERATOR_VERSION}:{scene_id}:{stream}{suffix}".encode("utf-8")
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


def _weather_transform(weather: str, parameters: ParametersC = DEFAULT_PARAMETERS):
    _require_upstream()
    p = parameters
    if weather == "rain_haze":
        fog_alpha = p.fog_alpha if p.rain_fog_alpha is None else p.rain_fog_alpha
        fog_min = p.fog_min if p.rain_fog_min is None else p.rain_fog_min
        fog_max = p.fog_max if p.rain_fog_max is None else p.rain_fog_max
        precipitation = A.RandomRain(
            slant_range=(p.rain_slant_min, p.rain_slant_max),
            drop_length=p.rain_drop_length,
            drop_width=1,
            drop_color=tuple(p.rain_drop_color),
            blur_value=p.rain_blur,
            brightness_coefficient=p.rain_brightness,
            rain_type=p.rain_type,
            p=1.0,
        )
    elif weather == "snow_haze":
        fog_alpha = p.fog_alpha if p.snow_fog_alpha is None else p.snow_fog_alpha
        fog_min = p.fog_min if p.snow_fog_min is None else p.snow_fog_min
        fog_max = p.fog_max if p.snow_fog_max is None else p.snow_fog_max
        precipitation = A.RandomSnow(
            brightness_coeff=p.snow_brightness,
            snow_point_range=(p.snow_point_min, p.snow_point_max),
            method=p.snow_method,
            p=1.0,
        )
    else:
        raise ValueError(f"Unknown weather {weather!r}")
    fog = A.RandomFog(
        alpha_coef=fog_alpha,
        fog_coef_range=(fog_min, fog_max),
        p=1.0,
    )
    return A.ReplayCompose([precipitation, fog], p=1.0)


def _sample_weather(
    clean_rgb_u8: np.ndarray,
    scene_id: str,
    weather: str,
    parameters: ParametersC = DEFAULT_PARAMETERS,
    seed_variant: int = 0,
):
    transform = _weather_transform(weather, parameters)
    seed = stable_seed(scene_id, weather, seed_variant)
    transform.set_random_seed(seed)
    sampled = transform(image=clean_rgb_u8)
    return sampled["replay"], seed


def build_realization(
    clean_rgb_u8: np.ndarray,
    scene_id: str,
    parameters: ParametersC = DEFAULT_PARAMETERS,
    seed_variant: int = 0,
) -> RealizationC:
    """Sample the two weather streams exactly once for a clean scene."""
    _require_upstream()
    _validate_clean(clean_rgb_u8)
    if not scene_id:
        raise ValueError("scene_id must be non-empty")
    rain_replay, rain_seed = _sample_weather(
        clean_rgb_u8, scene_id, "rain_haze", parameters, seed_variant
    )
    snow_replay, snow_seed = _sample_weather(
        clean_rgb_u8, scene_id, "snow_haze", parameters, seed_variant
    )
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
        parameters=parameters,
        seed_variant=seed_variant,
    )


def _apply_low(
    clean_rgb_u8: np.ndarray,
    gamma: float,
    seed: int,
    noise_std: float = 0.0,
) -> np.ndarray:
    transforms = [A.RandomGamma(gamma_limit=(gamma, gamma), p=1.0)]
    if noise_std > 0:
        transforms.append(
            A.GaussNoise(
                std_range=(noise_std, noise_std),
                mean_range=(0.0, 0.0),
                per_channel=True,
                noise_scale_factor=1.0,
                p=1.0,
            )
        )
    transform = A.Compose(
        transforms,
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

    p = realization.parameters
    low_weak = _apply_low(
        clean_rgb_u8,
        p.weak_gamma,
        stable_seed(realization.scene_id, "low_weak", realization.seed_variant),
        p.weak_noise_std,
    )
    low_strong = _apply_low(
        clean_rgb_u8,
        p.strong_gamma,
        stable_seed(realization.scene_id, "low_strong", realization.seed_variant),
        p.strong_noise_std,
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


def make_factorized_views(
    clean_rgb_u8: np.ndarray,
    scene_id: str,
    low_parameters: ParametersC,
    rain_parameters: ParametersC,
    snow_parameters: ParametersC,
    *,
    low_seed_variant: int,
    rain_seed_variant: int,
    snow_seed_variant: int,
) -> dict[str, np.ndarray]:
    """Build a factorial grid from independently calibrated factor profiles."""
    _require_upstream()
    _validate_clean(clean_rgb_u8)
    low_weak = _apply_low(
        clean_rgb_u8,
        low_parameters.weak_gamma,
        stable_seed(scene_id, "low_weak", low_seed_variant),
        low_parameters.weak_noise_std,
    )
    low_strong = _apply_low(
        clean_rgb_u8,
        low_parameters.strong_gamma,
        stable_seed(scene_id, "low_strong", low_seed_variant),
        low_parameters.strong_noise_std,
    )
    rain, _ = _sample_weather(
        clean_rgb_u8,
        scene_id,
        "rain_haze",
        rain_parameters,
        rain_seed_variant,
    )
    snow, _ = _sample_weather(
        clean_rgb_u8,
        scene_id,
        "snow_haze",
        snow_parameters,
        snow_seed_variant,
    )
    return {
        "low_weak": low_weak,
        "low_strong": low_strong,
        "rain_haze": _replay_weather(rain, clean_rgb_u8),
        "snow_haze": _replay_weather(snow, clean_rgb_u8),
        "low_weak_rain_haze": _replay_weather(rain, low_weak),
        "low_strong_rain_haze": _replay_weather(rain, low_strong),
        "low_weak_snow_haze": _replay_weather(snow, low_weak),
        "low_strong_snow_haze": _replay_weather(snow, low_strong),
    }


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
        "parameters": realization.parameters.__dict__,
        "seed_variant": realization.seed_variant,
        "severity_status": "UNCALIBRATED_F1_ONLY",
    }
