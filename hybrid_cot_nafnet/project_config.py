"""Stable Kaggle paths and official checkpoint-to-topology mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


KAGGLE_CDD11_ROOT = Path(
    "/kaggle/input/datasets/mintesnotfikir/cdd-11-30"
)
KAGGLE_PRETRAINED_ROOT = Path(
    "/kaggle/input/datasets/hoangkhanhtung/nafnetmodel"
)

PRETRAINED_FILENAMES: Dict[str, str] = {
    "gopro32": "NAFNet-GoPro-width32.pth",
    "gopro64": "NAFNet-GoPro-width64.pth",
    "sidd32": "NAFNet-SIDD-width32.pth",
    "sidd64": "NAFNet-SIDD-width64.pth",
    "nafnet32": "NAFNet-GoPro-width32.pth",
}


def pretrained_path_for_preset(preset: str) -> Path | None:
    """Return the exact Kaggle checkpoint path for a supported official preset."""
    filename = PRETRAINED_FILENAMES.get(preset)
    return KAGGLE_PRETRAINED_ROOT / filename if filename else None
