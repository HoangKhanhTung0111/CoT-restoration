"""CDD-11 paired-image loader with scene-level train/validation splitting."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


DEGRADATIONS: Tuple[str, ...] = ("low", "haze", "rain", "snow")
CDD11_TYPES: Tuple[str, ...] = (
    "low",
    "haze",
    "rain",
    "snow",
    "low_haze",
    "low_rain",
    "low_snow",
    "haze_rain",
    "haze_snow",
    "low_haze_rain",
    "low_haze_snow",
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_cdd11_root(path: str | Path) -> Path:
    """Resolve a directory containing CDD-11_train and CDD-11_test."""
    root = Path(path).expanduser()
    if (root / "CDD-11_train").is_dir() and (root / "CDD-11_test").is_dir():
        return root
    if not root.exists():
        raise FileNotFoundError(f"CDD-11 root does not exist: {root}")
    matches = sorted(
        candidate.parent
        for candidate in root.rglob("CDD-11_train")
        if candidate.is_dir() and (candidate.parent / "CDD-11_test").is_dir()
    )
    unique_matches = list(dict.fromkeys(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    if not unique_matches:
        raise FileNotFoundError(
            f"Could not find CDD-11_train and CDD-11_test below {root}"
        )
    raise RuntimeError(
        "Multiple CDD-11 roots found; pass --data-root explicitly: "
        + ", ".join(str(item) for item in unique_matches)
    )


def _image_map(directory: Path) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in result:
                raise RuntimeError(f"Duplicate image stem {path.stem!r} in {directory}")
            result[path.stem] = path
    return result


def _label_for_type(degradation_type: str) -> Tensor:
    tokens = set(degradation_type.split("_"))
    unknown = tokens.difference(DEGRADATIONS)
    if unknown:
        raise ValueError(f"Unknown degradation tokens in {degradation_type!r}: {unknown}")
    return torch.tensor(
        [1.0 if name in tokens else 0.0 for name in DEGRADATIONS],
        dtype=torch.float32,
    )


def _read_rgb(path: Path) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _pad_to_crop(lq: Tensor, gt: Tensor, crop_size: int) -> Tuple[Tensor, Tensor]:
    height, width = lq.shape[-2:]
    pad_h = max(0, crop_size - height)
    pad_w = max(0, crop_size - width)
    if pad_h or pad_w:
        padding = (0, pad_w, 0, pad_h)
        lq = F.pad(lq, padding, mode="replicate")
        gt = F.pad(gt, padding, mode="replicate")
    return lq, gt


def _paired_crop(
    lq: Tensor, gt: Tensor, crop_size: int, random_crop: bool
) -> Tuple[Tensor, Tensor]:
    if lq.shape != gt.shape:
        raise ValueError(f"LQ/GT shapes differ: {tuple(lq.shape)} vs {tuple(gt.shape)}")
    if crop_size <= 0:
        return lq, gt
    lq, gt = _pad_to_crop(lq, gt, crop_size)
    height, width = lq.shape[-2:]
    if random_crop:
        top = random.randint(0, height - crop_size)
        left = random.randint(0, width - crop_size)
    else:
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
    slices = (..., slice(top, top + crop_size), slice(left, left + crop_size))
    return lq[slices], gt[slices]


def _augment(lq: Tensor, gt: Tensor) -> Tuple[Tensor, Tensor]:
    if random.random() < 0.5:
        lq, gt = lq.flip(-1), gt.flip(-1)
    if random.random() < 0.5:
        lq, gt = lq.flip(-2), gt.flip(-2)
    if random.random() < 0.5:
        lq, gt = lq.transpose(-2, -1), gt.transpose(-2, -1)
    return lq.contiguous(), gt.contiguous()


class CDD11Dataset(Dataset):
    """Return paired CDD-11 samples while splitting by clean scene ID.

    ``mode='train'`` and ``mode='val'`` partition IDs from CDD-11_train.
    Consequently, all degradation variants of one scene remain in one split.
    ``mode='test'`` always uses CDD-11_test and never participates in tuning.
    """

    def __init__(
        self,
        root: str | Path,
        mode: str,
        crop_size: int = 256,
        val_fraction: float = 0.2,
        split_seed: int = 42,
        augment: bool = True,
        degradation_types: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        if mode not in {"train", "val", "test"}:
            raise ValueError("mode must be train, val, or test")
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        self.root = find_cdd11_root(root)
        self.mode = mode
        self.crop_size = int(crop_size)
        self.augment = bool(augment and mode == "train")
        split_dir = self.root / ("CDD-11_test" if mode == "test" else "CDD-11_train")
        clear_dir = split_dir / "clear"
        if not clear_dir.is_dir():
            raise FileNotFoundError(f"Missing clean target directory: {clear_dir}")
        clear_images = _image_map(clear_dir)
        if not clear_images:
            raise RuntimeError(f"No clean images found in {clear_dir}")

        selected_ids = sorted(clear_images)
        if mode in {"train", "val"} and val_fraction > 0:
            shuffled_ids = selected_ids.copy()
            random.Random(split_seed).shuffle(shuffled_ids)
            validation_count = max(1, round(len(shuffled_ids) * val_fraction))
            validation_ids = set(shuffled_ids[:validation_count])
            selected_ids = [
                scene_id
                for scene_id in selected_ids
                if (scene_id in validation_ids) == (mode == "val")
            ]

        types = tuple(degradation_types or CDD11_TYPES)
        invalid_types = sorted(set(types).difference(CDD11_TYPES))
        if invalid_types:
            raise ValueError(f"Unsupported CDD-11 types: {invalid_types}")

        samples: List[Tuple[Path, Path, Tensor, str, str]] = []
        missing_pairs: List[str] = []
        for degradation_type in types:
            input_dir = split_dir / degradation_type
            if not input_dir.is_dir():
                raise FileNotFoundError(f"Missing degradation directory: {input_dir}")
            inputs = _image_map(input_dir)
            label = _label_for_type(degradation_type)
            for scene_id in selected_ids:
                if scene_id not in inputs:
                    missing_pairs.append(f"{degradation_type}/{scene_id}")
                    continue
                samples.append(
                    (inputs[scene_id], clear_images[scene_id], label, scene_id, degradation_type)
                )
        if missing_pairs:
            preview = ", ".join(missing_pairs[:10])
            raise RuntimeError(f"Missing {len(missing_pairs)} degraded/clean pairs: {preview}")
        if not samples:
            raise RuntimeError(f"No samples selected for mode={mode}")
        target_sizes: Dict[Path, Tuple[int, int]] = {}
        for lq_path, gt_path, _, _, _ in samples:
            if gt_path not in target_sizes:
                with Image.open(gt_path) as image:
                    target_sizes[gt_path] = image.size
            with Image.open(lq_path) as image:
                lq_size = image.size
            if lq_size != target_sizes[gt_path]:
                raise RuntimeError(
                    f"LQ/GT image sizes differ for {lq_path}: "
                    f"{lq_size} vs {target_sizes[gt_path]}"
                )
        self.scene_ids = tuple(selected_ids)
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        lq_path, gt_path, label, scene_id, degradation_type = self.samples[index]
        lq, gt = _read_rgb(lq_path), _read_rgb(gt_path)
        lq, gt = _paired_crop(
            lq, gt, crop_size=self.crop_size, random_crop=self.mode == "train"
        )
        if self.augment:
            lq, gt = _augment(lq, gt)
        return {
            "lq": lq,
            "gt": gt,
            "label": label.clone(),
            "scene_id": scene_id,
            "degradation_type": degradation_type,
            "lq_path": str(lq_path),
        }

    def summary(self) -> str:
        return (
            f"CDD11Dataset(mode={self.mode}, scenes={len(self.scene_ids)}, "
            f"samples={len(self.samples)}, crop={self.crop_size})"
        )
