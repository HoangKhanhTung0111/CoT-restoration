"""Deterministic clean-to-low+haze data for formation-order controls.

The dataset deliberately exposes only one experimental variable: whether the
low-light or haze operator is applied first.  A and B samples reuse the same
clean scene and realization manifest.  Development splits are made by clean
scene and CDD-11_test is never opened here.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from ..audit_degradation_order import _stable_seed, apply_order, build_realization
from .cdd11 import IMAGE_EXTENSIONS, find_cdd11_root


ORDER_A: Tuple[str, str] = ("low", "haze")
ORDER_B: Tuple[str, str] = ("haze", "low")
ORDER_POLICIES = ("fixed_a", "fixed_b", "balanced", "both")


def _clean_map(directory: Path) -> Dict[str, Path]:
    result = {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if not result:
        raise RuntimeError(f"No clean images found in {directory}")
    return result


def _read_rgb(path: Path) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _crop_and_augment(clean: Tensor, crop_size: int, seed: int) -> Tensor:
    if crop_size <= 0:
        return clean
    height, width = clean.shape[-2:]
    pad_h = max(0, crop_size - height)
    pad_w = max(0, crop_size - width)
    if pad_h or pad_w:
        clean = F.pad(clean, (0, pad_w, 0, pad_h), mode="replicate")
        height, width = clean.shape[-2:]
    generator = torch.Generator().manual_seed(seed)
    top = int(torch.randint(height - crop_size + 1, (), generator=generator).item())
    left = int(torch.randint(width - crop_size + 1, (), generator=generator).item())
    clean = clean[..., top : top + crop_size, left : left + crop_size]
    if torch.rand((), generator=generator).item() < 0.5:
        clean = clean.flip(-1)
    if torch.rand((), generator=generator).item() < 0.5:
        clean = clean.flip(-2)
    if torch.rand((), generator=generator).item() < 0.5:
        clean = clean.transpose(-2, -1)
    return clean.contiguous()


class OrderControlDataset(Dataset):
    """Synthetic low+haze pairs with a controlled formation-order policy."""

    def __init__(
        self,
        root: str | Path,
        mode: str,
        order_policy: str,
        realizations: int,
        crop_size: int = 256,
        val_fraction: float = 0.2,
        split_seed: int = 42,
        generation_seed: int = 20260920,
    ) -> None:
        if mode not in {"train", "val"}:
            raise ValueError("OrderControlDataset mode must be train or val")
        if order_policy not in ORDER_POLICIES:
            raise ValueError(f"Unknown order policy: {order_policy}")
        if mode == "train" and order_policy == "both":
            raise ValueError("Training must use fixed_a, fixed_b, or balanced")
        if mode == "val" and order_policy != "both":
            raise ValueError("Validation must use both orders")
        if realizations <= 0:
            raise ValueError("realizations must be positive")
        if order_policy == "balanced" and realizations % 2:
            raise ValueError("balanced training requires an even realization count")
        if crop_size < 0:
            raise ValueError("crop_size must be non-negative")

        self.root = find_cdd11_root(root)
        self.mode = mode
        self.order_policy = order_policy
        self.realizations = int(realizations)
        self.crop_size = int(crop_size)
        self.generation_seed = int(generation_seed)
        self.epoch = 0

        clear_dir = self.root / "CDD-11_train" / "clear"
        clean_images = _clean_map(clear_dir)
        shuffled = sorted(clean_images)
        random.Random(split_seed).shuffle(shuffled)
        validation_count = max(1, round(len(shuffled) * val_fraction))
        validation_ids = set(shuffled[:validation_count])
        self.scene_ids = tuple(
            scene_id
            for scene_id in sorted(clean_images)
            if (scene_id in validation_ids) == (mode == "val")
        )
        if not self.scene_ids:
            raise RuntimeError(f"No clean scenes selected for mode={mode}")
        self.clean_paths = {scene_id: clean_images[scene_id] for scene_id in self.scene_ids}
        orders_per_realization = 2 if order_policy == "both" else 1
        self.samples: List[Tuple[str, int, int]] = [
            (scene_id, realization, order_slot)
            for scene_id in self.scene_ids
            for realization in range(self.realizations)
            for order_slot in range(orders_per_realization)
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def _order(self, realization: int, order_slot: int) -> Tuple[str, str]:
        if self.order_policy == "fixed_a":
            return ORDER_A
        if self.order_policy == "fixed_b":
            return ORDER_B
        if self.order_policy == "balanced":
            return ORDER_A if realization % 2 == 0 else ORDER_B
        return ORDER_A if order_slot == 0 else ORDER_B

    def __getitem__(self, index: int):
        scene_id, realization, order_slot = self.samples[index]
        # Validation is frozen. Training receives a new, but exactly replayable,
        # realization each epoch; all three policies share the same seed schedule.
        epoch_offset = self.epoch * self.realizations if self.mode == "train" else 0
        realization_key = epoch_offset + realization
        sample_seed = _stable_seed(
            self.generation_seed,
            scene_id,
            realization_key,
        )
        clean = _read_rgb(self.clean_paths[scene_id])
        if self.mode == "train":
            clean = _crop_and_augment(clean, self.crop_size, sample_seed)
        order = self._order(realization, order_slot)
        manifest = build_realization(clean, sample_seed)
        degraded = apply_order(clean, order, manifest)
        return {
            "lq": degraded,
            "gt": clean,
            "label": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            "scene_id": scene_id,
            "degradation_type": "low_haze",
            "formation_order": ">".join(order),
            "realization": realization,
            "generation_seed": sample_seed,
            "lq_path": f"synthetic://{scene_id}/{realization_key}/{'-'.join(order)}",
        }

    def summary(self) -> str:
        return (
            f"OrderControlDataset(mode={self.mode}, policy={self.order_policy}, "
            f"scenes={len(self.scene_ids)}, realizations={self.realizations}, "
            f"samples={len(self)}, crop={self.crop_size})"
        )
