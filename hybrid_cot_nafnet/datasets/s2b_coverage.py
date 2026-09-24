"""Locked datasets for the S2b R0/R1 coverage-only null pilot."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .cdd11 import CDD11_TYPES, IMAGE_EXTENSIONS, _label_for_type, _read_rgb


PROTOCOL_VERSION = "s2b-coverage-null-cv5-v2"
GENERATED_TYPES: Tuple[str, ...] = (
    "b_low_weak",
    "b_low_strong",
    "b_rain_haze",
    "b_snow_haze",
    "b_low_weak_rain_haze",
    "b_low_strong_rain_haze",
    "b_low_weak_snow_haze",
    "b_low_strong_snow_haze",
)
A_PROBE_TYPES: Tuple[str, ...] = tuple(name.replace("b_", "a_", 1) for name in GENERATED_TYPES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(path: Path) -> str:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def find_cdd11_train_dir(path: str | Path) -> Path:
    """Resolve an attached ``CDD-11_train`` tree without touching its test split."""
    root = Path(path).expanduser()
    direct = [root / "train", root / "CDD-11_train", root]
    matches = []
    for candidate in direct:
        if (candidate / "clear").is_dir() and all(
            (candidate / name).is_dir() for name in CDD11_TYPES
        ):
            matches.append(candidate.resolve())
    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise FileNotFoundError(
            f"Could not find a CDD-11 training tree below attached input {root}"
        )
    raise RuntimeError("Multiple CDD-11 training trees found: " + ", ".join(map(str, unique)))


def _image_map(directory: Path) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in result:
                raise RuntimeError(f"Duplicate image stem {path.stem!r} in {directory}")
            result[path.stem] = path
    return result


def build_cv_manifest(
    data_root: str | Path,
    *,
    development_count: int = 25,
    fold_count: int = 5,
    seed: int = 42,
) -> dict:
    """Lock five out-of-fold splits over the attached CDD-11-30 development pool."""
    if development_count < 2 or fold_count < 2:
        raise ValueError("development_count and fold_count must both be at least two")
    if development_count % fold_count:
        raise ValueError("development_count must be divisible by fold_count")
    train_dir = find_cdd11_train_dir(data_root)
    clear = _image_map(train_dir / "clear")
    if len(clear) != development_count:
        raise RuntimeError(
            f"CDD-11-30 development pool must contain exactly {development_count} scenes; "
            f"found {len(clear)} in {train_dir}"
        )
    shuffled = sorted(clear)
    random.Random(seed).shuffle(shuffled)

    records = []
    for scene in sorted(clear):
        target = clear[scene]
        with Image.open(target) as image:
            size = [int(image.width), int(image.height)]
        for degradation_type in CDD11_TYPES:
            candidate = train_dir / degradation_type / target.name
            if not candidate.is_file():
                raise FileNotFoundError(f"Missing CDD-11 pair: {candidate}")
            with Image.open(candidate) as image:
                if [image.width, image.height] != size:
                    raise RuntimeError(f"LQ/GT size mismatch: {candidate}")
        records.append(
            {
                "scene": scene,
                "file": target.name,
                "pixel_sha256": pixel_sha256(target),
                "size_wh": size,
            }
        )
    validation_count = development_count // fold_count
    folds = []
    for fold in range(fold_count):
        start = fold * validation_count
        validation = shuffled[start:start + validation_count]
        validation_set = set(validation)
        train = [scene for scene in shuffled if scene not in validation_set]
        folds.append({"fold": fold, "train": train, "validation": validation})

    return {
        "protocol_version": PROTOCOL_VERSION,
        "source": {
            "kaggle_dataset": "mintesnotfikir/cdd-11-30",
            "tree": "CDD-11_train",
            "attached_input_required": True,
            "internet_download_allowed": False,
            "test_partition": "CDD-11_test",
            "test_partition_status": "sealed_not_enumerated",
        },
        "selection": {
            "seed": seed,
            "development_count": development_count,
            "fold_count": fold_count,
            "train_count_per_fold": development_count - validation_count,
            "validation_count_per_fold": validation_count,
        },
        "scenes": records,
        "folds": folds,
    }


def write_new_manifest(path: str | Path, manifest: dict) -> None:
    """Write once; an existing manifest must be byte-for-byte equivalent JSON."""
    destination = Path(path)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(f"Refusing to overwrite a different manifest: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_cv_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unexpected S2b protocol version")
    selection = manifest.get("selection", {})
    development_count = int(selection.get("development_count", -1))
    fold_count = int(selection.get("fold_count", -1))
    train_count = int(selection.get("train_count_per_fold", -1))
    validation_count = int(selection.get("validation_count_per_fold", -1))
    records = manifest.get("scenes")
    folds = manifest.get("folds")
    if not isinstance(records, list) or len(records) != development_count:
        raise ValueError("Malformed S2b development scene records")
    scene_ids = [item.get("scene") for item in records]
    if len(set(scene_ids)) != development_count:
        raise ValueError("Duplicate S2b development scene IDs")
    universe = set(scene_ids)
    if not isinstance(folds, list) or len(folds) != fold_count:
        raise ValueError("Malformed S2b folds")
    validation_occurrences = {scene: 0 for scene in scene_ids}
    for index, fold in enumerate(folds):
        if fold.get("fold") != index:
            raise ValueError("S2b fold indices must be consecutive")
        train_ids = fold.get("train")
        validation_ids = fold.get("validation")
        if not isinstance(train_ids, list) or len(train_ids) != train_count:
            raise ValueError(f"Malformed S2b train split for fold {index}")
        if not isinstance(validation_ids, list) or len(validation_ids) != validation_count:
            raise ValueError(f"Malformed S2b validation split for fold {index}")
        if set(train_ids) & set(validation_ids) or set(train_ids) | set(validation_ids) != universe:
            raise ValueError(f"Leakage or missing scenes in S2b fold {index}")
        for scene in validation_ids:
            validation_occurrences[scene] += 1
    if set(validation_occurrences.values()) != {1}:
        raise ValueError("Every S2b development scene must be validation exactly once")
    source = manifest.get("source", {})
    if source.get("internet_download_allowed") is not False:
        raise ValueError("S2b attached-input protocol must forbid Internet data download")
    return manifest


def fold_records(manifest: dict, fold: int, split: str) -> List[dict]:
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    folds = manifest["folds"]
    if not 0 <= fold < len(folds):
        raise ValueError(f"fold must be in [0, {len(folds) - 1}]")
    by_scene = {record["scene"]: record for record in manifest["scenes"]}
    return [by_scene[scene] for scene in folds[fold][split]]


def _generated_label(degradation_type: str) -> Tensor:
    tokens = set(degradation_type.removeprefix("b_").split("_"))
    return torch.tensor(
        [float(name in tokens) for name in ("low", "haze", "rain", "snow")],
        dtype=torch.float32,
    )


class S2BCoverageDataset(Dataset):
    """R0 uses CDD-11; R1 adds cached Generator-B factorial views."""

    def __init__(
        self,
        root: str | Path,
        manifest_path: str | Path,
        cache_root: str | Path,
        *,
        fold: int,
        split: str,
        arm: str,
        crop_size: int = 256,
        augment: bool = True,
        degradation_types: Sequence[str] | None = None,
        verify_targets: bool = True,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if arm not in {"r0", "r1"}:
            raise ValueError("arm must be r0 or r1")
        self.train_dir = find_cdd11_train_dir(root)
        self.manifest_path = Path(manifest_path)
        self.cache_root = Path(cache_root)
        self.split = split
        self.fold = int(fold)
        self.arm = arm
        self.crop_size = int(crop_size)
        self.augment = bool(augment and split == "train")
        manifest = load_cv_manifest(self.manifest_path)
        records = fold_records(manifest, self.fold, split)
        probe_types = A_PROBE_TYPES if split == "validation" else ()
        available_types = CDD11_TYPES + (GENERATED_TYPES + probe_types if arm == "r1" else ())
        types = tuple(degradation_types or available_types)
        invalid = sorted(set(types).difference(available_types))
        if invalid:
            raise ValueError(f"Types not available in {arm}: {invalid}")

        samples = []
        for record in records:
            scene, filename = record["scene"], record["file"]
            target = self.train_dir / "clear" / filename
            if not target.is_file():
                raise FileNotFoundError(f"Missing locked clean target: {target}")
            if verify_targets and pixel_sha256(target) != record["pixel_sha256"]:
                raise RuntimeError(f"Clean target hash changed for scene {scene}")
            for degradation_type in types:
                if degradation_type in CDD11_TYPES:
                    lq = self.train_dir / degradation_type / filename
                    label = _label_for_type(degradation_type)
                else:
                    lq = self.cache_root / "generated" / degradation_type / filename
                    label = _generated_label(degradation_type)
                if not lq.is_file():
                    raise FileNotFoundError(f"Missing S2b input: {lq}")
                samples.append((lq, target, label, scene, degradation_type))
        if not samples:
            raise RuntimeError("No S2b samples selected")
        self.scene_ids = tuple(record["scene"] for record in records)
        self.samples = samples
        self.degradation_types = types

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        lq_path, gt_path, label, scene, degradation_type = self.samples[index]
        lq, gt = _read_rgb(lq_path), _read_rgb(gt_path)
        if lq.shape != gt.shape:
            raise RuntimeError(f"LQ/GT shape mismatch: {lq_path}")
        height, width = lq.shape[-2:]
        pad_h, pad_w = max(0, self.crop_size - height), max(0, self.crop_size - width)
        if pad_h or pad_w:
            padding = (0, pad_w, 0, pad_h)
            lq = F.pad(lq, padding, mode="replicate")
            gt = F.pad(gt, padding, mode="replicate")
        if self.crop_size > 0:
            height, width = lq.shape[-2:]
            if self.split == "train":
                top = random.randint(0, height - self.crop_size)
                left = random.randint(0, width - self.crop_size)
            else:
                top = (height - self.crop_size) // 2
                left = (width - self.crop_size) // 2
            region = (..., slice(top, top + self.crop_size), slice(left, left + self.crop_size))
            lq, gt = lq[region], gt[region]
        if self.augment:
            if random.random() < 0.5:
                lq, gt = lq.flip(-1), gt.flip(-1)
            if random.random() < 0.5:
                lq, gt = lq.flip(-2), gt.flip(-2)
            if random.random() < 0.5:
                lq, gt = lq.transpose(-2, -1), gt.transpose(-2, -1)
        return {
            "lq": lq.contiguous(),
            "gt": gt.contiguous(),
            "label": label.clone(),
            "scene_id": scene,
            "degradation_type": degradation_type,
            "lq_path": str(lq_path),
        }

    def summary(self) -> str:
        return (
            f"S2BCoverageDataset(fold={self.fold}, split={self.split}, arm={self.arm}, "
            f"scenes={len(self.scene_ids)}, types={len(self.degradation_types)}, "
            f"samples={len(self)}, crop={self.crop_size})"
        )
