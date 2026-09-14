"""Audit the exact Kaggle inputs before any CDD-11 training run."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Dict

import torch

from .datasets import CDD11Dataset, find_cdd11_root
from .model import build_model, count_parameters
from .project_config import (
    KAGGLE_CDD11_ROOT,
    KAGGLE_PRETRAINED_ROOT,
    PRETRAINED_FILENAMES,
)
from .train_kaggle import extract_state_dict, load_torch_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(KAGGLE_CDD11_ROOT))
    parser.add_argument("--pretrained-root", default=str(KAGGLE_PRETRAINED_ROOT))
    parser.add_argument(
        "--output",
        default="/kaggle/working/cot_nafnet_audit/audit.json",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sha256",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Hash every checkpoint; useful once, but slower than the default audit.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_report(path: Path, preset: str, with_hash: bool) -> Dict[str, object]:
    report: Dict[str, object] = {
        "path": str(path),
        "exists": path.is_file(),
    }
    if not path.is_file():
        return report
    report["bytes"] = path.stat().st_size
    if with_hash:
        report["sha256"] = file_sha256(path)

    state = extract_state_dict(load_torch_file(path))
    model = build_model("baseline", preset)
    target = model.state_dict()
    compatible = {
        key
        for key, value in state.items()
        if key in target and target[key].shape == value.shape
    }
    report.update(
        {
            "checkpoint_tensors": len(state),
            "model_tensors": len(target),
            "compatible_tensors": len(compatible),
            "missing_tensors": len(set(target) - compatible),
            "unexpected_tensors": sum(key not in target for key in state),
            "mismatched_tensors": sum(
                key in target and target[key].shape != value.shape
                for key, value in state.items()
            ),
            "parameters": count_parameters(model),
            "fully_compatible": compatible == set(target),
        }
    )
    del state, target, model
    gc.collect()
    return report


def main() -> None:
    args = parse_args()
    data_root = find_cdd11_root(args.data_root)
    train = CDD11Dataset(
        data_root,
        mode="train",
        crop_size=0,
        val_fraction=args.val_fraction,
        split_seed=args.seed,
        augment=False,
    )
    validation = CDD11Dataset(
        data_root,
        mode="val",
        crop_size=0,
        val_fraction=args.val_fraction,
        split_seed=args.seed,
        augment=False,
    )
    test = CDD11Dataset(data_root, mode="test", crop_size=0, augment=False)
    development_ids = set(train.scene_ids) | set(validation.scene_ids)
    dataset_report = {
        "root": str(data_root),
        "train_scenes": len(train.scene_ids),
        "validation_scenes": len(validation.scene_ids),
        "test_scenes": len(test.scene_ids),
        "train_samples": len(train),
        "validation_samples": len(validation),
        "test_samples": len(test),
        "train_validation_overlap": sorted(set(train.scene_ids) & set(validation.scene_ids)),
        "development_test_overlap": sorted(development_ids & set(test.scene_ids)),
        "train_scene_ids": list(train.scene_ids),
        "validation_scene_ids": list(validation.scene_ids),
        "test_scene_ids": list(test.scene_ids),
    }

    pretrained_root = Path(args.pretrained_root)
    checkpoint_reports: Dict[str, object] = {}
    for preset in ("gopro32", "gopro64", "sidd32", "sidd64"):
        filename = PRETRAINED_FILENAMES[preset]
        print(f"Auditing {preset}: {filename}", flush=True)
        checkpoint_reports[preset] = checkpoint_report(
            pretrained_root / filename, preset, args.sha256
        )

    payload = {
        "dataset": dataset_report,
        "pretrained_root": str(pretrained_root),
        "checkpoints": checkpoint_reports,
        "torch": torch.__version__,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Audit saved to {output}")
    if dataset_report["train_validation_overlap"] or dataset_report["development_test_overlap"]:
        raise RuntimeError("CDD-11 scene leakage detected; do not start training")
    if not all(item.get("fully_compatible") for item in checkpoint_reports.values()):
        raise RuntimeError("At least one pretrained checkpoint is missing or incompatible")


if __name__ == "__main__":
    main()
