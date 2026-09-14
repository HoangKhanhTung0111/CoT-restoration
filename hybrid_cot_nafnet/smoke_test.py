"""Create a tiny synthetic CDD-11 tree and exercise train/evaluate CLIs."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

from .datasets.cdd11 import CDD11_TYPES


def degrade(image: np.ndarray, degradation_type: str) -> np.ndarray:
    output = image.astype(np.float32)
    tokens = set(degradation_type.split("_"))
    if "low" in tokens:
        output *= 0.55
    if "haze" in tokens:
        output = output * 0.65 + 255.0 * 0.35
    if "rain" in tokens:
        output[:, ::5, :] = np.minimum(255.0, output[:, ::5, :] + 60.0)
    if "snow" in tokens:
        output[::6, ::6, :] = 255.0
    return np.clip(output, 0, 255).astype(np.uint8)


def create_split(root: Path, split: str, count: int, seed: int, offset: int = 0) -> None:
    split_dir = root / split
    for name in ("clear",) + CDD11_TYPES:
        (split_dir / name).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(count):
        scene_id = f"{index + offset:05d}"
        clear = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        Image.fromarray(clear).save(split_dir / "clear" / f"{scene_id}.png")
        for degradation_type in CDD11_TYPES:
            Image.fromarray(degrade(clear, degradation_type)).save(
                split_dir / degradation_type / f"{scene_id}.png"
            )


def run(command, cwd: Path) -> None:
    print("+", " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    # Keep a stable ignored workspace: managed Windows sandboxes may lock
    # directories created through tempfile before nested files are written.
    temporary_root = Path(__file__).resolve().parent / "_smoke_workspace"
    temporary_root.mkdir(exist_ok=True)
    with nullcontext(temporary_root):
        data_root = temporary_root / "cdd"
        create_split(data_root, "CDD-11_train", count=3, seed=1)
        create_split(data_root, "CDD-11_test", count=1, seed=2, offset=100)
        train_output = temporary_root / "train"
        eval_output = temporary_root / "eval"
        run(
            [
                sys.executable,
                "-m",
                "hybrid_cot_nafnet.train_kaggle",
                "--data-root",
                str(data_root),
                "--output-dir",
                str(train_output),
                "--preset",
                "compact",
                "--epochs",
                "1",
                "--max-minutes",
                "0",
                "--crop-size",
                "16",
                "--batch-size",
                "11",
                "--microbatch-size",
                "11",
                "--patches-per-image",
                "1",
                "--num-workers",
                "0",
                "--fft-weight",
                "0",
                "--no-amp",
                "--allow-cpu",
            ],
            project_root,
        )
        run(
            [
                sys.executable,
                "-m",
                "hybrid_cot_nafnet.evaluate",
                "--checkpoint",
                str(train_output / "best.pt"),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(eval_output),
                "--tile",
                "16",
                "--overlap",
                "0",
                "--num-workers",
                "0",
                "--no-save-images",
                "--no-amp",
                "--allow-cpu",
            ],
            project_root,
        )
        with (eval_output / "summary.json").open(encoding="utf-8") as handle:
            summary = json.load(handle)
        assert summary["samples"] == len(CDD11_TYPES)
        assert np.isfinite(summary["macro_psnr"])
        assert np.isfinite(summary["macro_ssim"])
        print("End-to-end smoke test passed.")


if __name__ == "__main__":
    main()
