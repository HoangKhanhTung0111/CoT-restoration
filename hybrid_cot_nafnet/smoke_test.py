"""Create a tiny synthetic CDD-11 tree and exercise train/evaluate CLIs."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from .datasets.cdd11 import CDD11_TYPES
from .degradation_metrics import multilabel_degradation_metrics
from .evaluate import tile_blend_window, tile_starts, tiled_inference


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
        # Larger than the training crop so the smoke run exercises the
        # independent full-frame validation protocol.
        clear = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
        Image.fromarray(clear).save(split_dir / "clear" / f"{scene_id}.png")
        for degradation_type in CDD11_TYPES:
            Image.fromarray(degrade(clear, degradation_type)).save(
                split_dir / degradation_type / f"{scene_id}.png"
            )


def run(command, cwd: Path) -> None:
    print("+", " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def check_tiled_inference() -> None:
    """Check complete coverage, normalization, and complementary feathering."""
    assert tile_starts(720, 256, 32) == [0, 155, 309, 464]
    assert tile_starts(1080, 256, 32) == [0, 206, 412, 618, 824]

    image = torch.rand(1, 3, 48, 56)
    prediction, auxiliary = tiled_inference(
        nn.Identity(), image, torch.device("cpu"), 32, 8, False
    )
    assert not auxiliary
    assert torch.allclose(prediction, image, atol=1e-6)

    left = tile_blend_window(32, 32, 8, 0, 0, 32, 56)[0, 0, 16, -8:]
    right = tile_blend_window(32, 32, 8, 0, 24, 32, 56)[0, 0, 16, :8]
    assert torch.allclose(left + right, torch.ones(8), atol=1e-6)
    print("Feathered tile checks passed.")


def check_degradation_metrics() -> None:
    targets = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    probabilities = np.asarray(
        [[0.9, 0.1], [0.2, 0.8], [0.7, 0.6]], dtype=np.float32
    )
    metrics = multilabel_degradation_metrics(
        targets, probabilities, ("first", "second")
    )
    assert metrics["micro_f1"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["exact_match"] == 1.0
    assert metrics["per_label"]["first"]["auroc"] == 1.0
    assert metrics["per_label"]["second"]["average_precision"] == 1.0
    print("Degradation metric checks passed.")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    check_tiled_inference()
    check_degradation_metrics()
    # Keep a stable ignored workspace outside the source package: managed
    # Windows sandboxes may lock tempfile directories before nested files are
    # written.
    temporary_root = (
        project_root / "artifacts" / "local_smoke" / "smoke_workspace_metrics_v2"
    )
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
                "--no-pin-memory",
                "--no-save-optimizer",
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
                "--save-comparisons",
                "--max-saved-per-type",
                "1",
                "--no-amp",
                "--allow-cpu",
            ],
            project_root,
        )
        with (eval_output / "summary.json").open(encoding="utf-8") as handle:
            summary = json.load(handle)
        with (train_output / "dataset_manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        with (train_output / "run_summary.json").open(encoding="utf-8") as handle:
            training_summary = json.load(handle)
        assert manifest["train_crop_size"] == 16
        assert manifest["validation_crop_size"] == 0
        assert training_summary["validation_protocol"] == "full_frame"
        assert (train_output / "best_reasoning.pt").is_file()
        assert np.isfinite(training_summary["best_degradation_macro_f1"])
        assert summary["samples"] == len(CDD11_TYPES)
        assert summary["saved_comparisons"] == len(CDD11_TYPES)
        assert len(list((eval_output / "comparisons").glob("*.png"))) == len(
            CDD11_TYPES
        )
        assert np.isfinite(summary["macro_psnr"])
        assert np.isfinite(summary["macro_ssim"])
        assert np.isfinite(summary["degradation_macro_f1"])
        assert set(summary["degradation_per_label"]) == {
            "low", "haze", "rain", "snow"
        }
        print("End-to-end smoke test passed.")


if __name__ == "__main__":
    main()
