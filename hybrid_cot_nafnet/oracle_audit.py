"""Audit the oracle headroom available in saved comparison contact sheets.

The evaluator writes lossless PNG sheets with a 44-pixel header and three
equal-width panels: degraded input, restored output, and ground truth.  This
script recovers those panels and measures the best result attainable when an
oracle may select either the input or the restored output at image or block
level.  Oracle results are diagnostic upper bounds, not deployable results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


HEADER_HEIGHT = 44


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("NAME", "COMPARISON_DIR"),
        required=True,
        help="Model name and directory containing comparison PNGs; repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocks", type=int, nargs="+", default=[32, 8])
    parser.add_argument("--beta-step", type=float, default=0.01)
    return parser.parse_args()


def load_sheet(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if array.shape[0] <= HEADER_HEIGHT or array.shape[1] % 3:
        raise ValueError(f"Unexpected comparison layout: {path} {array.shape}")
    body = array[HEADER_HEIGHT:]
    width = body.shape[1] // 3
    return tuple(body[:, i * width : (i + 1) * width] for i in range(3))  # type: ignore[return-value]


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(prediction - target), dtype=np.float64))


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    error = mse(prediction, target)
    return float("inf") if error == 0 else -10.0 * math.log10(error)


def block_oracle_metrics(
    degraded: np.ndarray,
    restored: np.ndarray,
    target: np.ndarray,
    block_size: int,
) -> tuple[float, float]:
    height, width = target.shape[:2]
    padded_height = math.ceil(height / block_size) * block_size
    padded_width = math.ceil(width / block_size) * block_size

    def block_sum(values: np.ndarray) -> np.ndarray:
        padded = np.zeros((padded_height, padded_width), dtype=np.float64)
        padded[:height, :width] = values
        return padded.reshape(
            padded_height // block_size,
            block_size,
            padded_width // block_size,
            block_size,
        ).sum(axis=(1, 3))

    input_sse = block_sum(np.square(degraded - target).sum(axis=2))
    restored_sse = block_sum(np.square(restored - target).sum(axis=2))
    valid_pixels = block_sum(np.ones((height, width), dtype=np.float64))
    use_restored = restored_sse < input_sse
    oracle_mse = float(
        np.minimum(input_sse, restored_sse).sum() / (height * width * 3)
    )
    restored_fraction = float(valid_pixels[use_restored].sum() / (height * width))
    return oracle_mse, restored_fraction


def image_oracle(
    degraded: np.ndarray, restored: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, float]:
    use_restored = mse(restored, target) < mse(degraded, target)
    return (restored if use_restored else degraded).copy(), float(use_restored)


def beta_values(step: float) -> Iterable[float]:
    if not 0 < step <= 1:
        raise ValueError("--beta-step must be in (0, 1]")
    count = int(round(1.0 / step))
    return np.linspace(0.0, 1.0, count + 1).tolist()


def psnr_from_mse(error: float) -> float:
    return float("inf") if error == 0 else -10.0 * math.log10(error)


def summarize(rows: list[dict[str, object]], model: str) -> dict[str, object]:
    selected = [row for row in rows if row["model"] == model]
    methods = sorted({str(row["method"]) for row in selected})
    result: dict[str, object] = {"count": len(selected) // len(methods)}
    restored_mean = float(
        np.mean([float(r["psnr"]) for r in selected if r["method"] == "restored"])
    )
    for method in methods:
        subset = [row for row in selected if row["method"] == method]
        mean_psnr = float(np.mean([float(row["psnr"]) for row in subset]))
        result[method] = {
            "mean_psnr": mean_psnr,
            "headroom_over_restored_db": mean_psnr - restored_mean,
            "mean_restored_fraction": float(
                np.mean([float(row["restored_fraction"]) for row in subset])
            ),
            "images_worse_than_input": int(
                sum(float(row["delta_vs_input_db"]) < -1e-12 for row in subset)
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    model_dirs = {name: Path(directory) for name, directory in args.model}
    if len(model_dirs) != len(args.model):
        raise ValueError("Model names must be unique")
    for name, directory in model_dirs.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing comparison directory for {name}: {directory}")

    common_names: set[str] | None = None
    paths_by_model: dict[str, dict[str, Path]] = {}
    for model, directory in model_dirs.items():
        paths = {path.name: path for path in directory.glob("*.png")}
        if not paths:
            raise FileNotFoundError(f"No comparison PNGs found in {directory}")
        paths_by_model[model] = paths
        common_names = set(paths) if common_names is None else common_names & set(paths)
    assert common_names

    panels: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for model, paths in paths_by_model.items():
        panels[model] = {name: load_sheet(paths[name]) for name in sorted(common_names)}

    reference_model = next(iter(model_dirs))
    for model in model_dirs:
        for name in common_names:
            ref_lq, _, ref_gt = panels[reference_model][name]
            lq, _, gt = panels[model][name]
            if not np.array_equal(ref_lq, lq) or not np.array_equal(ref_gt, gt):
                raise ValueError(f"Input/GT mismatch for {name}: {reference_model} vs {model}")

    betas = list(beta_values(args.beta_step))
    best_beta_by_model: dict[str, float] = {}
    for model in model_dirs:
        scores = []
        for beta in betas:
            values = []
            for degraded, restored, target in panels[model].values():
                base_error = degraded - target
                residual = restored - degraded
                a = float(np.mean(np.square(residual), dtype=np.float64))
                b = float(2.0 * np.mean(base_error * residual, dtype=np.float64))
                c = float(np.mean(np.square(base_error), dtype=np.float64))
                values.append(psnr_from_mse(a * beta * beta + b * beta + c))
            scores.append((float(np.mean(values)), beta))
        best_beta_by_model[model] = max(scores)[1]

    rows: list[dict[str, object]] = []
    for model in model_dirs:
        beta = best_beta_by_model[model]
        for filename in sorted(common_names):
            degraded, restored, target = panels[model][filename]
            input_psnr = psnr(degraded, target)
            candidates: list[tuple[str, np.ndarray, float]] = [
                ("input", degraded, 0.0),
                ("restored", restored, 1.0),
                (
                    f"fixed_beta_{beta:.2f}",
                    degraded + beta * (restored - degraded),
                    beta,
                ),
            ]
            oracle_image, fraction = image_oracle(degraded, restored, target)
            candidates.append(("oracle_image", oracle_image, fraction))
            for method, candidate, fraction in candidates:
                candidate_psnr = psnr(candidate, target)
                rows.append(
                    {
                        "model": model,
                        "sample": filename.removesuffix(".png"),
                        "method": method,
                        "psnr": candidate_psnr,
                        "input_psnr": input_psnr,
                        "delta_vs_input_db": candidate_psnr - input_psnr,
                        "restored_fraction": fraction,
                    }
                )
            for block_size in args.blocks:
                oracle_mse, fraction = block_oracle_metrics(
                    degraded, restored, target, block_size
                )
                candidate_psnr = psnr_from_mse(oracle_mse)
                rows.append(
                    {
                        "model": model,
                        "sample": filename.removesuffix(".png"),
                        "method": f"oracle_block_{block_size}",
                        "psnr": candidate_psnr,
                        "input_psnr": input_psnr,
                        "delta_vs_input_db": candidate_psnr - input_psnr,
                        "restored_fraction": fraction,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "scope": {
            "samples": sorted(common_names),
            "sample_count": len(common_names),
            "unique_source_scenes": sorted(
                {name.rsplit("_", 1)[-1].removesuffix(".png") for name in common_names}
            ),
            "warning": (
                "Exploratory only: the saved 20-epoch contact sheets contain one "
                "source scene per degradation type, not the full validation split."
            ),
        },
        "best_fixed_beta_selected_on_same_sample": best_beta_by_model,
        "models": {model: summarize(rows, model) for model in model_dirs},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
