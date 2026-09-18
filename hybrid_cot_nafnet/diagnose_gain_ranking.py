"""Diagnose whether a frozen gain score ranks useful restoration blocks.

This command intentionally evaluates only the existing CDD-11 validation
split.  It does not train, alter either checkpoint, or expose the test split.
Blocks are ranked independently inside each image and compared with matched
random rejection at the same coverage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .datasets import CDD11Dataset, find_cdd11_root
from .evaluate import calculate_psnr, calculate_ssim, load_torch_file, oracle_block_output
from .project_config import KAGGLE_CDD11_ROOT
from .train_gain_predictor import (
    LocalGainPredictor,
    block_errors,
    blocks_from_pair,
    load_restorer,
    predict_scores,
    restore,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Frozen A3-M best.pt")
    parser.add_argument("--predictor", required=True, help="Frozen gain_predictor.pt")
    parser.add_argument("--calibration", required=True, help="calibration.json with fixed beta")
    parser.add_argument("--data-root", default=str(KAGGLE_CDD11_ROOT))
    parser.add_argument("--output-dir", default="/kaggle/working/gain_predictor/ranking_diagnostic")
    parser.add_argument(
        "--rejection-fractions", type=float, nargs="+",
        default=[0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30],
    )
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--random-ssim-batch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def load_predictor(path: str | Path, device: torch.device) -> Tuple[LocalGainPredictor, dict]:
    payload = load_torch_file(path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError("Expected a gain_predictor.pt payload containing 'model'")
    predictor = LocalGainPredictor(int(payload.get("width", 24)))
    predictor.load_state_dict(payload["model"], strict=True)
    return predictor.to(device).eval(), payload


def gaussian_window(channels: int, size: int, device: torch.device) -> Tensor:
    coordinates = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    kernel = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
    kernel /= kernel.sum()
    return torch.outer(kernel, kernel).expand(channels, 1, size, size).contiguous()


@torch.inference_mode()
def batched_ssim(prediction: Tensor, target: Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    """Return standard whole-image RGB SSIM for every prediction in a batch."""
    values: List[Tensor] = []
    channels = prediction.shape[1]
    smallest_side = min(prediction.shape[-2:])
    window_size = min(11, smallest_side if smallest_side % 2 else smallest_side - 1)
    window_size = max(1, window_size)
    window = gaussian_window(channels, window_size, device)
    for start in range(0, len(prediction), batch_size):
        pred = prediction[start : start + batch_size].to(device)
        truth = target.expand(len(pred), -1, -1, -1).to(device)
        mu_pred = F.conv2d(pred, window, groups=channels)
        mu_truth = F.conv2d(truth, window, groups=channels)
        mu_pred_sq, mu_truth_sq = mu_pred.square(), mu_truth.square()
        covariance = F.conv2d(pred * truth, window, groups=channels) - mu_pred * mu_truth
        variance_pred = F.conv2d(pred.square(), window, groups=channels) - mu_pred_sq
        variance_truth = F.conv2d(truth.square(), window, groups=channels) - mu_truth_sq
        numerator = (2 * mu_pred * mu_truth + 0.01**2) * (2 * covariance + 0.03**2)
        denominator = (mu_pred_sq + mu_truth_sq + 0.01**2) * (
            variance_pred + variance_truth + 0.03**2
        )
        values.append(
            (numerator / denominator.clamp_min(1e-12)).mean(dim=(1, 2, 3)).cpu()
        )
    return torch.cat(values).numpy()


def outputs_from_block_choices(
    degraded: Tensor, restored: Tensor, choices: Tensor, block_size: int
) -> Tensor:
    """Apply one or more [N,L] boolean restore choices to a single image."""
    if choices.ndim == 1:
        choices = choices.unsqueeze(0)
    height, width = degraded.shape[-2:]
    rows, columns = math.ceil(height / block_size), math.ceil(width / block_size)
    if choices.shape[1] != rows * columns:
        raise ValueError("Block choice count does not match image dimensions")
    masks = choices.reshape(-1, 1, rows, columns)
    masks = masks.repeat_interleave(block_size, 2).repeat_interleave(block_size, 3)
    masks = masks[..., :height, :width]
    return torch.where(masks, restored.expand(len(masks), -1, -1, -1), degraded.expand(len(masks), -1, -1, -1))


def rejection_choices(scores: Tensor, fraction: float) -> Tuple[Tensor, int]:
    block_count = len(scores)
    rejected_count = min(block_count, max(0, int(round(fraction * block_count))))
    choices = torch.ones(block_count, dtype=torch.bool)
    if rejected_count:
        choices[torch.argsort(scores)[:rejected_count]] = False
    return choices, rejected_count


def random_choices(
    block_count: int, rejected_count: int, repeats: int, generator: torch.Generator
) -> Tensor:
    choices = torch.ones(repeats, block_count, dtype=torch.bool)
    if rejected_count:
        for repeat in range(repeats):
            choices[repeat, torch.randperm(block_count, generator=generator)[:rejected_count]] = False
    return choices


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_diagnostic(
    rows: Sequence[dict], random_values: Dict[float, Dict[str, List[np.ndarray]]],
    fractions: Sequence[float], baseline: dict,
) -> dict:
    curves = []
    for fraction in fractions:
        items = [row for row in rows if row["rejection_fraction"] == fraction]
        random_psnr_matrix = np.stack(random_values[fraction]["psnr"])
        random_ssim_matrix = np.stack(random_values[fraction]["ssim"])
        random_harm_matrix = np.stack(random_values[fraction]["harm"])
        # Each column is one matched random trial across all validation images.
        random_macro_psnr = random_psnr_matrix.mean(axis=0)
        random_macro_ssim = random_ssim_matrix.mean(axis=0)
        random_harmed_images = random_harm_matrix.sum(axis=0)
        true_positive = sum(int(row["rejected_harmful_blocks"]) for row in items)
        rejected = sum(int(row["rejected_blocks"]) for row in items)
        harmful = sum(int(row["harmful_blocks"]) for row in items)
        curve = {
            "rejection_fraction": fraction,
            "mean_psnr": float(np.mean([row["policy_psnr"] for row in items])),
            "mean_ssim": float(np.mean([row["policy_ssim"] for row in items])),
            "images_worse_than_input": int(sum(row["policy_worse_than_input"] for row in items)),
            "random_mean_psnr": float(random_macro_psnr.mean()),
            "random_std_psnr": float(random_macro_psnr.std(ddof=1)) if len(random_macro_psnr) > 1 else 0.0,
            "random_p95_psnr": float(np.quantile(random_macro_psnr, 0.95)),
            "random_mean_ssim": float(random_macro_ssim.mean()),
            "random_std_ssim": float(random_macro_ssim.std(ddof=1)) if len(random_macro_ssim) > 1 else 0.0,
            "random_mean_harmed_images": float(random_harmed_images.mean()),
            "gain_over_random_db": float(np.mean([row["policy_psnr"] for row in items]) - random_macro_psnr.mean()),
            "gain_over_beta1_db": float(np.mean([row["policy_psnr"] for row in items]) - baseline["beta1"]["mean_psnr"]),
            "gain_over_fixed_beta_db": float(np.mean([row["policy_psnr"] for row in items]) - baseline["fixed_beta"]["mean_psnr"]),
            "harmful_block_rejection_precision": true_positive / max(1, rejected),
            "harmful_block_rejection_recall": true_positive / max(1, harmful),
        }
        curves.append(curve)

    nonzero = [item for item in curves if item["rejection_fraction"] > 0]
    best_ranking = max(nonzero, key=lambda item: item["gain_over_random_db"])
    best_policy = max(curves, key=lambda item: item["mean_psnr"])
    ranking_useful = bool(
        best_ranking["gain_over_random_db"] >= 0.01
        and best_ranking["mean_psnr"] > best_ranking["random_p95_psnr"]
    )
    quality_tolerance = 0.02
    fixed = baseline["fixed_beta"]
    useful_vs_fixed = bool(
        best_policy["mean_psnr"] > fixed["mean_psnr"]
        or (
            best_policy["mean_psnr"] >= fixed["mean_psnr"] - quality_tolerance
            and best_policy["images_worse_than_input"] < fixed["images_worse_than_input"]
        )
    )
    test_eligible = bool(
        ranking_useful and best_policy["rejection_fraction"] > 0 and useful_vs_fixed
    )
    if not ranking_useful:
        interpretation = "current_score_not_usefully_ranked"
    elif best_policy["mean_psnr"] <= baseline["beta1"]["mean_psnr"]:
        interpretation = "ranks_relative_benefit_but_rejection_not_useful"
    elif not useful_vs_fixed:
        interpretation = "local_rejection_helps_but_not_beyond_fixed_beta"
    else:
        interpretation = "validation_policy_is_eligible_for_one_locked_test_run"
    return {
        "split": "validation_only",
        "curves": curves,
        "baselines": baseline,
        "best_ranking_fraction": best_ranking["rejection_fraction"],
        "selected_policy_fraction": best_policy["rejection_fraction"],
        "decision": {
            "ranking_useful_vs_matched_random": ranking_useful,
            "useful_vs_fixed_beta": useful_vs_fixed,
            "eligible_for_locked_test": test_eligible,
            "interpretation": interpretation,
            "ranking_rule": "gain over random >= 0.01 dB and above random 95th percentile",
            "fixed_beta_rule": "higher PSNR, or within 0.02 dB with fewer harmed images",
            "test_rule": "ranking useful, nonzero selected rejection, and useful vs fixed beta",
        },
    }


@torch.inference_mode()
def run_diagnostic(
    dataset: CDD11Dataset,
    restorer,
    predictor,
    fixed_beta: float,
    block_size: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    random_values: Dict[float, Dict[str, List[np.ndarray]]] = {
        fraction: defaultdict(list) for fraction in args.rejection_fractions
    }
    baseline_rows: List[dict] = []
    generator = torch.Generator().manual_seed(args.seed + 1009)

    for index in range(len(dataset)):
        sample = dataset[index]
        degraded, target = sample["lq"].unsqueeze(0), sample["gt"].unsqueeze(0)
        restored = restore(restorer, degraded, device, args.tile, args.overlap, args.amp)
        blocks = blocks_from_pair(degraded, restored, block_size)
        scores = predict_scores(predictor, blocks, device, args.batch_size)
        input_sse, output_sse = block_errors(degraded, restored, target, block_size)
        harmful = output_sse > input_sse
        input_psnr = calculate_psnr(degraded, target)
        restored_psnr = calculate_psnr(restored, target)
        restored_ssim = calculate_ssim(restored, target)
        blended = (degraded + fixed_beta * (restored - degraded)).clamp(0, 1)
        block_oracle, _ = oracle_block_output(degraded, restored, target, block_size)
        image_oracle = restored if restored_psnr >= input_psnr else degraded
        baseline_rows.append({
            "input_psnr": input_psnr,
            "input_ssim": calculate_ssim(degraded, target),
            "beta1_psnr": restored_psnr,
            "beta1_ssim": restored_ssim,
            "fixed_beta_psnr": calculate_psnr(blended, target),
            "fixed_beta_ssim": calculate_ssim(blended, target),
            "image_oracle_psnr": calculate_psnr(image_oracle, target),
            "image_oracle_ssim": calculate_ssim(image_oracle, target),
            "block_oracle_psnr": calculate_psnr(block_oracle, target),
            "block_oracle_ssim": calculate_ssim(block_oracle, target),
        })

        for fraction in args.rejection_fractions:
            choices, rejected_count = rejection_choices(scores, fraction)
            policy = outputs_from_block_choices(degraded, restored, choices, block_size)
            policy_psnr = calculate_psnr(policy, target)
            rejected = ~choices
            rows.append({
                "scene_id": sample["scene_id"],
                "type": sample["degradation_type"],
                "rejection_fraction": fraction,
                "rejected_blocks": rejected_count,
                "harmful_blocks": int(harmful.sum()),
                "rejected_harmful_blocks": int((rejected & harmful).sum()),
                "policy_psnr": policy_psnr,
                "policy_ssim": calculate_ssim(policy, target),
                "policy_worse_than_input": int(policy_psnr < input_psnr - 1e-12),
            })

            if rejected_count == 0:
                random_psnr = np.full(args.random_repeats, restored_psnr)
                random_ssim = np.full(args.random_repeats, restored_ssim)
            else:
                choices_random = random_choices(
                    len(scores), rejected_count, args.random_repeats, generator
                )
                random_outputs = outputs_from_block_choices(
                    degraded, restored, choices_random, block_size
                )
                mse = (random_outputs - target).square().mean(dim=(1, 2, 3)).numpy()
                random_psnr = -10 * np.log10(np.maximum(mse, 1e-12))
                random_ssim = batched_ssim(
                    random_outputs, target, device, args.random_ssim_batch
                )
            random_values[fraction]["psnr"].append(random_psnr)
            random_values[fraction]["ssim"].append(random_ssim)
            random_values[fraction]["harm"].append(random_psnr < input_psnr - 1e-12)
        print(f"diagnostic [{index + 1:03d}/{len(dataset)}] {sample['degradation_type']}/{sample['scene_id']}")

    def method_summary(prefix: str) -> dict:
        psnr = np.array([row[f"{prefix}_psnr"] for row in baseline_rows])
        return {
            "mean_psnr": float(psnr.mean()),
            "mean_ssim": float(np.mean([row[f"{prefix}_ssim"] for row in baseline_rows])),
            "images_worse_than_input": int(sum(
                value < row["input_psnr"] - 1e-12
                for value, row in zip(psnr, baseline_rows)
            )),
        }

    baseline = {
        "input": method_summary("input"),
        "beta1": method_summary("beta1"),
        "fixed_beta": {**method_summary("fixed_beta"), "beta": fixed_beta},
        "image_oracle": method_summary("image_oracle"),
        "block_oracle": method_summary("block_oracle"),
    }
    summary = aggregate_diagnostic(rows, random_values, args.rejection_fractions, baseline)
    summary.update({
        "count": len(dataset),
        "block_size": block_size,
        "random_repeats": args.random_repeats,
        "local_oracle_value_over_image_oracle_db": (
            baseline["block_oracle"]["mean_psnr"] - baseline["image_oracle"]["mean_psnr"]
        ),
        "local_oracle_value_over_fixed_beta_db": (
            baseline["block_oracle"]["mean_psnr"] - baseline["fixed_beta"]["mean_psnr"]
        ),
    })
    return rows, summary


def main() -> None:
    args = parse_args()
    if args.random_repeats < 2 or args.random_ssim_batch <= 0:
        raise ValueError("random-repeats must be >=2 and random-ssim-batch must be positive")
    fractions = sorted(set(float(value) for value in args.rejection_fractions))
    if not fractions or fractions[0] < 0 or fractions[-1] >= 1:
        raise ValueError("rejection fractions must be in [0, 1)")
    args.rejection_fractions = fractions
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable; enable a Kaggle GPU or pass --allow-cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = bool(args.amp and device.type == "cuda")
    seed_everything(args.seed)

    restorer, _ = load_restorer(args.checkpoint, device)
    predictor, predictor_payload = load_predictor(args.predictor, device)
    block_size = int(predictor_payload.get("block_size", 32))
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    fixed_beta = float(calibration["selected_fixed_beta"])
    dataset = CDD11Dataset(
        find_cdd11_root(args.data_root), mode="val", crop_size=0,
        val_fraction=args.val_fraction, split_seed=args.seed, augment=False,
    )
    rows, summary = run_diagnostic(
        dataset, restorer, predictor, fixed_beta, block_size, device, args
    )
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "per_image_curve.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    protocol = {
        "purpose": "validation-only within-image ranking diagnostic",
        "test_split_touched": False,
        "restorer_checkpoint": str(Path(args.checkpoint).resolve()),
        "predictor_checkpoint": str(Path(args.predictor).resolve()),
        "validation_scene_ids": list(dataset.scene_ids),
        "args": vars(args),
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
