"""Run selected CDD-11 ablations sequentially and evaluate each best checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from .project_config import KAGGLE_CDD11_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/calibration_a1_a2.json",
        help="JSON file containing shared settings and a runs list.",
    )
    parser.add_argument("--data-root", default=str(KAGGLE_CDD11_ROOT))
    parser.add_argument(
        "--experiments-root", default="/kaggle/working/experiments"
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=[],
        help="Optional run names; empty selects every run in the config.",
    )
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print validated commands without training or writing outputs.",
    )
    return parser.parse_args()


def read_config(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config.get("shared"), dict):
        raise ValueError("Ablation config requires a 'shared' object")
    runs = config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Ablation config requires a non-empty 'runs' list")
    names = [run.get("name") for run in runs if isinstance(run, dict)]
    if len(names) != len(runs) or any(not isinstance(name, str) for name in names):
        raise ValueError("Every ablation run requires a string 'name'")
    if len(set(names)) != len(names):
        raise ValueError("Ablation run names must be unique")
    return config


def read_json_object(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def select_runs(
    config: Dict[str, object], selected_names: Iterable[str]
) -> List[Dict[str, object]]:
    shared = dict(config["shared"])
    available = {run["name"]: run for run in config["runs"]}
    requested = list(selected_names) or list(available)
    unknown = sorted(set(requested).difference(available))
    if unknown:
        raise ValueError(
            f"Unknown runs {unknown}; available runs are {sorted(available)}"
        )
    return [{**shared, **available[name]} for name in requested]


def _flag(command: List[str], name: str, value: bool) -> None:
    command.append(f"--{'' if value else 'no-'}{name.replace('_', '-')}")


def training_command(
    run: Dict[str, object], data_root: Path, output_dir: Path, nproc: int
) -> List[str]:
    required = (
        "model",
        "preset",
        "pretrained",
        "adapter_hidden",
        "epochs",
        "max_minutes",
        "crop_size",
        "val_crop_size",
        "batch_size",
        "microbatch_size",
        "patches_per_image",
        "num_workers",
        "learning_rate",
        "backbone_lr_scale",
        "weight_decay",
        "fft_weight",
        "degradation_weight",
        "content_weight",
        "decorrelation_weight",
        "gate_weight",
        "freeze_backbone_epochs",
        "seed",
    )
    missing = [key for key in required if key not in run]
    if missing:
        raise ValueError(f"Run {run.get('name')!r} is missing settings: {missing}")
    if nproc < 1:
        raise ValueError("nproc-per-node must be positive")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc),
        "-m",
        "hybrid_cot_nafnet.train_kaggle",
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_dir),
    ]
    for key in required:
        command.extend((f"--{key.replace('_', '-')}", str(run[key])))
    _flag(command, "skip_gates", bool(run.get("skip_gates", True)))
    _flag(
        command,
        "multi_scale_degradation",
        bool(run.get("multi_scale_degradation", False)),
    )
    _flag(
        command,
        "balanced_degradation_loss",
        bool(run.get("balanced_degradation_loss", False)),
    )
    _flag(command, "multi_gpu", nproc > 1)
    _flag(command, "pin_memory", bool(run.get("pin_memory", False)))
    _flag(command, "save_optimizer", bool(run.get("save_optimizer", False)))
    _flag(command, "amp", bool(run.get("amp", True)))
    command.extend(("--save-every", "0"))
    return command


def evaluation_command(
    run: Dict[str, object],
    data_root: Path,
    output_dir: Path,
    checkpoint_name: str = "best.pt",
    evaluation_name: str = "evaluation",
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "hybrid_cot_nafnet.evaluate",
        "--checkpoint",
        str(output_dir / checkpoint_name),
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_dir / evaluation_name),
        "--split",
        "validation",
        "--tile",
        str(run.get("evaluation_tile", 0)),
        "--num-workers",
        str(run.get("evaluation_num_workers", 0)),
    ]
    _flag(
        command,
        "save_images",
        bool(run.get("save_evaluation_images", True)),
    )
    _flag(
        command,
        "save_comparisons",
        bool(run.get("save_comparisons", False)),
    )
    command.extend(
        ("--max-saved-per-type", str(run.get("max_saved_per_type", 1)))
    )
    _flag(command, "oracle_audit", bool(run.get("oracle_audit", False)))
    if run.get("oracle_block_sizes"):
        command.append("--oracle-block-sizes")
        command.extend(str(size) for size in run["oracle_block_sizes"])
    command.extend(
        ("--oracle-beta-step", str(run.get("oracle_beta_step", 0.01)))
    )
    _flag(command, "amp", bool(run.get("amp", True)))
    return command


def display_command(command: Iterable[str]) -> str:
    return " ".join(str(part) for part in command)


def run_checked(command: List[str], dry_run: bool) -> None:
    print("+", display_command(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_config(config_path)
    runs = select_runs(config, args.runs)
    data_root = Path(args.data_root)
    experiments_root = Path(args.experiments_root)
    print(f"Config: {config_path}")
    print(f"Selected runs: {[run['name'] for run in runs]}")
    print("Validation protocol: full_frame (val_crop_size=0, tile=0)")

    for run in runs:
        output_dir = experiments_root / str(run["name"])
        evaluation_summary = output_dir / "evaluation" / "summary.json"
        evaluate_reasoning = bool(run.get("evaluate_reasoning_checkpoint", False))
        reasoning_summary = output_dir / "reasoning_evaluation" / "summary.json"
        checkpoint = output_dir / "best.pt"
        training_summary = read_json_object(output_dir / "run_summary.json")
        if training_summary:
            status = training_summary.get("status")
            completed_epochs = int(training_summary.get("completed_epochs", 0))
            expected_epochs = int(run["epochs"])
            valid_completion = (
                status == "completed" and completed_epochs == expected_epochs
            )
            valid_time_limit = (
                status == "time_limit" and float(run.get("max_minutes", 0)) > 0
            )
            if not (valid_completion or valid_time_limit):
                raise RuntimeError(
                    f"Unusable training summary at {output_dir}: status={status!r}, "
                    f"completed_epochs={completed_epochs}, expected={expected_epochs}"
                )
        evaluations_complete = evaluation_summary.is_file() and (
            not evaluate_reasoning or reasoning_summary.is_file()
        )
        if evaluations_complete:
            if not training_summary:
                raise RuntimeError(
                    f"Evaluation exists without a training summary at {output_dir}"
                )
            print(f"Skipping completed run: {run['name']}")
            continue
        if not training_summary:
            if output_dir.exists() and any(output_dir.iterdir()) and not args.dry_run:
                raise RuntimeError(
                    f"Partial output already exists at {output_dir}. Preserve it, then "
                    "use a new experiments root or resume the run explicitly."
                )
            run_checked(
                training_command(run, data_root, output_dir, args.nproc_per_node),
                args.dry_run,
            )
        else:
            if not checkpoint.is_file():
                raise RuntimeError(
                    f"Training summary exists but best.pt is missing at {output_dir}"
                )
            print(f"Using existing checkpoint: {checkpoint}")
        if not evaluation_summary.is_file():
            run_checked(evaluation_command(run, data_root, output_dir), args.dry_run)
        if evaluate_reasoning and not reasoning_summary.is_file():
            reasoning_checkpoint = output_dir / "best_reasoning.pt"
            if not args.dry_run and not reasoning_checkpoint.is_file():
                raise RuntimeError(
                    "Reasoning evaluation requested but checkpoint is missing: "
                    f"{reasoning_checkpoint}"
                )
            run_checked(
                evaluation_command(
                    run,
                    data_root,
                    output_dir,
                    checkpoint_name="best_reasoning.pt",
                    evaluation_name="reasoning_evaluation",
                ),
                args.dry_run,
            )

    summary_command = [
        sys.executable,
        "-m",
        "hybrid_cot_nafnet.summarize_experiments",
        "--experiments-root",
        str(experiments_root),
        "--output",
        str(experiments_root / "ablation_summary.csv"),
    ]
    run_checked(summary_command, args.dry_run)


if __name__ == "__main__":
    main()
