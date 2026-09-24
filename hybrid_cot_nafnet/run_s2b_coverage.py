"""Run locked R0/R1 training, three paired probes, and the S2b decision summary."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .run_ablation import read_config, select_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s2b_coverage_null_cv5_v2.json")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_config(config_path)
    runs = select_runs(config, [])
    if len(runs) != 2 or {item["training_data"] for item in runs} != {"s2b_r0", "s2b_r1"}:
        raise ValueError("S2b orchestration requires exactly one R0 and one R1 run")
    if len({item["samples_per_epoch"] for item in runs}) != 1:
        raise ValueError("R0/R1 forward budgets differ")
    experiments_root = Path(args.experiments_root).resolve()
    experiments_root.mkdir(parents=True, exist_ok=True)
    evaluation_paths: dict[str, dict[str, list[Path]]] = {
        arm: {probe: [] for probe in ("original", "generator_b", "generator_a")}
        for arm in ("r0", "r1")
    }
    for fold in range(5):
        fold_root = experiments_root / f"fold_{fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        resolved_config = json.loads(json.dumps(config))
        resolved_config["shared"]["s2b_manifest"] = str(Path(args.manifest).resolve())
        resolved_config["shared"]["s2b_cache_root"] = str(Path(args.cache_root).resolve())
        resolved_config["shared"]["s2b_fold"] = fold
        resolved_config_path = fold_root / "resolved_s2b_config.json"
        if resolved_config_path.exists():
            existing = json.loads(resolved_config_path.read_text(encoding="utf-8"))
            if existing != resolved_config:
                raise RuntimeError(f"Refusing to replace a different fold-{fold} config")
        else:
            resolved_config_path.write_text(
                json.dumps(resolved_config, indent=2) + "\n", encoding="utf-8"
            )
        run(
            [
                sys.executable, "-m", "hybrid_cot_nafnet.run_ablation",
                "--config", str(resolved_config_path),
                "--data-root", str(Path(args.data_root).resolve()),
                "--experiments-root", str(fold_root),
                "--nproc-per-node", str(args.nproc_per_node),
                *(["--dry-run"] if args.dry_run else []),
            ],
            False,
        )
        for item in runs:
            arm = "r0" if item["training_data"] == "s2b_r0" else "r1"
            checkpoint = fold_root / item["name"] / "best.pt"
            for probe in ("original", "generator_b", "generator_a"):
                destination = fold_root / item["name"] / f"evaluation_{probe}"
                evaluation_paths[arm][probe].append(destination / "metrics.csv")
                if not (destination / "summary.json").is_file() or args.dry_run:
                    run(
                        [
                            sys.executable, "-m", "hybrid_cot_nafnet.evaluate_s2b_coverage",
                            "--checkpoint", str(checkpoint),
                            "--data-root", str(Path(args.data_root).resolve()),
                            "--manifest", str(Path(args.manifest).resolve()),
                            "--cache-root", str(Path(args.cache_root).resolve()),
                            "--fold", str(fold),
                            "--output-dir", str(destination),
                            "--probe", probe,
                            "--tile", str(args.tile),
                        ],
                        args.dry_run,
                    )
    summary_path = experiments_root / "s2b_coverage_summary.json"
    command = [
        sys.executable, "-m", "hybrid_cot_nafnet.summarize_s2b_coverage",
        "--output", str(summary_path),
    ]
    for arm in ("r0", "r1"):
        for probe in ("original", "generator_b", "generator_a"):
            for path in evaluation_paths[arm][probe]:
                command.extend((f"--{arm}-{probe.replace('_', '-')}", str(path)))
    run(command, args.dry_run)
    if not args.dry_run:
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"S2b decision: {result['decision']}")


if __name__ == "__main__":
    main()
