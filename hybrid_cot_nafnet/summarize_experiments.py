"""Collect completed experiment folders into one ablation CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-root", default="/kaggle/working/experiments"
    )
    parser.add_argument(
        "--output", default="/kaggle/working/experiments/ablation_summary.csv"
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def flatten(prefix: str, value: object) -> Dict[str, object]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: Dict[str, object] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            result.update(flatten(name, item))
        elif not isinstance(item, (list, tuple)):
            result[name] = item
    return result


def experiment_rows(root: Path) -> Iterable[Dict[str, object]]:
    if not root.is_dir():
        return
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        config = read_json(directory / "run_config.json")
        training = read_json(directory / "run_summary.json")
        evaluation = read_json(directory / "evaluation" / "summary.json")
        if not config and not training and not evaluation:
            continue
        row: Dict[str, object] = {"run": directory.name}
        row.update(flatten("config", config))
        row.update(flatten("train", training))
        # The evaluator can target validation or test. A neutral prefix avoids
        # mislabelling development metrics as held-out test results.
        row.update(flatten("evaluation", evaluation))
        yield row


def main() -> None:
    args = parse_args()
    rows = list(experiment_rows(Path(args.experiments_root)))
    if not rows:
        raise RuntimeError(f"No experiment records found below {args.experiments_root}")
    columns = ["run"] + sorted({key for row in rows for key in row if key != "run"})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} runs to {output}")


if __name__ == "__main__":
    main()
