"""Validate and summarize an S1 replication archive using NumPy only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .audit_low_weather_replication import (
    GENERATORS, MODELS, REGIONS, VIEWS, ProtocolError, _fixture_records,
    generator_b, manifest_from_fixture, validate_manifest,
)
from .common_failure_audit import save_json, sha256


WEATHERS = ("rain_haze", "snow_haze")
STRENGTHS = ("weak", "strong")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260923
RUN_REQUIRED = ("run.json", "protocol.json", "manifest.json", "sources.json")


class SummaryError(ValueError):
    """The result package does not match the frozen S1 protocol."""


def _read_json_zip(archive: zipfile.ZipFile, name: str):
    try:
        return json.loads(archive.read(name))
    except KeyError as exc:
        raise SummaryError(f"Missing archive entry {name}") from exc


def _validate_metrics(manifest: dict, fixture: dict, run: dict,
                      protocol: dict, sources: dict,
                      source_fixture: dict, statuses: dict[str, dict],
                      metrics: dict[str, list[dict]]) -> dict:
    if run.get("errors") != []:
        raise SummaryError(f"Run recorded errors: {run.get('errors')}")
    commit = run.get("project_code_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise SummaryError("Missing pinned project code commit")
    if protocol.get("protocol_version") != "low-weather-replication-v1.1":
        raise SummaryError("Unexpected protocol version")
    if protocol.get("manifest_fixture_sha256") != fixture.get("source_manifest_sha256"):
        raise SummaryError("Protocol manifest fixture hash mismatch")
    expected_repos = source_fixture.get("repos", {})
    if {key: value.get("revision") for key, value in sources.get("repos", {}).items()} != {
            key: value.get("revision") for key, value in expected_repos.items()}:
        raise SummaryError("Pinned external source revisions differ from source fixture")
    for filename, expected in source_fixture.get("weights", {}).items():
        record = sources.get("weights", {}).get(filename)
        if not record or record.get("sha256") != expected.get("sha256"):
            raise SummaryError(f"Checkpoint hash differs from source fixture: {filename}")
    expected_scenes = {row["scene"] for row in _fixture_records(fixture)
                       if row["partition"] == "confirmation"}
    manifest_rows = validate_manifest(manifest, fixture, "confirmation", 0)
    if {row["scene"] for row in manifest_rows} != expected_scenes:
        raise SummaryError("Confirmation split does not match sealed manifest fixture")
    if set(statuses) != set(MODELS) or set(metrics) != set(MODELS):
        raise SummaryError("Both frozen models must have result metrics and status")
    indexed: dict[str, dict[tuple[str, str, str, str], dict]] = {}
    status_manifest_hashes = set()
    status_source_hashes = set()
    input_mismatch_count = 0
    max_input_delta = 0.0
    expected_keys = {(generator, scene, weather, strength)
                     for generator in GENERATORS for scene in expected_scenes
                     for weather in WEATHERS for strength in STRENGTHS}
    for model in MODELS:
        status = statuses[model]
        if status.get("status") != "complete" or status.get("partition") != "confirmation":
            raise SummaryError(f"{model}: status is incomplete or has wrong partition")
        if status.get("model") != model or status.get("scene_count") != 39:
            raise SummaryError(f"{model}: model or confirmation scene count mismatch")
        if status.get("metric_rows") != 312 or len(metrics[model]) != 312:
            raise SummaryError(f"{model}: expected 312 metric rows")
        if status.get("unique_forward_count") != 702 or status.get("expected_forward_count") != 702:
            raise SummaryError(f"{model}: expected 702 unique inputs/forwards")
        if status.get("holdout_content_opened") is not False:
            raise SummaryError(f"{model}: holdout was marked as opened")
        if status.get("project_commit") != commit:
            raise SummaryError(f"{model}: project commit differs from run metadata")
        status_manifest_hashes.add(status.get("manifest_sha256"))
        status_source_hashes.add(status.get("sources_sha256"))
        rows_by_key = {}
        for row in metrics[model]:
            key = (row.get("generator"), row.get("scene"), row.get("weather"), row.get("strength"))
            if key in rows_by_key:
                raise SummaryError(f"{model}: duplicate metric key {key}")
            if key not in expected_keys:
                raise SummaryError(f"{model}: unexpected metric key {key}")
            if row.get("model") != model or row.get("scene") not in expected_scenes:
                raise SummaryError(f"{model}: row is outside the confirmation split")
            expected_seed = (generator_b.stable_seed(row["scene"]) if row["generator"] == "B" else
                             _a_seed(row["scene"]))
            if row.get("seed") != expected_seed:
                raise SummaryError(f"{model}/{key}: realization seed mismatch")
            if row.get("ground_truth_sha256") != next(
                    item["pixel_sha256"] for item in manifest_rows if item["scene"] == row["scene"]):
                raise SummaryError(f"{model}/{key}: clean image hash mismatch")
            if set(row.get("input_sha256", {})) != set(VIEWS) or set(row.get("views", {})) != set(VIEWS):
                raise SummaryError(f"{model}/{key}: expected four quartet view hashes and metrics")
            for view in VIEWS:
                item = row["views"][view]
                if row["input_sha256"][view] != item.get("sha256"):
                    raise SummaryError(f"{model}/{key}/{view}: input hash mismatch")
                for field in ("mse_full", "mse_inner32", "output_mse_full", "output_mse_inner32", "ssim_full"):
                    value = item.get(field)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise SummaryError(f"{model}/{key}/{view}: invalid {field}")
                    if field.startswith("mse") or field.startswith("output_mse"):
                        if value < 0:
                            raise SummaryError(f"{model}/{key}/{view}: negative {field}")
                expected_psnr = (-10*math.log10(item["output_mse_full"])
                                 if item["output_mse_full"] > 0 else None)
                if expected_psnr is None:
                    if item.get("psnr_full") is not None or item.get("psnr_reason") != "perfect_reconstruction":
                        raise SummaryError(f"{model}/{key}/{view}: perfect PSNR must be null with a reason")
                elif (item.get("psnr_reason") is not None or item.get("psnr_full") is None
                      or not np.isclose(item["psnr_full"], expected_psnr, rtol=1e-9, atol=1e-10)):
                    raise SummaryError(f"{model}/{key}/{view}: PSNR formula mismatch")
                oracle = item.get("oracle")
                if not isinstance(oracle, dict) or len(oracle.get("gain", [])) != 3 or len(oracle.get("bias", [])) != 3:
                    raise SummaryError(f"{model}/{key}/{view}: missing per-channel affine oracle")
                if not all(math.isfinite(float(v)) for field in ("gain", "bias") for v in oracle[field]):
                    raise SummaryError(f"{model}/{key}/{view}: nonfinite affine parameter")
                if oracle.get("calibrated_mse_full", oracle.get("raw_mse_full", 0)) > oracle.get("raw_mse_full", 0) + 1e-10:
                    raise SummaryError(f"{model}/{key}/{view}: affine full MSE increased")
            if set(row.get("full", {})) != {"input_mse", "output_mse", "calibrated_mse",
                                             "remaining_error_fraction", "D", "Q", "P_db", "P_reason",
                                             "Q_cal", "P_cal_db", "P_cal_reason", "Q_minus_Q_cal"}:
                raise SummaryError(f"{model}/{key}: malformed full region metrics")
            for region in REGIONS:
                block = row.get(region)
                if not isinstance(block, dict):
                    raise SummaryError(f"{model}/{key}: missing {region} metrics")
                for field in ("input_mse", "output_mse", "calibrated_mse"):
                    if set(block.get(field, {})) != set(VIEWS):
                        raise SummaryError(f"{model}/{key}/{region}: missing {field} views")
                    for value in block[field].values():
                        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                            raise SummaryError(f"{model}/{key}/{region}: invalid {field} value")
                expected_q = (block["output_mse"]["combined"]
                              - max(block["output_mse"]["low"], block["output_mse"]["weather"]))
                expected_q_cal = (block["calibrated_mse"]["combined"]
                                  - max(block["calibrated_mse"]["low"], block["calibrated_mse"]["weather"]))
                if not np.isclose(block.get("Q"), expected_q, rtol=1e-9, atol=1e-12):
                    raise SummaryError(f"{model}/{key}/{region}: Q formula mismatch")
                if not np.isclose(block.get("Q_cal"), expected_q_cal, rtol=1e-9, atol=1e-12):
                    raise SummaryError(f"{model}/{key}/{region}: calibrated Q formula mismatch")
                raw_max = max(block["output_mse"]["low"], block["output_mse"]["weather"])
                cal_max = max(block["calibrated_mse"]["low"], block["calibrated_mse"]["weather"])
                expected_p = (10 * math.log10(block["output_mse"]["combined"] / raw_max)
                              if block["output_mse"]["combined"] > 0 and raw_max > 0 else None)
                expected_p_cal = (10 * math.log10(block["calibrated_mse"]["combined"] / cal_max)
                                  if block["calibrated_mse"]["combined"] > 0 and cal_max > 0 else None)
                for name, expected in (("P_db", expected_p), ("P_cal_db", expected_p_cal),
                                       ("Q_minus_Q_cal", expected_q-expected_q_cal)):
                    actual = block.get(name)
                    if expected is None:
                        if actual is not None:
                            raise SummaryError(f"{model}/{key}/{region}: {name} should be null")
                    elif not isinstance(actual, (int, float)) or not np.isclose(
                            actual, expected, rtol=1e-9, atol=1e-12):
                        raise SummaryError(f"{model}/{key}/{region}: {name} formula mismatch")
                for name, value, reason_key in (("P_db", expected_p, "P_reason"),
                                                ("P_cal_db", expected_p_cal, "P_cal_reason")):
                    reason = block.get(reason_key)
                    if value is None and not isinstance(reason, str):
                        raise SummaryError(f"{model}/{key}/{region}: null {name} must have a reason")
                    if value is not None and reason is not None:
                        raise SummaryError(f"{model}/{key}/{region}: unexpected {reason_key}")
                fractions = block["remaining_error_fraction"]
                for view in ("low", "weather", "combined"):
                    denom = block["input_mse"][view]
                    expected_fraction = block["output_mse"][view]/denom if denom > 1e-6 else None
                    actual = fractions.get(view)
                    if expected_fraction is None:
                        if actual is not None:
                            raise SummaryError(f"{model}/{key}/{region}: invalid remaining-error fraction")
                    elif actual is None or not np.isclose(actual, expected_fraction, rtol=1e-9, atol=1e-12):
                        raise SummaryError(f"{model}/{key}/{region}: remaining-error fraction mismatch")
                for view in VIEWS:
                    actual = block["input_mse"][view]
                    stored = row["views"][view]["mse_full" if region == "full" else "mse_inner32"]
                    if not np.isclose(actual, stored, rtol=1e-9, atol=1e-12):
                        raise SummaryError(f"{model}/{key}/{region}/{view}: input MSE disagrees")
                    raw_key = "output_mse_full" if region == "full" else "output_mse_inner32"
                    if not np.isclose(block["output_mse"][view], row["views"][view][raw_key],
                                      rtol=1e-9, atol=1e-12):
                        raise SummaryError(f"{model}/{key}/{region}/{view}: output MSE disagrees")
                    oracle_key = ("calibrated_mse_full" if region == "full" else "mse_inner32")
                    if not np.isclose(block["calibrated_mse"][view],
                                      row["views"][view]["oracle"][oracle_key],
                                      rtol=1e-9, atol=1e-12):
                        raise SummaryError(f"{model}/{key}/{region}/{view}: calibrated MSE disagrees")
            rows_by_key[key] = row
        if set(rows_by_key) != expected_keys:
            raise SummaryError(f"{model}: incomplete confirmation coverage")
        indexed[model] = rows_by_key
    if len(status_manifest_hashes) != 1 or len(status_source_hashes) != 1:
        raise SummaryError("Models were not evaluated against the same manifest and checkpoints")
    for key in expected_keys:
        first, second = indexed[MODELS[0]][key], indexed[MODELS[1]][key]
        if first["seed"] != second["seed"]:
            raise SummaryError(f"Cross-model seed mismatch for {key}")
        for view in VIEWS:
            if first["input_sha256"][view] != second["input_sha256"][view]:
                input_mismatch_count += 1
            d = abs(first["full"]["input_mse"][view] - second["full"]["input_mse"][view])
            max_input_delta = max(max_input_delta, d)
    if input_mismatch_count or max_input_delta > 1e-12:
        raise SummaryError(f"Inputs differ between model runs: hashes={input_mismatch_count}, maxMSE={max_input_delta}")
    return {"confirmation_scene_count": len(expected_scenes), "rows_per_model": 312,
            "unique_forwards_per_model": 702, "matched_keys": len(expected_keys),
            "cross_model_input_hash_mismatch_count": input_mismatch_count,
            "cross_model_input_mse_max_abs_delta": max_input_delta,
            "manifest_fixture_sha256": fixture["source_manifest_sha256"],
            "project_code_commit": commit}, indexed


def _a_seed(scene: str) -> int:
    from .audit_degradation_order import _stable_seed
    return _stable_seed(20260923, scene, 0)


def _boot(values: list[float | None], indices: np.ndarray) -> dict[str, Any]:
    if any(value is None for value in values):
        return {"n": len(values), "valid_n": sum(value is not None for value in values),
                "null_count": sum(value is None for value in values), "mean": None,
                "median": None, "ci95": None, "positive_count": None}
    arr = np.asarray(values, dtype=np.float64)
    boot = arr[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975], method="linear")
    return {"n": int(arr.size), "valid_n": int(arr.size), "null_count": 0,
            "mean": float(arr.mean()), "median": float(np.median(arr)),
            "ci95": [float(low), float(high)], "positive_count": int(np.count_nonzero(arr > 0))}


def _boot_ratio(numerators: list[float], denominators: list[float], indices: np.ndarray):
    n = np.asarray(numerators, dtype=np.float64)
    d = np.asarray(denominators, dtype=np.float64)
    if np.any(d <= 0):
        return {"n": len(d), "mean": None, "ci95": None, "reason": "nonpositive_denominator"}
    samples = n[indices].sum(axis=1) / d[indices].sum(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975], method="linear")
    return {"n": int(d.size), "mean": float(n.sum() / d.sum()),
            "ci95": [float(low), float(high)]}


def _paired_indices(n: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    return rng.integers(0, n, size=(BOOTSTRAP_SAMPLES, n), endpoint=False)


def screen_condition(q_full: list[float | None], q_inner: list[float | None],
                     p_full_db: list[float | None], indices: np.ndarray):
    """Apply the frozen four-part strong-condition screen without dropping scenes."""
    if any(value is None for value in (*q_full, *q_inner, *p_full_db)):
        return None, "missing_metric"
    q_stats, inner_stats, p_stats = (_boot(q_full, indices), _boot(q_inner, indices),
                                     _boot(p_full_db, indices))
    if q_stats["ci95"] is None:
        return None, "missing_q_interval"
    passed = (q_stats["ci95"][0] > 0 and q_stats["positive_count"] >= 24
              and inner_stats["mean"] > 0 and p_stats["mean"] >= 0.3)
    return bool(passed), "pass" if passed else "criterion_failed"


def summarize(indexed: dict[str, dict], scenes_in_order: list[str]):
    positions = {scene: index for index, scene in enumerate(scenes_in_order)}
    indices = _paired_indices(len(scenes_in_order))
    conditions, screen_fail = {}, False
    inconclusive = False
    for generator in GENERATORS:
        for model in MODELS:
            for weather in WEATHERS:
                for strength in STRENGTHS:
                    ordered = sorted((row for key, row in indexed[model].items()
                                      if key[0] == generator and key[2] == weather and key[3] == strength),
                                     key=lambda row: positions[row["scene"]])
                    if len(ordered) != len(scenes_in_order):
                        raise SummaryError("Condition has a missing confirmation scene")
                    fields = {}
                    for name, region, field in (
                        ("Q_full", "full", "Q"), ("Q_inner32", "inner32", "Q"),
                        ("P_full_db", "full", "P_db"), ("Q_cal_full", "full", "Q_cal"),
                        ("Q_cal_inner32", "inner32", "Q_cal"),
                        ("Q_minus_Q_cal_full", "full", "Q_minus_Q_cal"),
                        ("P_cal_full_db", "full", "P_cal_db"),
                        ("D_full", "full", "D"), ("D_inner32", "inner32", "D")):
                        fields[name] = _boot([row[region].get(field) for row in ordered], indices)
                    oracle = {}
                    for view in ("low", "weather", "combined"):
                        numerator, denominator = [], []
                        for row in ordered:
                            raw = row["views"][view]["output_mse_full"]
                            calibrated = row["views"][view]["oracle"]["calibrated_mse_full"]
                            denominator.append(raw)
                            numerator.append(raw-calibrated)
                        oracle[f"F_{view}_ratio_of_sums"] = _boot_ratio(numerator, denominator, indices)
                    passed = None
                    if strength == "strong":
                        ordered_q = [row["full"].get("Q") for row in ordered]
                        ordered_qi = [row["inner32"].get("Q") for row in ordered]
                        ordered_p = [row["full"].get("P_db") for row in ordered]
                        passed, gate_reason = screen_condition(ordered_q, ordered_qi, ordered_p, indices)
                        if passed is None:
                            inconclusive = True
                        elif not passed:
                            screen_fail = True
                    conditions[f"{generator}/{model}/{weather}/{strength}"] = {
                        "scene_count": len(ordered), "metrics": fields,
                        "affine_oracle": oracle, "screen_pass": passed,
                        "screen_reason": gate_reason if strength == "strong" else "report_only"}
    status = ("FAIL_SCREEN" if screen_fail else "INCONCLUSIVE" if inconclusive else "PASS_REPLICATION")
    return {"status": status, "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_engine": "NumPy PCG64",
            "quantile_method": "linear", "screen_conditions": 8,
            "conditions": conditions,
            "interpretation_limit": "Replication/descriptive diagnostic only. Affine oracle has GT and is not deployable; S1 does not identify a restoration mechanism or establish novelty."}


def load_from_zip(path: Path):
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise SummaryError(f"Corrupt ZIP member: {bad}")
        names = set(archive.namelist())
        missing = {"run.json", "protocol.json", "manifest.json", "sources.json",
                   "manifest_fixture.json", "source_fixture.json"} - names
        if missing:
            raise SummaryError(f"Missing ZIP entries: {sorted(missing)}")
        run = _read_json_zip(archive, "run.json")
        protocol = _read_json_zip(archive, "protocol.json")
        manifest = _read_json_zip(archive, "manifest.json")
        sources = _read_json_zip(archive, "sources.json")
        fixture = _read_json_zip(archive, "manifest_fixture.json")
        source_fixture = _read_json_zip(archive, "source_fixture.json")
        statuses = {model: _read_json_zip(archive, f"s1/confirmation/{model}/status.json")
                    for model in MODELS}
        metrics = {model: _read_json_zip(archive, f"s1/confirmation/{model}/metrics.json")
                   for model in MODELS}
        return run, protocol, manifest, sources, fixture, source_fixture, statuses, metrics


def summarize_archive(zip_path: Path, output_path: Path, report_path: Path):
    run, protocol, manifest, sources, fixture, source_fixture, statuses, metrics = load_from_zip(zip_path)
    audit, indexed = _validate_metrics(manifest, fixture, run, protocol, sources,
                                       source_fixture, statuses, metrics)
    scenes = [row["scene"] for row in _fixture_records(fixture) if row["partition"] == "confirmation"]
    result = {"protocol_version": protocol["protocol_version"],
              "archive_sha256": sha256(zip_path), "audit": audit,
              **summarize(indexed, scenes)}
    save_json(output_path, result)
    lines = [f"# S1 replication summary", "", f"Gate: **{result['status']}**", "",
             f"Confirmation scenes: {audit['confirmation_scene_count']}; rows/model: {audit['rows_per_model']}; ",
             f"forwards/model: {audit['unique_forwards_per_model']}; paired input-hash mismatches: ",
             f"{audit['cross_model_input_hash_mismatch_count']}.", "",
             "| Generator | Model | Weather | Strength | Mean Q full [95% CI] | Q>0 | Mean Q inner32 | Mean P (dB) | Screen |",
             "|---|---|---|---|---:|---:|---:|---:|---|"]
    for name, condition in result["conditions"].items():
        generator, model, weather, strength = name.split("/")
        metrics_row = condition["metrics"]
        q, qi, p = metrics_row["Q_full"], metrics_row["Q_inner32"], metrics_row["P_full_db"]
        ci = "null" if q["ci95"] is None else f"[{q['ci95'][0]:.6g}, {q['ci95'][1]:.6g}]"
        count = "null" if q["positive_count"] is None else f"{q['positive_count']}/39"
        vals = (q["mean"], qi["mean"], p["mean"])
        fmt = lambda v: "null" if v is None else f"{v:.6g}"
        lines.append(f"| {generator} | {model} | {weather} | {strength} | {fmt(q['mean'])} {ci} | "
                     f"{count} | {fmt(qi['mean'])} | {fmt(p['mean'])} | {condition['screen_pass']} |")
    lines.extend(["", "Affine oracle ratios-of-sums F (low/weather/combined), and Q after calibration are in the JSON.",
                  "The oracle sees GT; it is diagnostic and cannot be reported as deployed restoration performance.",
                  "A passing screen confirms the frozen-model phenomenon on this synthetic generator and split only.",
                  "It does not identify a mechanism, prove novelty, or authorize training a candidate method."])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_archive(args.zip.resolve(), args.output.resolve(), args.report.resolve())
    print(f"S1 summary: {result['status']}", flush=True)


if __name__ == "__main__":
    main()
