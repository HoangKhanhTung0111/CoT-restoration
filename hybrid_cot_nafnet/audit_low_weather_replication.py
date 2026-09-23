"""Run the locked S1 low-light × weather replication (frozen models only)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
import types
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import low_weather_generator_b as generator_b
from .audit_degradation_order import _stable_seed, apply_order, build_realization
from .audit_low_weather_interaction import LOW_LEVELS as A_LEVELS
from .audit_low_weather_interaction import WEATHERS, mean_error
from .common_failure_audit import HASHES, REPOS, load_model, safe_extract, save_json, sha256


PROJECT_URL = "https://github.com/HoangKhanhTung0111/CoT-restoration.git"
SOURCE_FIXTURE = "configs/low_weather_replication/source_fixture.json"
MODELS = ("onerestore", "mirage")
GENERATORS = ("A", "B")
WEATHER_FACTORS = {"rain_haze": ("rain", "haze"), "snow_haze": ("snow", "haze")}
STRENGTHS = generator_b.STRENGTHS
VIEWS = ("clean", "low", "weather", "combined")
REGIONS = ("full", "inner32")
S1_CODE_VERSION = "low-weather-replication-v1.1"


class ProtocolError(ValueError):
    pass


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_records(fixture: dict) -> list[dict]:
    raw_records = fixture.get("scene_records")
    if fixture.get("schema") != 1 or not isinstance(raw_records, list) or len(raw_records) != 195:
        raise ProtocolError("Manifest fixture must contain exactly 195 locked scene records")
    if fixture.get("scene_record_columns") == ["scene", "file", "pixel_sha256", "size_wh", "partition"]:
        records = []
        for item in raw_records:
            if not isinstance(item, list) or len(item) != 5 or "x" not in item[3]:
                raise ProtocolError("Malformed compact manifest tuple")
            width, height = item[3].split("x", 1)
            records.append({"scene": item[0], "file": item[1], "pixel_sha256": item[2],
                            "size": [int(width), int(height)], "partition": item[4]})
    else:
        records = raw_records
    counts = {name: sum(row.get("partition") == name for row in records)
              for name in ("discovery", "confirmation", "holdout")}
    if counts != {"discovery": 78, "confirmation": 39, "holdout": 78}:
        raise ProtocolError(f"Fixture partition counts differ from lock: {counts}")
    required = {"scene", "file", "pixel_sha256", "size", "partition"}
    if any(set(row) != required for row in records):
        raise ProtocolError("Fixture scene record schema changed")
    ids = [row["scene"] for row in records]
    if len(ids) != len(set(ids)):
        raise ProtocolError("Fixture has duplicate scene IDs")
    return records


def manifest_from_fixture(fixture: dict, root: Path, groups: list[str]) -> dict:
    """Attach a runtime data root without inspecting sealed image contents."""
    records = _fixture_records(fixture)
    return {"schema": 1, "root": str(root.resolve()), "groups": list(groups),
            "scenes": records, "fixture_source_zip_sha256": fixture["source_zip_sha256"],
            "fixture_source_manifest_sha256": fixture["source_manifest_sha256"]}


def validate_fixture_files(root: Path, fixture: dict) -> None:
    """Confirm that every locked clean scene exists below the CDD-11 clear directory."""
    for row in _fixture_records(fixture):
        path = root / "clear" / row["file"]
        if not path.is_file():
            raise FileNotFoundError(path)


def validate_manifest(manifest: dict, fixture: dict, partition: str,
                      max_scenes: int = 0) -> list[dict]:
    expected = _fixture_records(fixture)
    actual = manifest.get("scenes")
    keys = ("scene", "file", "pixel_sha256", "size", "partition")
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise ProtocolError("Runtime manifest scene count differs from locked fixture")
    actual_tuples = [{key: row.get(key) for key in keys} for row in actual]
    if actual_tuples != expected:
        raise ProtocolError("Ordered runtime manifest tuples differ from S0 fixture")
    if not isinstance(manifest.get("groups"), list) or len(manifest["groups"]) != 11:
        raise ProtocolError("Runtime manifest must identify all 11 CDD-11 groups")
    if partition not in ("discovery", "confirmation"):
        raise ProtocolError("S1 may read discovery smoke or confirmation only")
    selected = [row for row in actual if row["partition"] == partition]
    if partition == "confirmation":
        if max_scenes != 0 or len(selected) != 39:
            raise ProtocolError("Confirmation is all 39 scenes; partial confirmation is forbidden")
    else:
        if not 1 <= max_scenes <= 3:
            raise ProtocolError("Discovery smoke is limited to 1..3 scenes")
        selected = selected[:max_scenes]
    return selected


def _image_fingerprint(path: Path) -> tuple[str, list[int], np.ndarray]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        pixels = np.asarray(rgb, dtype=np.uint8).copy()
        size = list(rgb.size)
    digest = hashlib.sha256(str(tuple(size)).encode() + pixels.tobytes()).hexdigest()
    return digest, size, pixels


def _check_scene_file(root: Path, record: dict) -> np.ndarray:
    if Path(record["file"]).name != record["file"]:
        raise ProtocolError("Unsafe scene filename in fixture")
    path = root / "clear" / record["file"]
    digest, size, pixels = _image_fingerprint(path)
    if digest != record["pixel_sha256"] or size != record["size"]:
        raise ProtocolError(f"Clean image provenance mismatch for {record['scene']}")
    return pixels


def _sha_array(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _mse(x: np.ndarray, y: np.ndarray, border: int = 0) -> float:
    if border:
        if min(x.shape[:2]) <= border * 2:
            raise ProtocolError("Image is too small for inner32 scoring")
        x, y = x[border:-border, border:-border], y[border:-border, border:-border]
    delta = x.astype(np.float64) - y.astype(np.float64)
    return float(np.mean(delta * delta, dtype=np.float64))


def _fit_affine_oracle(output: np.ndarray, target: np.ndarray):
    """Fit per-channel gain/bias on full image, then apply with clipping."""
    x = output.astype(np.float64)
    y = target.astype(np.float64)
    gains, biases, fallbacks = [], [], []
    calibrated = np.empty_like(output, dtype=np.float32)
    for channel in range(3):
        xc, yc = x[..., channel], y[..., channel]
        mean_x, mean_y = float(xc.mean()), float(yc.mean())
        variance = float(np.mean((xc - mean_x) ** 2))
        if variance <= 1e-12:
            gain, fallback = 1.0, True
        else:
            covariance = float(np.mean((xc - mean_x) * (yc - mean_y)))
            gain, fallback = covariance / variance, False
        bias = mean_y - gain * mean_x
        calibrated[..., channel] = np.clip(gain * output[..., channel] + bias, 0, 1)
        gains.append(float(gain))
        biases.append(float(bias))
        fallbacks.append(fallback)
    raw_full, calibrated_full = _mse(output, target), _mse(calibrated, target)
    if calibrated_full > raw_full + 1e-10:
        raise ProtocolError(f"Affine family increased full MSE: {raw_full} -> {calibrated_full}")
    return calibrated, {"gain": gains, "bias": biases, "near_constant_fallback": fallbacks,
                        "negative_gain_count": sum(value < 0 for value in gains),
                        "raw_mse_full": raw_full, "calibrated_mse_full": calibrated_full}


def _psnr(mse: float) -> tuple[float | None, str | None]:
    if mse == 0:
        return None, "perfect_reconstruction"
    return -10.0 * math.log10(mse), None


def _ssim(target: np.ndarray, output: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(target, output, data_range=1.0, channel_axis=-1))


def _a_views(clean_u8: np.ndarray, scene_id: str, weather: str, realization=None):
    clean = torch.from_numpy(clean_u8.copy()).permute(2, 0, 1).float() / 255.0
    seed = _stable_seed(20260923, scene_id, 0)
    realization = realization if realization is not None else build_realization(clean, seed)
    low = {}
    for level in A_LEVELS:
        gain = realization["low_illumination"].pow(float(realization["low_gamma"]) - 1)
        low[level] = (clean * (1 - level * (1 - gain))
                      + level * realization["low_noise"]).clamp(0, 1)
    result = {"clean": clean_u8.copy()}
    order = WEATHER_FACTORS[weather]
    for level in A_LEVELS:
        low_u8 = (low[level].clamp(0, 1).mul(255).round().byte()
                  .permute(1, 2, 0).cpu().numpy().copy())
        combined = apply_order(low[level], order, realization).clamp(0, 1)
        result[f"low_{level}"] = low_u8
        result[f"combined_{level}"] = (combined.mul(255).round().byte()
                                        .permute(1, 2, 0).cpu().numpy().copy())
    weather_view = apply_order(clean, order, realization).clamp(0, 1)
    result["weather"] = (weather_view.mul(255).round().byte()
                         .permute(1, 2, 0).cpu().numpy().copy())
    return result, realization, seed


def _views_for(generator_name: str, clean_u8: np.ndarray, scene_id: str,
               weather: str, a_realization=None):
    if generator_name == "A":
        views, a_realization, seed = _a_views(clean_u8, scene_id, weather, a_realization)
        return views, {"version": "audit_degradation_order-v1", "seed": seed}, a_realization
    realization = generator_b.build_realization(clean_u8, scene_id)
    return (generator_b.make_views(clean_u8, realization, weather),
            generator_b.realization_metadata(realization), realization)


def unique_scene_views(generator_name: str, clean_u8: np.ndarray, scene_id: str):
    """Return the exact nine inputs per scene and generator, replayed across weather."""
    generated: dict[str, np.ndarray] = {}
    if generator_name == "A":
        clean = torch.from_numpy(clean_u8.copy()).permute(2, 0, 1).float() / 255.0
        seed = _stable_seed(20260923, scene_id, 0)
        realization = build_realization(clean, seed)
        info = {"version": "audit_degradation_order-v1", "seed": seed}
        for weather in generator_b.WEATHERS:
            views, _, _ = _views_for("A", clean_u8, scene_id, weather, realization)
            for name, image in views.items():
                key = name if name in ("clean", "low_0.6", "low_1.0") else f"{name}:{weather}"
                generated[key] = image
    elif generator_name == "B":
        realization = generator_b.build_realization(clean_u8, scene_id)
        info = generator_b.realization_metadata(realization)
        for weather in generator_b.WEATHERS:
            views = generator_b.make_views(clean_u8, realization, weather)
            for name, image in views.items():
                key = name if name in ("clean", "low_weak", "low_strong") else f"{name}:{weather}"
                generated[key] = image
    else:
        raise ValueError(f"Unknown generator {generator_name!r}")
    return generated, info


def _forward(model, input_u8: np.ndarray, device: torch.device):
    pil = Image.fromarray(input_u8, mode="RGB")
    tensor = torch.from_numpy(input_u8.copy()).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)
    height, width = tensor.shape[-2:]
    padded = F.pad(tensor, (0, (-width) % 8, 0, (-height) % 8), mode="replicate")
    started = time.perf_counter()
    output = model(padded, pil)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.shape != padded.shape or not torch.isfinite(output).all():
        raise ProtocolError("Frozen restorer returned nonfinite values or wrong shape")
    pred = output[0, :, :height, :width].permute(1, 2, 0).float().cpu().numpy()
    return np.clip(pred, 0, 1).astype(np.float32), seconds, float(pred.min()), float(pred.max())


def _region_metrics(inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray], target: np.ndarray,
                    oracle_outputs: dict[str, np.ndarray], region: str) -> dict:
    border = 32 if region == "inner32" else 0
    input_mse = {name: _mse(value.astype(np.float32) / 255.0, target, border)
                 for name, value in inputs.items()}
    output_mse = {name: _mse(value, target, border) for name, value in outputs.items()}
    calibrated_mse = {name: _mse(value, target, border) for name, value in oracle_outputs.items()}
    fractions = {name: (output_mse[name] / input_mse[name] if input_mse[name] > 1e-6 else None)
                 for name in ("low", "weather", "combined")}
    d = (fractions["combined"] - max(fractions["low"], fractions["weather"])
         if all(value is not None for value in fractions.values()) else None)
    q = output_mse["combined"] - max(output_mse["low"], output_mse["weather"])
    q_cal = calibrated_mse["combined"] - max(calibrated_mse["low"], calibrated_mse["weather"])
    max_component = max(output_mse["low"], output_mse["weather"])
    p = (10 * math.log10(output_mse["combined"] / max_component)
         if output_mse["combined"] > 0 and max_component > 0 else None)
    max_cal_component = max(calibrated_mse["low"], calibrated_mse["weather"])
    p_cal = (10 * math.log10(calibrated_mse["combined"] / max_cal_component)
             if calibrated_mse["combined"] > 0 and max_cal_component > 0 else None)
    p_reason = ("zero_composite_mse" if output_mse["combined"] == 0 else
                "zero_component_mse" if max_component == 0 else None)
    p_cal_reason = ("zero_calibrated_composite_mse" if calibrated_mse["combined"] == 0 else
                    "zero_calibrated_component_mse" if max_cal_component == 0 else None)
    return {"input_mse": input_mse, "output_mse": output_mse,
            "calibrated_mse": calibrated_mse, "remaining_error_fraction": fractions,
            "D": d, "Q": q, "P_db": p, "P_reason": p_reason,
            "Q_cal": q_cal, "P_cal_db": p_cal, "P_cal_reason": p_cal_reason,
            "Q_minus_Q_cal": q - q_cal}


def prepare(work: Path, project: Path, fixture_path: Path, source_fixture_path: Path):
    """Download pinned inputs and reconstruct the locked manifest without opening holdout images."""
    from huggingface_hub import hf_hub_download
    import gdown

    fixture = _load_json(fixture_path)
    source_fixture = _load_json(source_fixture_path)
    _fixture_records(fixture)
    work.mkdir(parents=True, exist_ok=True)
    repos, paths = {}, {}
    for name, (repo, revision) in REPOS.items():
        destination = work / name
        if not destination.exists():
            subprocess.run(["git", "clone", f"https://github.com/{repo}.git", str(destination)], check=True)
            subprocess.run(["git", "-C", str(destination), "checkout", "--detach", revision], check=True)
        actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(destination), "status", "--porcelain",
                                         "--untracked-files=no"], text=True).strip()
        if actual != revision or dirty:
            raise ProtocolError(f"Unexpected external source revision or edits: {destination}")
        repos[name] = {"repo": repo, "revision": actual}
    for filename, expected_hash in HASHES.items():
        is_dataset = filename == "test.zip"
        expected_source = source_fixture.get("weights", {}).get(filename)
        if not is_dataset and (expected_source is None or expected_source.get("sha256") != expected_hash):
            raise ProtocolError(f"Source fixture hash mismatch for {filename}")
        local_path = hf_hub_download(repo_id="gy65896/CDD-11" if is_dataset else "gy65896/OneRestore",
                                     repo_type="dataset" if is_dataset else "model", filename=filename,
                                     local_dir=str(work / "downloads"))
        if sha256(local_path) != expected_hash:
            raise ProtocolError(f"Downloaded file hash mismatch: {filename}")
        paths[filename] = local_path
    mirage_record = source_fixture["weights"].get("CDD11_small.ckpt")
    if not mirage_record or not mirage_record.get("sha256"):
        raise ProtocolError("Source fixture has no MIRAGE checkpoint hash")
    mirage_path = work / "downloads" / "CDD11_small.ckpt"
    if not mirage_path.is_file():
        partial = mirage_path.with_suffix(".ckpt.part")
        ok = gdown.download(id="1GLRMUDfjgWR7aW4DqnUDJzsPO98-cw5G", output=str(partial), quiet=False)
        if not ok or not partial.is_file():
            raise RuntimeError("MIRAGE checkpoint download failed; no random-weight fallback")
        partial.replace(mirage_path)
    if sha256(mirage_path) != mirage_record["sha256"]:
        raise ProtocolError("MIRAGE checkpoint hash differs from S0 source fixture")
    paths["CDD11_small.ckpt"] = str(mirage_path)

    data = work / "data"
    marker = data / ".extracted"
    if not marker.exists():
        safe_extract(paths["test.zip"], data)
        marker.write_text(HASHES["test.zip"], encoding="utf-8")
    roots = [path.parent for path in data.rglob("clear") if path.is_dir()]
    if len(roots) != 1:
        raise ProtocolError(f"Expected one CDD-11 clear directory; found {roots}")
    groups = list(_load_json(project / "configs/low_weather_replication_v1.json")["groups"])
    manifest = manifest_from_fixture(fixture, roots[0], groups)
    validate_fixture_files(roots[0], fixture)
    manifest_path = work / "manifest.json"
    if manifest_path.exists() and _load_json(manifest_path) != manifest:
        raise ProtocolError("Existing manifest differs from the locked S1 fixture")
    save_json(manifest_path, manifest)
    weights = {name: {"path": str(path), "sha256": sha256(path)}
               for name, path in paths.items() if name != "test.zip"}
    if weights["CDD11_small.ckpt"]["sha256"] != mirage_record["sha256"]:
        raise ProtocolError("Recorded MIRAGE hash mismatch")
    save_json(work / "sources.json", {"repos": repos, "weights": weights,
                                      "dataset": {"sha256": HASHES["test.zip"]},
                                      "fixture_sha256": fixture["source_manifest_sha256"]})
    print("Prepared locked S1 data; partitions are 78/39/78. Holdout contents were not opened.", flush=True)


def preflight(work: Path, max_scenes: int = 3,
              fixture_path: Path | None = None):
    """Validate generator inputs and save input-only panels on discovery scenes."""
    project = Path(__file__).resolve().parents[1]
    fixture_path = fixture_path or project / "configs/low_weather_replication/manifest_fixture.json"
    fixture = _load_json(fixture_path)
    manifest = _load_json(work / "manifest.json")
    selected = validate_manifest(manifest, fixture, "discovery", max_scenes)
    root = Path(manifest["root"])
    output_dir = work / "s1" / "preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    records, unique_inputs = [], 0
    for record in selected:
        clean = _check_scene_file(root, record)
        for generator_name in GENERATORS:
            views, info = unique_scene_views(generator_name, clean, record["scene"])
            if len(views) != 9:
                raise ProtocolError(f"{generator_name}/{record['scene']}: expected 9 unique views")
            stats = {}
            for key, image in views.items():
                if image.dtype != np.uint8 or image.shape != clean.shape or not np.isfinite(image).all():
                    raise ProtocolError(f"Malformed {generator_name} input {key}/{record['scene']}")
                stats[key] = {"sha256": _sha_array(image),
                              "uint8_endpoint_fraction": float(np.mean((image == 0) | (image == 255))),
                              "mean_srgb": float(image.mean() / 255.0)}
                unique_inputs += 1
                if key.startswith("combined_"):
                    label = key.replace(":", "_").replace(".", "p")
                    panel = np.concatenate((image, clean), axis=1)
                    Image.fromarray(panel).save(output_dir / f"{generator_name}_{record['scene']}_{label}_INPUT-GT.png")
            records.append({"scene": record["scene"], "generator": generator_name,
                            "seed": info["seed"], "field_or_input_hashes": stats,
                            "realization": info if generator_name == "B" else None})
    result = {"status": "complete", "partition": "discovery", "scene_count": len(selected),
              "generators": list(GENERATORS), "unique_inputs": unique_inputs,
              "holdout_content_opened": False, "confirmation_content_opened": False,
              "records": records}
    save_json(output_dir / "preflight.json", result)
    print(f"Input preflight complete: {len(selected)} discovery scenes, "
          f"{unique_inputs} unique generator inputs; no restorer was loaded.", flush=True)


def evaluate(work: Path, model_name: str, partition: str, max_scenes: int = 0,
             fixture_path: Path | None = None):
    if not torch.cuda.is_available():
        raise RuntimeError("S1 real-model evaluation requires a Kaggle GPU")
    if model_name not in MODELS:
        raise ValueError(model_name)
    project = Path(__file__).resolve().parents[1]
    fixture_path = fixture_path or project / "configs/low_weather_replication/manifest_fixture.json"
    fixture = _load_json(fixture_path)
    manifest_path = work / "manifest.json"
    manifest = _load_json(manifest_path)
    selected = validate_manifest(manifest, fixture, partition, max_scenes)
    root = Path(manifest["root"])
    sources_path = work / "sources.json"
    sources = _load_json(sources_path)
    expected_repos = _load_json(project / SOURCE_FIXTURE)["repos"]
    for source, record in expected_repos.items():
        if sources.get("repos", {}).get(source, {}).get("revision") != record["revision"]:
            raise ProtocolError(f"Pinned source revision mismatch: {source}")
    for weight_name, record in sources.get("weights", {}).items():
        if sha256(record["path"]) != record["sha256"]:
            raise ProtocolError(f"Checkpoint hash changed: {weight_name}")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20260923)
    model = load_model(model_name, work, device)
    output_dir = work / "s1" / partition / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    metrics_path = output_dir / "metrics.json"
    save_json(status_path, {"status": "running", "model": model_name, "partition": partition})
    rows, timing_by_generator, forward_count, peak_vram = [], {name: [] for name in GENERATORS}, 0, 0
    run_started = time.perf_counter()
    with torch.inference_mode():
        for scene_index, record in enumerate(selected):
            clean_u8 = _check_scene_file(root, record)
            gt = clean_u8.astype(np.float32) / 255.0
            for generator_name in GENERATORS:
                generated, realization_info = unique_scene_views(generator_name, clean_u8, record["scene"])
                output_cache: dict[str, np.ndarray] = {}
                oracle_cache: dict[str, np.ndarray] = {}
                input_metric_cache: dict[str, dict] = {}
                for cache_key, input_u8 in generated.items():
                    pred, seconds, raw_min, raw_max = _forward(model, input_u8, device)
                    output_cache[cache_key] = pred
                    calibrated, oracle = _fit_affine_oracle(pred, gt)
                    oracle_cache[cache_key] = calibrated
                    full_input = input_u8.astype(np.float32) / 255.0
                    input_metric_cache[cache_key] = {
                        "sha256": _sha_array(input_u8),
                        "mse_full": _mse(full_input, gt),
                        "mse_inner32": _mse(full_input, gt, 32),
                    }
                    oracle["mse_inner32"] = _mse(calibrated, gt, 32)
                    output_mse_full = _mse(pred, gt)
                    input_metric_cache[cache_key].update({
                        "output_mse_full": output_mse_full,
                        "output_mse_inner32": _mse(pred, gt, 32),
                        "psnr_full": _psnr(output_mse_full)[0],
                        "psnr_reason": _psnr(output_mse_full)[1],
                        "ssim_full": _ssim(gt, pred),
                        "oracle": oracle,
                    })
                    timing_by_generator[generator_name].append(seconds)
                    forward_count += 1
                    peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated(device)))
                    if partition == "confirmation" and scene_index < 3 and cache_key.startswith("combined_"):
                        view_name, weather = cache_key.split(":", 1)
                        strength = view_name.removeprefix("combined_")
                        panel_dir = output_dir / "panels"
                        panel_dir.mkdir(exist_ok=True)
                        raw_view = np.concatenate((input_u8 / 255.0, pred,
                                                   calibrated, gt), axis=1)
                        panel = np.rint(np.clip(raw_view, 0, 1) * 255).astype(np.uint8)
                        panel_path = panel_dir / (
                            f"{generator_name}_{record['scene']}_{weather}_{strength}_ORACLE-GT.png")
                        Image.fromarray(panel).save(panel_path)

                for weather in generator_b.WEATHERS:
                    for strength in STRENGTHS:
                        if generator_name == "A":
                            level = 0.6 if strength == "weak" else 1.0
                            keys = {"clean": "clean", "low": f"low_{level}",
                                    "weather": f"weather:{weather}",
                                    "combined": f"combined_{level}:{weather}"}
                        else:
                            level = STRENGTHS[strength]
                            keys = {"clean": "clean", "low": f"low_{strength}",
                                    "weather": f"weather:{weather}",
                                    "combined": f"combined_{strength}:{weather}"}
                        inputs = {name: generated[key] for name, key in keys.items()}
                        outputs = {name: output_cache[key] for name, key in keys.items()}
                        oracle_outputs = {name: oracle_cache[key] for name, key in keys.items()}
                        regions = {region: _region_metrics(inputs, outputs, gt, oracle_outputs, region)
                                   for region in REGIONS}
                        per_view = {name: input_metric_cache[key] for name, key in keys.items()}
                        rows.append({"generator": generator_name, "model": model_name,
                                     "scene": record["scene"], "scene_file": record["file"],
                                     "weather": weather, "strength": strength,
                                     "low_level": level, "seed": realization_info["seed"],
                                     "realization": realization_info,
                                     "ground_truth_sha256": record["pixel_sha256"],
                                     "input_sha256": {name: input_metric_cache[key]["sha256"]
                                                      for name, key in keys.items()},
                                     "views": per_view, **regions})
                del generated, output_cache, oracle_cache, input_metric_cache
                if len(rows) % 8 == 0:
                    save_json(metrics_path, rows)
            print(f"{partition} {model_name} {record['scene']} rows={len(rows)} forwards={forward_count}",
                  flush=True)
    expected_rows = len(selected) * 8
    expected_forwards = len(selected) * 18
    if len(rows) != expected_rows or forward_count != expected_forwards:
        raise ProtocolError(f"Cache/schema count mismatch: rows={len(rows)}, forwards={forward_count}")
    elapsed = time.perf_counter() - run_started
    save_json(metrics_path, rows)
    status = {"status": "complete", "model": model_name, "partition": partition,
              "scene_count": len(selected), "metric_rows": len(rows), "unique_forward_count": forward_count,
              "expected_forward_count": expected_forwards, "wall_seconds": elapsed,
              "peak_vram_bytes": peak_vram, "torch_version": torch.__version__,
              "numpy_version": np.__version__, "device": torch.cuda.get_device_name(device),
              "manifest_sha256": sha256(manifest_path), "sources_sha256": sha256(sources_path),
              "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project,
                                                          text=True).strip(),
              "generator_seconds": {key: value for key, value in timing_by_generator.items()},
              "warmup_forward_count_per_generator": 3 if partition == "discovery" else 0,
              "protocol_version": S1_CODE_VERSION,
              "metric_protocol": "FP32_fullframe_pad8_rgb_uint8_input_clamp01_output; MSE full+inner32; skimage default SSIM",
              "confirmation_pixel_reads": len(selected) if partition == "confirmation" else 0,
              "holdout_content_opened": False}
    save_json(status_path, status)
    print(f"COMPLETE {partition}/{model_name}: {len(rows)} rows, {forward_count} forwards, "
          f"{elapsed/3600:.3f} wall hours", flush=True)


def estimate_confirmation_gpu_hours(discovery_statuses: dict[str, dict], *, safety_factor: float = 1.3):
    """Estimate from post-warmup p90 forwards and complete smoke wall time."""
    if set(discovery_statuses) != set(MODELS):
        raise ProtocolError("Runtime estimate needs smoke statuses for both models")
    forward_seconds = 0.0
    scaled_wall_seconds = 0.0
    detail = {}
    for model_name in MODELS:
        status = discovery_statuses[model_name]
        if status.get("status") != "complete" or status.get("partition") != "discovery":
            raise ProtocolError(f"Missing completed discovery smoke for {model_name}")
        if status.get("scene_count") != 3 or status.get("unique_forward_count") != 54:
            raise ProtocolError(f"Smoke forward count differs for {model_name}")
        if status.get("warmup_forward_count_per_generator") != 3:
            raise ProtocolError(f"Smoke warmup count is missing for {model_name}")
        smoke_wall = status.get("wall_seconds")
        if not isinstance(smoke_wall, (int, float)) or not math.isfinite(smoke_wall) or smoke_wall <= 0:
            raise ProtocolError(f"Smoke end-to-end wall time is missing for {model_name}")
        scaled_wall_seconds += float(smoke_wall) * 13.0
        for generator_name in GENERATORS:
            values = status.get("generator_seconds", {}).get(generator_name)
            if not isinstance(values, list) or len(values) != 27:
                raise ProtocolError(f"Expected 27 smoke timings for {model_name}/{generator_name}")
            measured = np.asarray(values[3:], dtype=np.float64)
            if measured.size < 20 or not np.isfinite(measured).all() or np.any(measured <= 0):
                raise ProtocolError(f"Need 20 finite post-warmup timings for {model_name}/{generator_name}")
            p90 = float(np.quantile(measured, 0.90, method="linear"))
            # 39 scenes × 9 unique inputs for each model/generator.
            seconds = p90 * 39 * 9
            detail[f"{model_name}/{generator_name}"] = {
                "post_warmup_count": int(measured.size), "p90_forward_seconds": p90,
                "confirmation_forwards": 351, "estimated_seconds": seconds}
            forward_seconds += seconds
    projected_before_buffer = max(forward_seconds, scaled_wall_seconds)
    estimate = projected_before_buffer * safety_factor / 3600.0
    return {"estimate_gpu_hours": estimate, "safety_factor": safety_factor,
            "estimated_total_forwards": 1404,
            "forward_p90_projection_seconds": forward_seconds,
            "smoke_wall_scaled_projection_seconds": scaled_wall_seconds,
            "projection_before_buffer_seconds": projected_before_buffer,
            "projection_after_buffer_seconds": projected_before_buffer*safety_factor,
            "groups": detail}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--work", type=Path, required=True)
    p_prepare.add_argument("--project", type=Path, default=Path.cwd())
    p_prepare.add_argument("--fixture", type=Path, default=Path(SOURCE_FIXTURE))
    p_prepare.add_argument("--manifest-fixture", type=Path,
                           default=Path("configs/low_weather_replication/manifest_fixture.json"))
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--work", type=Path, required=True)
    p_eval.add_argument("--model", choices=MODELS, required=True)
    p_eval.add_argument("--partition", choices=("discovery", "confirmation"), required=True)
    p_eval.add_argument("--max-scenes", type=int, default=0)
    p_eval.add_argument("--manifest-fixture", type=Path,
                        default=Path("configs/low_weather_replication/manifest_fixture.json"))
    p_preflight = sub.add_parser("preflight")
    p_preflight.add_argument("--work", type=Path, required=True)
    p_preflight.add_argument("--max-scenes", type=int, default=3)
    p_preflight.add_argument("--manifest-fixture", type=Path,
                             default=Path("configs/low_weather_replication/manifest_fixture.json"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.work.resolve(), args.project.resolve(), args.manifest_fixture.resolve(), args.fixture.resolve())
    elif args.action == "preflight":
        preflight(args.work.resolve(), args.max_scenes, args.manifest_fixture.resolve())
    else:
        evaluate(args.work.resolve(), args.model, args.partition, args.max_scenes,
                 args.manifest_fixture.resolve())


if __name__ == "__main__":
    main()
