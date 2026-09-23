"""Tests for S1 split controls, generator parity, affine readout and gates."""
from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

from . import low_weather_generator_b as gen_b
from .audit_degradation_order import _stable_seed, apply_order, build_realization
from .audit_low_weather_interaction import LOW_LEVELS, WEATHERS, apply_low, make_views
from .audit_low_weather_replication import (
    _fit_affine_oracle, _psnr, estimate_confirmation_gpu_hours,
    manifest_from_fixture, unique_scene_views, validate_manifest,
)
from .summarize_low_weather_replication import _a_seed, _boot_ratio, _paired_indices, screen_condition
from .summarize_low_weather_replication import (
    _validate_metrics, summarize as summarize_s1, summarize_archive,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "configs/low_weather_replication/manifest_fixture.json").read_text())
SOURCE_FIXTURE = json.loads((ROOT / "configs/low_weather_replication/source_fixture.json").read_text())


class ReplicationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        y, x = np.mgrid[0:80, 0:96]
        rgb = np.stack((0.15 + x / 130, 0.12 + y / 120,
                        0.2 + (x + y) / 250), axis=-1)
        cls.clean = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def test_split_lock_accepts_only_discovery_smoke_and_full_confirmation(self):
        manifest = manifest_from_fixture(FIXTURE, ROOT, ["low", "haze", "rain", "snow", "low_haze",
                                                         "low_rain", "low_snow", "haze_rain", "haze_snow",
                                                         "low_haze_rain", "low_haze_snow"])
        self.assertEqual(len(validate_manifest(manifest, FIXTURE, "discovery", 3)), 3)
        self.assertEqual(len(validate_manifest(manifest, FIXTURE, "confirmation", 0)), 39)
        with self.assertRaises(ValueError):
            validate_manifest(manifest, FIXTURE, "holdout", 0)
        with self.assertRaises(ValueError):
            validate_manifest(manifest, FIXTURE, "confirmation", 3)

    def test_a_wrapper_is_pixel_equal_to_frozen_generator(self):
        from .audit_low_weather_replication import _a_views
        clean = torch.from_numpy(self.clean.copy()).permute(2, 0, 1).float() / 255
        realization = build_realization(clean, _stable_seed(20260923, "fixture", 0))
        actual, _, _ = _a_views(self.clean, "fixture", "rain_haze", realization)
        expected = {"clean": self.clean.copy()}
        for level in LOW_LEVELS:
            low = apply_low(clean, realization, level)
            expected[f"low_{level}"] = (low.clamp(0, 1).mul(255).round().byte()
                                         .permute(1, 2, 0).numpy())
            combined = apply_order(low, WEATHERS["rain_haze"], realization)
            expected[f"combined_{level}"] = (combined.clamp(0, 1).mul(255).round().byte()
                                              .permute(1, 2, 0).numpy())
        weather = apply_order(clean, WEATHERS["rain_haze"], realization)
        expected["weather"] = (weather.clamp(0, 1).mul(255).round().byte()
                                .permute(1, 2, 0).numpy())
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_each_generator_caches_nine_unique_views_across_weather(self):
        for name in ("A", "B"):
            views, metadata = unique_scene_views(name, self.clean, "fixture-cache")
            self.assertEqual(len(views), 9)
            self.assertEqual(metadata["seed"], (gen_b.stable_seed("fixture-cache") if name == "B"
                                                  else _stable_seed(20260923, "fixture-cache", 0)))
            for value in views.values():
                self.assertEqual(value.shape, self.clean.shape)
                self.assertEqual(value.dtype, np.uint8)
        a, _ = unique_scene_views("A", self.clean, "fixture-cache")
        b, _ = unique_scene_views("B", self.clean, "fixture-cache")
        self.assertEqual(a["clean"].tobytes(), b["clean"].tobytes())

    def test_affine_oracle_controls_and_perfect_psnr(self):
        target = self.clean.astype(np.float32) / 255
        x = target.astype(np.float32)
        pred = (0.75 * x + 0.08).astype(np.float32)
        calibrated, info = _fit_affine_oracle(pred, target)
        self.assertLess(info["calibrated_mse_full"], 1e-10)
        self.assertEqual(info["negative_gain_count"], 0)
        constant = np.full_like(target, 0.4)
        calibrated_constant, constant_info = _fit_affine_oracle(constant, target)
        self.assertTrue(all(constant_info["near_constant_fallback"]))
        self.assertLessEqual(constant_info["calibrated_mse_full"], constant_info["raw_mse_full"])
        descending = 1.0 - target
        _, negative = _fit_affine_oracle(descending, target)
        self.assertEqual(negative["negative_gain_count"], 3)
        self.assertEqual(_psnr(0), (None, "perfect_reconstruction"))
        self.assertEqual(calibrated.shape, target.shape)
        self.assertEqual(calibrated_constant.shape, target.shape)

    def test_screen_positive_scene_boundary_23_fails_24_passes(self):
        indices = _paired_indices(39)
        for positive_count, expected in ((23, False), (24, True)):
            q = [1.0] * positive_count + [-0.01] * (39 - positive_count)
            passed, reason = screen_condition(q, [0.2] * 39, [0.5] * 39, indices)
            self.assertEqual(passed, expected)
            self.assertEqual(reason, "pass" if expected else "criterion_failed")
        passed, reason = screen_condition([None] + [1.0] * 38, [0.2] * 39,
                                          [0.5] * 39, indices)
        self.assertIsNone(passed)
        self.assertEqual(reason, "missing_metric")

    def test_smoke_budget_requires_post_warmup_samples_and_counts(self):
        statuses = {model: {"status": "complete", "partition": "discovery",
                            "scene_count": 3, "unique_forward_count": 54,
                            "warmup_forward_count_per_generator": 3,
                            "wall_seconds": 54.0,
                            "generator_seconds": {name: [1.0] * 27 for name in ("A", "B")}}
                    for model in ("onerestore", "mirage")}
        estimate = estimate_confirmation_gpu_hours(statuses)
        self.assertAlmostEqual(estimate["estimate_gpu_hours"], 39*9*4*1.3/3600)
        statuses["mirage"]["generator_seconds"]["A"] = [1.0] * 19
        with self.assertRaises(ValueError):
            estimate_confirmation_gpu_hours(statuses)

    def test_oracle_ratio_uses_ratio_of_sums(self):
        result = _boot_ratio([0.5, 0.9], [1.0, 9.0], np.array([[0, 1], [1, 1], [0, 0]]))
        self.assertAlmostEqual(result["mean"], 0.14)
        self.assertNotAlmostEqual(result["mean"], np.mean([0.5, 0.1]))

    def _fake_archive_payload(self):
        from .audit_low_weather_replication import manifest_from_fixture

        scenes = [row[0] for row in FIXTURE["scene_records"] if row[4] == "confirmation"]
        manifest = manifest_from_fixture(FIXTURE, ROOT, ["low", "haze", "rain", "snow", "low_haze",
                                                         "low_rain", "low_snow", "haze_rain", "haze_snow",
                                                         "low_haze_rain", "low_haze_snow"])
        commit = "a" * 40
        protocol = {"protocol_version": "low-weather-replication-v1.1",
                    "manifest_fixture_sha256": FIXTURE["source_manifest_sha256"],
                    "project_code_commit": commit}
        sources = {"repos": SOURCE_FIXTURE["repos"],
                   "weights": {name: {"path": "fixture", "sha256": row["sha256"]}
                               for name, row in SOURCE_FIXTURE["weights"].items()}}
        run = {"errors": [], "project_code_commit": commit}
        statuses, metrics = {}, {}
        for model in ("onerestore", "mirage"):
            statuses[model] = {"status": "complete", "partition": "confirmation", "model": model,
                               "scene_count": 39, "metric_rows": 312,
                               "unique_forward_count": 702, "expected_forward_count": 702,
                               "holdout_content_opened": False, "project_commit": commit,
                               "manifest_sha256": "b"*64, "sources_sha256": "c"*64}
            rows = []
            for generator in ("A", "B"):
                for scene in scenes:
                    for weather in ("rain_haze", "snow_haze"):
                        for strength in ("weak", "strong"):
                            seed = (_a_seed(scene) if generator == "A" else
                                    gen_b.stable_seed(scene))
                            input_mse = {"clean": 0.0, "low": 0.10, "weather": 0.12, "combined": 0.15}
                            output_mse = {"clean": 0.01, "low": 0.04, "weather": 0.05, "combined": 0.08}
                            calibrated = {"clean": 0.008, "low": 0.03, "weather": 0.04, "combined": 0.06}
                            views = {}
                            hashes = {}
                            for view in input_mse:
                                hashes[view] = f"{generator}:{scene}:{view}"
                                views[view] = {"sha256": hashes[view], "mse_full": input_mse[view],
                                               "mse_inner32": input_mse[view],
                                               "output_mse_full": output_mse[view],
                                               "output_mse_inner32": output_mse[view], "ssim_full": .5,
                                               "psnr_full": -10*np.log10(output_mse[view]),
                                               "psnr_reason": None,
                                               "oracle": {"gain": [1, 1, 1], "bias": [0, 0, 0],
                                                          "near_constant_fallback": [False]*3,
                                                          "negative_gain_count": 0,
                                                          "raw_mse_full": output_mse[view],
                                                          "calibrated_mse_full": calibrated[view],
                                                          "mse_inner32": calibrated[view]}}
                            frac = {name: (output_mse[name]/input_mse[name] if input_mse[name] else None)
                                    for name in ("low", "weather", "combined")}
                            block = {"input_mse": input_mse, "output_mse": output_mse,
                                     "calibrated_mse": calibrated,
                                     "remaining_error_fraction": frac,
                                     "D": frac["combined"]-max(frac["low"], frac["weather"]),
                                     "Q": .03, "P_db": 10*np.log10(.08/.05), "P_reason": None,
                                     "Q_cal": .02, "P_cal_db": 10*np.log10(.06/.04),
                                     "P_cal_reason": None,
                                     "Q_minus_Q_cal": .01}
                            rows.append({"generator": generator, "model": model, "scene": scene,
                                         "scene_file": scene+".png", "weather": weather,
                                         "strength": strength,
                                         "low_level": ((.6 if strength == "weak" else 1.) if generator == "A"
                                                       else (.5 if strength == "weak" else .2)),
                                         "seed": seed, "realization": {},
                                         "ground_truth_sha256": next(r[2] for r in FIXTURE["scene_records"]
                                                                      if r[0] == scene),
                                         "input_sha256": hashes, "views": views,
                                         "full": block, "inner32": block})
            metrics[model] = rows
        return manifest, run, protocol, sources, statuses, metrics, scenes

    def test_full_confirmation_schema_and_summary_pass(self):
        manifest, run, protocol, sources, statuses, metrics, scenes = self._fake_archive_payload()
        audit, indexed = _validate_metrics(manifest, FIXTURE, run, protocol, sources,
                                           SOURCE_FIXTURE, statuses, metrics)
        result = summarize_s1(indexed, scenes)
        self.assertEqual(audit["matched_keys"], 312)
        self.assertEqual(result["status"], "PASS_REPLICATION")
        self.assertEqual(len(result["conditions"]), 16)

    def test_full_summary_rejects_duplicate_or_cross_model_input(self):
        manifest, run, protocol, sources, statuses, metrics, _ = self._fake_archive_payload()
        metrics["mirage"].append(metrics["mirage"][0])
        with self.assertRaises(ValueError):
            _validate_metrics(manifest, FIXTURE, run, protocol, sources,
                              SOURCE_FIXTURE, statuses, metrics)

    def test_archive_cli_path_validates_and_writes_report(self):
        manifest, run, protocol, sources, statuses, metrics, _ = self._fake_archive_payload()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "synthetic_s1.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                payload = {"run.json": run, "protocol.json": protocol,
                           "manifest.json": manifest, "sources.json": sources,
                           "manifest_fixture.json": FIXTURE,
                           "source_fixture.json": SOURCE_FIXTURE}
                for name, value in payload.items():
                    archive.writestr(name, json.dumps(value))
                for model in ("onerestore", "mirage"):
                    archive.writestr(f"s1/confirmation/{model}/status.json",
                                    json.dumps(statuses[model]))
                    archive.writestr(f"s1/confirmation/{model}/metrics.json",
                                    json.dumps(metrics[model]))
            result = summarize_archive(archive_path, root / "summary.json", root / "summary.md")
            self.assertEqual(result["status"], "PASS_REPLICATION")
            self.assertTrue((root / "summary.json").is_file())
            self.assertIn("PASS_REPLICATION", (root / "summary.md").read_text(encoding="utf-8"))
        manifest, run, protocol, sources, statuses, metrics, _ = self._fake_archive_payload()
        metrics["mirage"][0]["input_sha256"]["combined"] = "different"
        with self.assertRaises(ValueError):
            _validate_metrics(manifest, FIXTURE, run, protocol, sources,
                              SOURCE_FIXTURE, statuses, metrics)


if __name__ == "__main__":
    unittest.main()
