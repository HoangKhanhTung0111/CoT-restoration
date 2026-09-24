from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from .datasets.cdd11 import CDD11_TYPES
from .datasets.s2b_coverage import A_PROBE_TYPES, GENERATED_TYPES
from .summarize_s2b_coverage import summarize


FIELDS = (
    "fold", "scene", "degradation_type", "psnr", "ssim", "input_psnr",
    "input_ssim", "inner32_psnr", "inner32_ssim",
    "inner32_input_psnr", "seconds",
)


class S2BSummaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write(self, name, types, arm, a_negative=False):
        path = self.root / f"{name}_{arm}.csv"
        rows = []
        for scene_index in range(25):
            for kind in types:
                delta = 0.0
                if arm == "r1":
                    if kind.startswith("b_low_strong_"):
                        delta = 0.3
                    elif kind.startswith("a_low_strong_"):
                        delta = -0.1 if a_negative else 0.1
                rows.append(
                    {
                        "fold": scene_index // 5,
                        "scene": f"{scene_index:06d}",
                        "degradation_type": kind,
                        "psnr": 20.0 + delta,
                        "ssim": 0.8 + max(delta, 0) * 0.001,
                        "input_psnr": 15.0,
                        "input_ssim": 0.6,
                        "inner32_psnr": 20.0 + delta,
                        "inner32_ssim": 0.8,
                        "inner32_input_psnr": 15.0,
                        "seconds": 0.1,
                    }
                )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return str(path)

    def _paths(self, a_negative=False):
        groups = {
            "original": CDD11_TYPES,
            "generator_b": GENERATED_TYPES,
            "generator_a": A_PROBE_TYPES,
        }
        return {
            arm: {
                probe: self._write(probe, types, arm, a_negative)
                for probe, types in groups.items()
            }
            for arm in ("r0", "r1")
        }

    def test_passes_coverage_null_when_all_locked_criteria_pass(self):
        result = summarize(self._paths())
        self.assertEqual(result["decision"], "COVERAGE_SUFFICIENT_NULL_RESULT")
        self.assertTrue(all(result["criteria"].values()))
        self.assertEqual(result["candidate_method_versions_used"], 0)

    def test_labels_generator_specific_gain(self):
        result = summarize(self._paths(a_negative=True))
        self.assertEqual(result["decision"], "COVERAGE_GAIN_GENERATOR_SPECIFIC")
        self.assertFalse(result["criteria"]["a_cross_generator_both_target_means_positive"])


if __name__ == "__main__":
    unittest.main()
