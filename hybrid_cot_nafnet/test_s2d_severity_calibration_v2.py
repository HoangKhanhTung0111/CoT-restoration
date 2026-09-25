from __future__ import annotations

import unittest

from .calibrate_s2d_severity import FEATURES
from .calibrate_s2d_severity_v2 import _condition_gate


class S2DSeverityCalibrationV2Test(unittest.TestCase):
    def _row(self, mechanism: str, offset: float) -> dict:
        row = {name: float(index + offset) for index, name in enumerate(FEATURES)}
        row["mechanism"] = mechanism
        return row

    def test_common_support_accepts_between_mechanism_medians(self):
        reference = []
        candidate = []
        for index in range(10):
            reference.append(self._row("A", 0.01 * index))
            reference.append(self._row("B", 1.0 + 0.01 * index))
            candidate.append(self._row("C", 0.5 + 0.01 * index))
        gate = {
            "absolute_median_input_psnr_db": 0.5,
            "absolute_median_input_ssim": 0.02,
            "absolute_median_input_lpips": 0.03,
            "energy_margin": 0.25,
        }
        result = _condition_gate(reference, candidate, gate)
        self.assertTrue(result["pass"])
        self.assertTrue(all(result["checks"].values()))

    def test_common_support_rejects_outside_candidate(self):
        reference = [self._row("A", 0.0), self._row("B", 1.0)] * 5
        candidate = [self._row("C", 20.0)] * 5
        gate = {
            "absolute_median_input_psnr_db": 0.5,
            "absolute_median_input_ssim": 0.02,
            "absolute_median_input_lpips": 0.03,
            "energy_margin": 0.25,
        }
        result = _condition_gate(reference, candidate, gate)
        self.assertFalse(result["pass"])
        self.assertGreater(result["failed_checks"], 0)


if __name__ == "__main__":
    unittest.main()
