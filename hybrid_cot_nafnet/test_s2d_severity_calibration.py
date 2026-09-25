from __future__ import annotations

import unittest

import numpy as np

from .calibrate_s2d_severity import CONDITIONS, FEATURES, analyze_candidates, energy_statistic


class S2DSeverityCalibrationTest(unittest.TestCase):
    def test_energy_statistic_identity_and_shift(self):
        reference = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        self.assertAlmostEqual(energy_statistic(reference, reference), 0.0)
        self.assertGreater(energy_statistic(reference + 4.0, reference), 0.0)

    def test_selection_prefers_exact_reference_and_passes(self):
        profiles = [{"id": 0}, {"id": 1}]
        rows = []
        for condition_index, condition in enumerate(CONDITIONS):
            base = np.arange(len(FEATURES), dtype=np.float64) + condition_index
            for scene in range(4):
                values = base + scene * 0.01
                for mechanism in ("A", "B"):
                    row = {name: float(values[index]) for index, name in enumerate(FEATURES)}
                    row.update({"condition": condition, "mechanism": mechanism, "candidate_id": None})
                    rows.append(row)
                exact = {name: float(values[index]) for index, name in enumerate(FEATURES)}
                exact.update({"condition": condition, "mechanism": "C", "candidate_id": 0})
                rows.append(exact)
                shifted = {name: float(values[index] + 2.0) for index, name in enumerate(FEATURES)}
                shifted.update({"condition": condition, "mechanism": "C", "candidate_id": 1})
                rows.append(shifted)
        tolerances = {
            "absolute_median_input_psnr_db": 0.5,
            "absolute_median_input_ssim": 0.02,
            "absolute_median_input_lpips": 0.03,
            "standardized_energy_distance": 0.25,
        }
        result = analyze_candidates(rows, profiles, tolerances)
        self.assertEqual(result["selected_candidate_id"], 0)
        selected = result["candidates"][0]
        self.assertTrue(selected["all_conditions_pass"])

    def test_energy_rejects_malformed_arrays(self):
        with self.assertRaises(ValueError):
            energy_statistic(np.zeros(3), np.zeros((3, 1)))


if __name__ == "__main__":
    unittest.main()
