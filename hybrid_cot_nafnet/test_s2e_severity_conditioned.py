from __future__ import annotations

import unittest

import numpy as np

from .analyze_s2e_severity_conditioned import (
    _design_matrices,
    _effective_sample_size,
    cross_fit_rows,
)


def _config() -> dict:
    return {
        "primary_conditions": ["c0", "c1"],
        "severity": {"features": ["input_psnr", "input_ssim", "input_lpips"]},
        "estimators": {"ridge_alpha": 1.0, "logistic_c": 1.0},
    }


class S2ESeverityConditionedTest(unittest.TestCase):
    def test_design_has_condition_specific_slopes(self):
        rows = []
        for condition in ("c0", "c1"):
            for index in range(4):
                rows.append(
                    {
                        "condition": condition,
                        "input_psnr": 10.0 + index,
                        "input_ssim": 0.4 + 0.1 * index,
                        "input_lpips": 0.5 - 0.05 * index,
                    }
                )
        train, test = _design_matrices(
            rows, rows[:2], ("c0", "c1"), ("input_psnr", "input_ssim", "input_lpips")
        )
        self.assertEqual(train.shape, (8, 8))
        self.assertEqual(test.shape, (2, 8))
        self.assertTrue(np.all(test[:, 5:] == 0.0))

    def test_cross_fit_removes_severity_prediction_without_using_generator(self):
        rows = []
        for fold in range(5):
            for scene_index in range(4):
                scene = f"{fold}-{scene_index}"
                for condition_index, condition in enumerate(("c0", "c1")):
                    for generator, offset in (("A", -0.2), ("B", 0.2)):
                        severity = fold + scene_index / 10 + condition_index + offset
                        rows.append(
                            {
                                "scene": scene,
                                "fold": fold,
                                "condition": condition,
                                "generator": generator,
                                "input_psnr": 20.0 - severity,
                                "input_ssim": 0.9 - 0.02 * severity,
                                "input_lpips": 0.1 + 0.03 * severity,
                                "gain_psnr": 2.0 * severity + condition_index,
                            }
                        )
        adjusted = cross_fit_rows(rows, _config())
        self.assertEqual(len(adjusted), len(rows))
        self.assertTrue(all(0.0 < row["propensity_b"] < 1.0 for row in adjusted))
        residuals = np.asarray([row["residual_gain"] for row in adjusted])
        self.assertLess(abs(float(residuals.mean())), 0.15)

    def test_effective_sample_size(self):
        self.assertAlmostEqual(_effective_sample_size(np.ones(10)), 10.0)
        self.assertLess(_effective_sample_size(np.asarray([1.0, 0.1, 0.1, 0.1])), 4.0)


if __name__ == "__main__":
    unittest.main()

