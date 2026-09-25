from __future__ import annotations

import unittest

import numpy as np

from .s2d_mechanism_probe import (
    extract_feature_families,
    extract_residual_feature_families,
    grouped_probe,
    pair_moment_match,
)
from .summarize_s2d_pilot import evaluate_gate


class S2DMechanismProbeTest(unittest.TestCase):
    def test_feature_schema_is_finite_and_deterministic(self):
        yy, xx = np.indices((64, 80))
        image = np.stack(
            ((xx * 3 + yy) % 256, (xx + yy * 5) % 256, (xx * 7 + yy * 2) % 256),
            axis=-1,
        ).astype(np.uint8)
        first = extract_feature_families(image)
        second = extract_feature_families(image.copy())
        self.assertEqual(set(first), {"color", "spectrum", "wavelet", "combined"})
        for family in first:
            np.testing.assert_array_equal(first[family][0], second[family][0])
            self.assertEqual(first[family][1], second[family][1])
            self.assertTrue(np.isfinite(first[family][0]).all())
        self.assertEqual(len(first["combined"][0]), sum(len(first[name][0]) for name in ("color", "spectrum", "wavelet")))

    def test_residual_features_are_zero_for_clean_input(self):
        image = np.full((48, 64, 3), 127, dtype=np.uint8)
        features = extract_residual_feature_families(image, image.copy())
        self.assertEqual(set(features), {"statistics", "spectrum", "wavelet", "combined"})
        for vector, names in features.values():
            self.assertEqual(len(vector), len(names))
            np.testing.assert_array_equal(vector, np.zeros_like(vector))

    def test_pair_moment_matching_equalizes_channel_statistics(self):
        rng = np.random.default_rng(7)
        left = rng.integers(0, 100, size=(80, 64, 3), dtype=np.uint8)
        right = rng.integers(140, 256, size=(80, 64, 3), dtype=np.uint8)
        matched_left, matched_right = pair_moment_match(left, right)
        for statistic in (np.mean, np.std):
            left_stat = statistic(matched_left.astype(np.float64), axis=(0, 1))
            right_stat = statistic(matched_right.astype(np.float64), axis=(0, 1))
            np.testing.assert_allclose(left_stat, right_stat, atol=0.7)

    def test_scene_grouped_probe_detects_non_scene_mechanism(self):
        features, labels, scenes = [], [], []
        fold_by_scene = {}
        for scene_index in range(10):
            scene = f"s{scene_index:02d}"
            fold_by_scene[scene] = scene_index % 5
            for condition in range(3):
                for label in (-1, 1):
                    features.append([label * 4.0, condition, scene_index / 10])
                    labels.append(label)
                    scenes.append(scene)
        result = grouped_probe(
            np.asarray(features, dtype=np.float64),
            np.asarray(labels),
            np.asarray(scenes),
            fold_by_scene,
        )
        self.assertEqual(result["balanced_accuracy"], 1.0)
        self.assertEqual(len(result["folds"]), 5)

    def test_decision_gate_requires_all_controls(self):
        probe = {
            "best_raw_family": "combined",
            "best_raw_balanced_accuracy": 0.91,
            "best_matched_balanced_accuracy": 0.82,
            "clean_gt_negative_controls": {
                "color": {"balanced_accuracy": 0.5},
                "combined": {"balanced_accuracy": 0.5},
            },
            "label_permutation_control_for_best_raw_family": {"q95": 0.58},
        }
        scene_rows = [
            {"target_generalization_gap_b_minus_a": 3.0 + index / 10}
            for index in range(25)
        ]
        fold_rows = [
            {
                "generator_b_target_delta_psnr": 4.0,
                "generator_a_target_delta_psnr": 0.2,
            }
            for _ in range(5)
        ]
        s2b = {
            "exploratory_scene_analysis": {
                "scene_rows": scene_rows,
                "fold_mean_delta_psnr": fold_rows,
            }
        }
        result = evaluate_gate(probe, s2b)
        self.assertEqual(result["decision"], "GO_FULL_AUDIT")
        probe["best_matched_balanced_accuracy"] = 0.69
        result = evaluate_gate(probe, s2b)
        self.assertEqual(result["decision"], "NO_GO_OR_REVISE_S2D")


if __name__ == "__main__":
    unittest.main()
