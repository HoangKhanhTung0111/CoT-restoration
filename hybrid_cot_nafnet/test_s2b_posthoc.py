from __future__ import annotations

import unittest

from .analyze_s2b_generator_specific import average_ranks, correlation, quadrant_counts


class S2BPosthocTest(unittest.TestCase):
    def test_average_ranks_uses_midrank_for_ties(self):
        self.assertEqual(average_ranks([5.0, 1.0, 1.0, 9.0]), [3.0, 1.5, 1.5, 4.0])

    def test_correlations_and_sign_quadrants(self):
        result = correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        self.assertAlmostEqual(result["pearson"], -1.0)
        self.assertAlmostEqual(result["spearman"], -1.0)
        self.assertEqual(
            quadrant_counts([1.0, 2.0, -1.0, 0.0], [1.0, -1.0, 2.0, 0.0]),
            {
                "left_positive_right_positive": 1,
                "left_positive_right_nonpositive": 1,
                "left_nonpositive_right_positive": 1,
                "left_nonpositive_right_nonpositive": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
