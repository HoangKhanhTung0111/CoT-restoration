from __future__ import annotations

import unittest

import numpy as np

from .audit_s2c_safe_robust import (
    conditional_risk_variance,
    project_single_protected_gradient,
    run_audit,
    vrex_by_factor,
)


class S2CSafeRobustAuditTest(unittest.TestCase):
    def test_conditional_variance_is_vrex_by_factor(self):
        risks = np.asarray([[1.0, 3.0], [2.0, 5.0], [4.0, 4.0]])
        self.assertAlmostEqual(
            conditional_risk_variance(risks), vrex_by_factor(risks)
        )

    def test_single_protected_gradient_projection(self):
        proposed = np.asarray([-2.0, 1.0])
        protected = np.asarray([1.0, 0.0])
        projected = project_single_protected_gradient(proposed, protected)
        np.testing.assert_allclose(projected, np.asarray([0.0, 1.0]))
        self.assertGreaterEqual(float(np.dot(projected, protected)), 0.0)

    def test_nonconflicting_gradient_is_unchanged(self):
        proposed = np.asarray([2.0, 1.0])
        protected = np.asarray([1.0, 0.0])
        np.testing.assert_allclose(
            project_single_protected_gradient(proposed, protected), proposed
        )

    def test_audit_rejects_candidate_without_consuming_version(self):
        result = run_audit()
        self.assertEqual(result["decision"], "NO_GO_AS_CVPR_METHOD_CONTRIBUTION")
        self.assertEqual(result["candidate_versions_consumed"], 0)
        self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
