from __future__ import annotations

import copy
import unittest
from pathlib import Path

from .audit_s2d_full_protocol import audit, load_config


class S2DFullProtocolAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repository = Path(__file__).resolve().parents[1]
        cls.config = load_config(
            repository / "configs" / "s2d_full_frozen_audit_v1.json"
        )

    def test_locked_protocol_stops_after_f2_severity_failure(self):
        result = audit(self.config)
        self.assertEqual(result["status"], "VALID")
        self.assertEqual(result["decision"], "STOP_F2_REVISE_OR_REJECT_RENDERER_C")
        self.assertEqual(result["counts"]["frozen_systems"], 5)
        self.assertEqual(result["counts"]["development_forwards"], 12000)
        self.assertEqual(result["counts"]["final_forwards"], 24000)
        self.assertIn("real_endpoint_license", result["blocking_gates"])

    def test_training_or_weatherbench_download_invalidates_protocol(self):
        changed = copy.deepcopy(self.config)
        changed["scientific_scope"]["restoration_training_allowed"] = True
        changed["real_endpoint"]["download_allowed"] = True
        result = audit(changed)
        self.assertEqual(result["status"], "INVALID")
        self.assertIn("restoration_training_allowed", result["errors"])
        self.assertIn(
            "weatherbench_download_enabled_before_permission", result["errors"]
        )

    def test_condition_or_mechanism_drift_is_rejected(self):
        changed = copy.deepcopy(self.config)
        changed["conditions"].pop()
        changed["mechanisms"][2]["family"] = changed["mechanisms"][0]["family"]
        result = audit(changed)
        self.assertEqual(result["status"], "INVALID")
        self.assertIn("condition_grid_changed", result["errors"])
        self.assertIn("mechanism_families_not_independent", result["errors"])


if __name__ == "__main__":
    unittest.main()
