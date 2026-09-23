"""CPU preflight for matched 2×2 counterfactuals and the MSE contrast."""

import unittest

import torch

from .audit_degradation_order import apply_order, build_realization
from .audit_low_weather_interaction import (
    LOW_LEVELS, WEATHERS, apply_low, interaction, make_views, quantize_view,
    relative_error, _forward, mean_error,
)


class CounterfactualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = torch.linspace(.12, .89, 3 * 64 * 64).reshape(3, 64, 64)
        cls.manifest = build_realization(cls.clean, 20260923)

    def test_low_zero_and_full_agree_with_original_operator(self):
        self.assertTrue(torch.equal(apply_low(self.clean, self.manifest, 0), self.clean))
        expected = apply_order(self.clean, ("low",), self.manifest)
        self.assertTrue(torch.allclose(apply_low(self.clean, self.manifest, 1), expected, atol=1e-7))

    def test_weather_replay_for_both_levels_and_orders(self):
        frozen = {k: v.clone() for k, v in self.manifest.items() if isinstance(v, torch.Tensor)}
        for weather, order in WEATHERS.items():
            first = make_views(self.clean, self.manifest, weather, LOW_LEVELS[0])
            second = make_views(self.clean, self.manifest, weather, LOW_LEVELS[1])
            self.assertTrue(torch.equal(first["weather"], second["weather"]))
            for level, views in zip(LOW_LEVELS, (first, second)):
                self.assertTrue(torch.equal(views["clean"], self.clean))
                self.assertTrue(torch.equal(views["low"], apply_low(self.clean, self.manifest, level)))
                self.assertTrue(torch.equal(views["weather"], apply_order(self.clean, order, self.manifest)))
                self.assertTrue(torch.equal(views["combined"], apply_order(views["low"], order, self.manifest)))
                for item in views.values():
                    self.assertTrue(torch.isfinite(item).all())
                    self.assertGreaterEqual(float(item.min()), 0)
                    self.assertLessEqual(float(item.max()), 1)
                    encoded = quantize_view(item)
                    self.assertEqual(encoded.dtype, torch.uint8)
                    self.assertEqual(tuple(encoded.shape), tuple(item.shape))
            self.assertFalse(torch.equal(first["low"], second["low"]))
            self.assertFalse(torch.equal(first["combined"], second["combined"]))
        for key, original in frozen.items():
            self.assertTrue(torch.equal(self.manifest[key], original), key)

    def test_contrast_is_zero_for_perfect_restorer(self):
        errors = {name: 0.0 for name in ("clean", "low", "weather", "combined")}
        self.assertEqual(interaction(errors), 0.0)

    def test_interaction_not_equivalent_to_composite_error(self):
        errors = {"clean": .01, "low": .03, "weather": .02, "combined": .06}
        self.assertAlmostEqual(interaction(errors), .02)
        self.assertNotEqual(interaction(errors), errors["combined"])

    def test_remaining_error_fraction_has_identity_and_perfect_controls(self):
        inp = {"low": .03, "weather": .04, "combined": .02}
        identity, gap = relative_error(inp, inp)
        self.assertEqual(identity, {"low": 1, "weather": 1, "combined": 1})
        self.assertEqual(gap, 0)
        perfect, gap = relative_error(inp, {name: 0 for name in inp})
        self.assertEqual(perfect, {"low": 0, "weather": 0, "combined": 0})
        self.assertEqual(gap, 0)
        _, gap = relative_error(inp, {"low": .003, "weather": .004,
                                      "combined": .006})
        self.assertAlmostEqual(gap, .2)

    def test_cpu_identity_inference_protocol_preserves_odd_shapes(self):
        image = self.clean[:, :61, :59]
        seen = {}

        def identity(x, pil):
            seen['tensor_size'] = tuple(x.shape[-2:])
            seen['pil_size'] = pil.size
            return x

        source, output = _forward(identity, image, torch.device('cpu'))
        self.assertEqual(seen['tensor_size'], (64, 64))
        self.assertEqual(seen['pil_size'], (59, 61))
        self.assertEqual(source.shape, (61, 59, 3))
        self.assertAlmostEqual(mean_error(source, output), 0)


if __name__ == "__main__":
    unittest.main()
