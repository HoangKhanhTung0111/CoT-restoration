"""CPU tests for the frozen low-weather S1 generator and oracle math."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from . import low_weather_generator_b as gen


class GeneratorBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        y, x = np.mgrid[0:64, 0:64]
        rgb = np.stack((x / 80 + 0.1, y / 80 + 0.1,
                        (x + y) / 160 + 0.1), axis=-1)
        cls.clean = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def test_srgb_roundtrip_and_breakpoints(self):
        points = np.array([0, 0.04045, 0.25, 0.5, 1], dtype=np.float32)
        rgb = np.repeat(points[:, None, None], 3, axis=2)
        recovered = gen.linear_to_srgb(gen.srgb_to_linear(rgb))
        np.testing.assert_allclose(recovered, rgb, rtol=0, atol=2e-7)
        self.assertEqual(float(gen.srgb_to_linear(np.zeros((1, 1, 3))).max()), 0)
        self.assertAlmostEqual(float(gen.linear_to_srgb(np.ones((1, 1, 3))).min()), 1, places=6)

    def test_stable_replay_shape_range_and_no_mutation(self):
        a = gen.build_realization(self.clean, "scene-A")
        b = gen.build_realization(self.clean, "scene-A")
        self.assertEqual(a.seed, b.seed)
        self.assertEqual(a.field_hashes, b.field_hashes)
        before = dict(a.field_hashes)
        low = gen.make_views(self.clean, a, "rain_haze")
        other = gen.make_views(self.clean, a, "snow_haze")
        self.assertEqual(low["clean"].dtype, np.uint8)
        self.assertTrue(np.array_equal(low["low_weak"], other["low_weak"]))
        self.assertTrue(np.array_equal(low["low_strong"], other["low_strong"]))
        for views in (low, other):
            self.assertEqual(set(views), {"clean", "low_weak", "low_strong", "weather",
                                          "combined_weak", "combined_strong"})
            for view in views.values():
                self.assertEqual(view.shape, self.clean.shape)
                self.assertEqual(view.dtype, np.uint8)
                self.assertTrue(np.isfinite(view).all())
        self.assertEqual(before, a.field_hashes)

    def test_identity_overrides(self):
        realization = gen.build_realization(self.clean, "scene-identity")
        linear = gen.srgb_to_linear(self.clean.astype(np.float32) / 255)
        np.testing.assert_array_equal(gen.apply_low(linear, realization, 1.0, k=0, sigma_r=0), linear)
        identity = gen.apply_weather(linear, realization, "rain_haze", t_override=1.0,
                                     alpha_override=0.0)
        np.testing.assert_array_equal(identity, linear)

    def test_noise_free_linear_interaction_equation(self):
        realization = gen.build_realization(self.clean, "scene-equation")
        linear = np.full_like(gen.srgb_to_linear(self.clean.astype(np.float32) / 255), 0.2)
        t = np.full(linear.shape[:2], 0.8, dtype=np.float32)
        alpha = np.full(linear.shape[:2], 0.1, dtype=np.float32)
        weak = gen.apply_low(linear, realization, 0.5, k=0, sigma_r=0)
        combined = gen.apply_weather(weak, realization, "rain_haze", t_override=t,
                                     alpha_override=alpha)
        weather = gen.apply_weather(linear, realization, "rain_haze", t_override=t,
                                    alpha_override=alpha)
        expected = t[..., None] * (1-alpha[..., None]) * (0.5-1.0) * linear
        np.testing.assert_allclose(combined-weather, expected, atol=2e-7, rtol=0)

    def test_weather_field_is_deterministic_and_zero_boundary(self):
        realization = gen.build_realization(self.clean, "scene-kernel")
        again = gen.build_realization(self.clean, "scene-kernel")
        np.testing.assert_array_equal(realization.rain_kernel, again.rain_kernel)
        np.testing.assert_array_equal(realization.snow_kernel, again.snow_kernel)
        self.assertEqual(float(realization.rain_alpha.min()), 0.0)
        self.assertEqual(float(realization.snow_alpha.min()), 0.0)


if __name__ == "__main__":
    unittest.main()
