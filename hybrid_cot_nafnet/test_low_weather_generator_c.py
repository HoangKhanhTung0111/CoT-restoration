"""CPU-only invariants for the pinned mechanism-C renderer."""
from __future__ import annotations

import unittest

import numpy as np

from . import low_weather_generator_c as gen


@unittest.skipUnless(
    gen.A is not None and gen.A.__version__ == gen.ALBUMENTATIONS_VERSION,
    "requires the pinned S2d Albumentations dependency",
)
class GeneratorCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        y, x = np.mgrid[0:96, 0:128]
        checker = ((x // 12 + y // 12) % 2).astype(np.float32)
        rgb = np.stack(
            (0.10 + 0.75 * x / 127, 0.10 + 0.75 * y / 95, 0.15 + 0.65 * checker),
            axis=-1,
        )
        cls.clean = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def test_replay_is_deterministic_and_does_not_mutate_input(self):
        clean = self.clean.copy()
        first = gen.build_realization(clean, "fixture-01")
        first_views = gen.make_views(clean, first)
        second = gen.build_realization(clean, "fixture-01")
        second_views = gen.make_views(clean, second)
        np.testing.assert_array_equal(clean, self.clean)
        self.assertEqual(first.seeds, second.seeds)
        self.assertEqual(first.replay_sha256, second.replay_sha256)
        for condition in gen.CONDITIONS:
            np.testing.assert_array_equal(first_views[condition], second_views[condition])

    def test_grid_shape_dtype_range_and_nonidentity(self):
        realization = gen.build_realization(self.clean, "fixture-02")
        views = gen.make_views(self.clean, realization)
        self.assertEqual(tuple(views), gen.CONDITIONS)
        for condition, image in views.items():
            self.assertEqual(image.shape, self.clean.shape, condition)
            self.assertEqual(image.dtype, np.uint8, condition)
            self.assertTrue(np.isfinite(image).all(), condition)
            self.assertFalse(np.array_equal(image, self.clean), condition)

    def test_factorial_grid_reuses_each_weather_realization(self):
        realization = gen.build_realization(self.clean, "fixture-03")
        before = dict(realization.replay_sha256)
        gen.make_views(self.clean, realization)
        self.assertEqual(before, realization.replay_sha256)
        rain_parameters = realization.rain_haze_replay["transforms"]
        snow_parameters = realization.snow_haze_replay["transforms"]
        self.assertEqual(len(rain_parameters), 2)
        self.assertEqual(len(snow_parameters), 2)
        self.assertTrue(all(item["applied"] for item in rain_parameters))
        self.assertTrue(all(item["applied"] for item in snow_parameters))

    def test_scene_id_changes_weather_draws(self):
        first = gen.build_realization(self.clean, "fixture-A")
        second = gen.build_realization(self.clean, "fixture-B")
        self.assertNotEqual(first.seeds, second.seeds)
        self.assertNotEqual(first.replay_sha256, second.replay_sha256)

    def test_invalid_input_and_shape_mismatch_are_rejected(self):
        with self.assertRaises(ValueError):
            gen.build_realization(self.clean.astype(np.float32), "fixture")
        realization = gen.build_realization(self.clean, "fixture")
        with self.assertRaises(ValueError):
            gen.make_views(self.clean[:-1], realization)


if __name__ == "__main__":
    unittest.main()
