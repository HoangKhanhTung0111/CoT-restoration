from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from .datasets.cdd11 import CDD11_TYPES
from .datasets.s2b_coverage import (
    A_PROBE_TYPES,
    GENERATED_TYPES,
    S2BCoverageDataset,
    build_cv_manifest,
    load_cv_manifest,
    write_new_manifest,
)
from .prepare_s2b_coverage import materialize, validate_cache


class S2BCoverageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "cdd-11-30" / "CDD-11_train"
        for name in ("clear", *CDD11_TYPES):
            (self.data / name).mkdir(parents=True)
        rng = np.random.default_rng(7)
        for index in range(10):
            filename = f"{index:06d}.png"
            clean = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
            Image.fromarray(clean).save(self.data / "clear" / filename)
            for type_index, name in enumerate(CDD11_TYPES):
                degraded = np.clip(clean.astype(np.int16) - type_index - 1, 0, 255).astype(np.uint8)
                Image.fromarray(degraded).save(self.data / name / filename)
        self.manifest_path = self.root / "s2b_manifest.json"
        manifest = build_cv_manifest(
            self.root / "cdd-11-30",
            development_count=10,
            fold_count=5,
            seed=42,
        )
        write_new_manifest(self.manifest_path, manifest)
        self.cache = self.root / "cache"
        materialize(self.root / "cdd-11-30", self.manifest_path, self.cache)

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_disjoint_split_and_stable_manifest(self):
        manifest = load_cv_manifest(self.manifest_path)
        train = set(manifest["folds"][0]["train"])
        val = set(manifest["folds"][0]["validation"])
        self.assertEqual(len(train), 8)
        self.assertEqual(len(val), 2)
        self.assertFalse(train & val)
        validation_ids = [scene for fold in manifest["folds"] for scene in fold["validation"]]
        self.assertEqual(len(validation_ids), len(set(validation_ids)))
        self.assertEqual(set(validation_ids), {row["scene"] for row in manifest["scenes"]})
        self.assertFalse(manifest["source"]["internet_download_allowed"])
        write_new_manifest(self.manifest_path, manifest)
        changed = json.loads(json.dumps(manifest))
        changed["selection"]["seed"] = 9
        with self.assertRaises(RuntimeError):
            write_new_manifest(self.manifest_path, changed)

    def test_r0_r1_sizes_and_no_test_dependency(self):
        r0 = S2BCoverageDataset(
            self.root / "cdd-11-30", self.manifest_path, self.cache,
            fold=0, split="train", arm="r0", crop_size=16,
        )
        r1 = S2BCoverageDataset(
            self.root / "cdd-11-30", self.manifest_path, self.cache,
            fold=0, split="train", arm="r1", crop_size=16,
        )
        self.assertEqual(len(r0), 8 * len(CDD11_TYPES))
        self.assertEqual(len(r1), 8 * (len(CDD11_TYPES) + len(GENERATED_TYPES)))
        sample = r1[-1]
        self.assertEqual(tuple(sample["lq"].shape), (3, 16, 16))
        self.assertEqual(tuple(sample["gt"].shape), (3, 16, 16))
        self.assertEqual(tuple(sample["label"].shape), (4,))
        self.assertFalse((self.root / "cdd-11-30" / "CDD-11_test").exists())

    def test_generated_probe_and_cache_validation(self):
        generated = S2BCoverageDataset(
            self.root / "cdd-11-30", self.manifest_path, self.cache,
            fold=0, split="validation", arm="r1", crop_size=0, augment=False,
            degradation_types=GENERATED_TYPES,
        )
        self.assertEqual(len(generated), 2 * len(GENERATED_TYPES))
        cache_manifest = validate_cache(self.manifest_path, self.cache)
        expected = 10 * (len(GENERATED_TYPES) + len(A_PROBE_TYPES))
        self.assertEqual(cache_manifest["record_count"], expected)

    def test_wrong_development_count_is_rejected(self):
        with self.assertRaises(RuntimeError):
            build_cv_manifest(self.root / "cdd-11-30", development_count=25, fold_count=5)


if __name__ == "__main__":
    unittest.main()
