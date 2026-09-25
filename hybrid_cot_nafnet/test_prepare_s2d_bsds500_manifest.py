from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .prepare_s2d_bsds500_manifest import build_manifest


class BSDS500ManifestTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, dict[str, int]]:
        counts = {"train": 2, "val": 1, "test": 2}
        image_root = root / "images"
        for split, count in counts.items():
            split_root = image_root / split
            split_root.mkdir(parents=True)
            for index in range(count):
                (split_root / f"{100 * (index + 1) + len(split)}.jpg").write_bytes(
                    f"jpeg-fixture:{split}:{index}".encode()
                )
            (split_root / "Thumbs.db").write_bytes(f"thumb:{split}".encode())
        archive = root / "BSR_bsds500.tgz"
        archive.write_bytes(b"archive-fixture")
        return image_root, archive, counts

    def test_manifest_is_deterministic_and_does_not_decode_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            image_root, archive, counts = self._fixture(Path(directory))
            first = build_manifest(image_root, archive, counts)
            second = build_manifest(image_root, archive, counts)
            self.assertEqual(first, second)
            self.assertEqual(first["scene_count"], 5)
            self.assertEqual(first["split_counts"], counts)
            self.assertFalse(first["image_pixels_decoded"])
            self.assertFalse(first["final_split_pixels_decoded"])
            self.assertEqual(len(first["ancillary_records"]), 3)

    def test_unknown_ancillary_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            image_root, archive, counts = self._fixture(Path(directory))
            (image_root / "val" / "notes.txt").write_text("unexpected")
            with self.assertRaises(RuntimeError):
                build_manifest(image_root, archive, counts)


if __name__ == "__main__":
    unittest.main()
