"""Validate the official BSDS500 image splits and freeze a byte-level manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROTOCOL_VERSION = "s2d-full-frozen-audit-v1.0"
SOURCE_URL = (
    "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/"
    "grouping/BSR/BSR_bsds500.tgz"
)
EXPECTED_SPLITS = {"train": 200, "val": 100, "test": 200}
ALLOWED_ANCILLARY_FILES = {"Thumbs.db"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    image_root: Path,
    archive: Path,
    expected_splits: dict[str, int] | None = None,
) -> dict:
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing BSDS500 image root: {image_root}")
    if not archive.is_file():
        raise FileNotFoundError(f"Missing BSDS500 archive: {archive}")

    expected_splits = EXPECTED_SPLITS if expected_splits is None else expected_splits
    records = []
    ancillary_records = []
    seen_names: set[str] = set()
    split_counts = {}
    for split, expected_count in expected_splits.items():
        split_root = image_root / split
        files = sorted(split_root.glob("*.jpg"), key=lambda path: int(path.stem))
        ancillary = sorted(
            path for path in split_root.iterdir() if path.is_file() and path.suffix.lower() != ".jpg"
        )
        unexpected = [path.name for path in ancillary if path.name not in ALLOWED_ANCILLARY_FILES]
        if unexpected:
            raise RuntimeError(f"Unexpected files in {split}: {unexpected[:5]}")
        ancillary_records.extend(
            {
                "split": split,
                "file": f"{split}/{path.name}",
                "bytes": path.stat().st_size,
                "file_sha256": sha256_file(path),
                "role": "official_archive_packaging_debris_ignored",
            }
            for path in ancillary
        )
        if len(files) != expected_count:
            raise RuntimeError(
                f"BSDS500 {split} count changed: expected {expected_count}, found {len(files)}"
            )
        for path in files:
            if path.name in seen_names:
                raise RuntimeError(f"Duplicate BSDS500 image name across splits: {path.name}")
            seen_names.add(path.name)
            records.append(
                {
                    "scene": path.stem,
                    "split": split,
                    "file": f"{split}/{path.name}",
                    "bytes": path.stat().st_size,
                    "file_sha256": sha256_file(path),
                }
            )
        split_counts[split] = len(files)

    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "BSDS500",
        "source_url": SOURCE_URL,
        "archive_file": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "image_root_suffix": "BSR/BSDS500/data/images",
        "split_counts": split_counts,
        "scene_count": len(records),
        "official_split_preserved": True,
        "image_pixels_decoded": False,
        "final_split_pixels_decoded": False,
        "ancillary_records": ancillary_records,
        "records": records,
    }


def write_locked_manifest(path: Path, manifest: dict) -> None:
    payload = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"Refusing to overwrite changed manifest: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root).resolve()
    archive = Path(args.archive).resolve()
    output = Path(args.output).resolve()
    manifest = build_manifest(image_root, archive)
    write_locked_manifest(output, manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest": str(output),
                "manifest_sha256": sha256_file(output),
                "archive_sha256": manifest["archive_sha256"],
                "archive_bytes": manifest["archive_bytes"],
                "split_counts": manifest["split_counts"],
                "scene_count": manifest["scene_count"],
                "image_pixels_decoded": manifest["image_pixels_decoded"],
                "final_split_pixels_decoded": manifest["final_split_pixels_decoded"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
