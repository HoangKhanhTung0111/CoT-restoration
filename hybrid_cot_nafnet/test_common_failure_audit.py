"""CPU unit tests; never download models or access the real dataset."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from .common_failure_audit import GROUPS, make_manifest, safe_extract


class AuditTests(unittest.TestCase):
    def test_manifest_determinism_exclusions_and_partitions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for group in ['clear'] + GROUPS:
                (root / group).mkdir()
            rng = np.random.default_rng(4)
            for i in range(30):
                im = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype='uint8'))
                for group in ['clear'] + GROUPS:
                    im.save(root / group / f'{i:05d}.png')
            # Exact duplicate must not enter another partition.
            for group in ['clear'] + GROUPS:
                (root/group/'00029.png').write_bytes((root/group/'00028.png').read_bytes())
            first = make_manifest(root, {'00000', '00001'})
            self.assertEqual(first, make_manifest(root, {'00001', '00000'}))
            scenes = first['scenes']
            self.assertEqual(len(scenes), 27)
            self.assertFalse({'00000', '00001', '00029'} & {r['scene'] for r in scenes})
            self.assertEqual({r['partition'] for r in scenes}, {'discovery', 'confirmation', 'holdout'})
            (root / GROUPS[0] / scenes[0]['file']).unlink()
            with self.assertRaises(FileNotFoundError):
                make_manifest(root, {'00000', '00001'})

    def test_zip_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            z = Path(folder) / 'bad.zip'
            with zipfile.ZipFile(z, 'w') as archive:
                archive.writestr('../escaped.txt', 'bad')
            with self.assertRaises(ValueError):
                safe_extract(z, Path(folder) / 'data')
            self.assertFalse((Path(folder) / 'escaped.txt').exists())

    def test_notebook_compiles_and_imports_subprocess(self):
        path = Path(__file__).resolve().parents[1] / 'notebooks/kaggle_common_failure_smoke.ipynb'
        notebook = json.loads(path.read_text(encoding='utf-8'))
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                compile(source, f'cell-{i}', 'exec')
                self.assertIn('subprocess', source.splitlines()[0])


if __name__ == '__main__':
    unittest.main()
