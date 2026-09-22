"""Bounded CDD-11 smoke audit. Never trains or evaluates sealed partitions."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
import types
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

REPOS = {
    'onerestore': ('gy65896/OneRestore', '23a2e6af066864dca013aeed7f6c6f5958aa7259'),
    'mirage': ('Amazingren/MIRAGE', '926dc83646272a0d99b36f4bff496985ab064795'),
}
HASHES = {
    'test.zip': '421567fceda89659a6d047fc78d82f77c68703203336d9c795213ba7096646dc',
    'onerestore_cdd-11.tar': 'e02e2d87ce56740a9bedb5cffe8b129d3b206590ae3b97050c9b53aa549c7c9f',
    'embedder_model.tar': '01c1ed1fe4fd06a73c78a7d1fac1b4092a098f3ea08bc71bcab1e4fdad7f6ec2',
}
GROUPS = ['low', 'haze', 'rain', 'snow', 'low_haze', 'low_rain', 'low_snow',
          'haze_rain', 'haze_snow', 'low_haze_rain', 'low_haze_snow']
SMOKE_GROUPS = ['rain', 'low_haze', 'low_haze_snow']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def safe_extract(archive, target):
    target = Path(target).resolve()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            dest = (target / item.filename).resolve()
            if not dest.is_relative_to(target) or '\\' in item.filename:
                raise ValueError(f'Unsafe ZIP member: {item.filename}')
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('ZIP symlink rejected')
        z.extractall(target)


def fingerprint(path):
    with Image.open(path) as im:
        rgb = im.convert('RGB')
        pixels = np.asarray(rgb)
        digest = hashlib.sha256(str(rgb.size).encode() + pixels.tobytes()).hexdigest()
        small = np.asarray(rgb.convert('L').resize((9, 8), Image.Resampling.BILINEAR))
        bits = (small[:, 1:] > small[:, :-1]).flatten()
        dhash = sum(int(v) << i for i, v in enumerate(bits))
        return digest, dhash, list(rgb.size)


def make_manifest(root, old_ids):
    root = Path(root)
    clean = sorted((root / 'clear').glob('*.png'))
    if len(clean) < 20:
        raise ValueError('Expected full CDD-11 test, not a tiny fixture/subset')
    rows, seen, near = [], [], []
    for path in clean:
        digest, dhash, size = fingerprint(path)
        duplicate = next((sid for sid, sh, dh in seen
                          if sh == digest or (dh ^ dhash).bit_count() <= 4), None)
        # Conservatively discard later similar scenes rather than split duplicates.
        if duplicate is not None:
            near.append([path.stem, duplicate])
            continue
        seen.append((path.stem, digest, dhash))
        if path.stem in old_ids:
            continue
        for group in GROUPS:
            pair = root / group / path.name
            if not pair.is_file():
                raise FileNotFoundError(pair)
            with Image.open(pair) as im:
                if list(im.size) != size:
                    raise ValueError(f'Pair size mismatch: {pair}')
        rows.append({'scene': path.stem, 'file': path.name, 'pixel_sha256': digest, 'size': size})
    # Remove earlier retained images that resemble OLD scenes encountered later.
    old_fps = [fingerprint(p)[:2] for p in clean if p.stem in old_ids]
    fps = {sid: (sh, dh) for sid, sh, dh in seen}
    rows = [r for r in rows if not any(fps[r['scene']][0] == sh or
             (fps[r['scene']][1] ^ dh).bit_count() <= 4 for sh, dh in old_fps)]
    rows.sort(key=lambda r: hashlib.sha256(('audit-20260922:' + r['scene']).encode()).hexdigest())
    if len(rows) < 15:
        raise ValueError('Too few independent scenes after exclusions')
    n = len(rows)
    for i, row in enumerate(rows):
        row['partition'] = 'discovery' if i < int(.4*n) else ('confirmation' if i < int(.6*n) else 'holdout')
    return {'schema': 1, 'root': str(root.resolve()), 'groups': GROUPS, 'scenes': rows,
            'excluded_old_ids': sorted(old_ids), 'near_duplicate_candidates_discarded': near,
            'limitations': 'dHash is a conservative screen, not semantic dedup proof; old images absent from full test cannot be pixel-compared.'}


def prepare(work):
    from huggingface_hub import hf_hub_download
    import gdown
    work.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, (repo, revision) in REPOS.items():
        dest = work / name
        if not dest.exists():
            subprocess.run(['git', 'clone', f'https://github.com/{repo}.git', str(dest)], check=True)
            subprocess.run(['git', '-C', str(dest), 'checkout', '--detach', revision], check=True)
        actual = subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(dest), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
        if actual != revision or dirty:
            raise RuntimeError(f'Unexpected revision/edits in {dest}; use a fresh work directory')
        sources[name] = {'repo': repo, 'revision': actual}
    paths = {}
    for name, expected in HASHES.items():
        dataset = name == 'test.zip'
        path = hf_hub_download(repo_id='gy65896/CDD-11' if dataset else 'gy65896/OneRestore',
                               repo_type='dataset' if dataset else 'model', filename=name,
                               local_dir=str(work / 'downloads'))
        if sha256(path) != expected:
            raise RuntimeError(f'Hash mismatch: {name}')
        paths[name] = path
    mirage_path = work / 'downloads' / 'CDD11_small.ckpt'
    if not mirage_path.exists():
        partial = str(mirage_path) + '.part'
        if not gdown.download(id='1GLRMUDfjgWR7aW4DqnUDJzsPO98-cw5G', output=partial, quiet=False):
            raise RuntimeError('MIRAGE Drive download failed; no random-weight fallback')
        Path(partial).rename(mirage_path)
    if mirage_path.stat().st_size < 1_000_000:
        raise RuntimeError('MIRAGE download is too small; possibly an HTML error')
    paths['CDD11_small.ckpt'] = str(mirage_path)
    data = work / 'data'
    marker = data / '.extracted'
    if not marker.exists():
        safe_extract(paths['test.zip'], data)
        marker.write_text(HASHES['test.zip'], encoding='utf-8')
    roots = [p.parent for p in data.rglob('clear') if p.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f'Ambiguous test roots: {roots}')
    project = Path(__file__).resolve().parents[1]
    old = json.loads((project / 'results/2026-09-15_baseline_sidd32_fullframe_seed42/dataset_manifest.json').read_text())
    old_ids = set(old['train_scene_ids'] + old['validation_scene_ids'] + old['test_scene_ids'])
    manifest = make_manifest(roots[0], old_ids)
    manifest_path = work / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError('Locked manifest changed; refusing to overwrite')
    save_json(manifest_path, manifest)
    save_json(work / 'sources.json', {'repos': sources, 'weights': {
        k: {'path': v, 'sha256': sha256(v)} for k, v in paths.items() if k != 'test.zip'}})
    print('Manifest locked:', {p: sum(r['partition'] == p for r in manifest['scenes'])
                               for p in ['discovery', 'confirmation', 'holdout']}, flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(name, work, device):
    import torch
    from torchvision import transforms, models
    sources = json.loads((work / 'sources.json').read_text())
    def weights(filename):
        record = sources['weights'][filename]
        if sha256(record['path']) != record['sha256']:
            raise RuntimeError('Checkpoint changed since preparation')
        return torch.load(record['path'], map_location='cpu', weights_only=True)
    if name == 'mirage':
        model = load_module('audit_mirage', work / name / 'net/mirage_small.py').MIRAGE()
        checkpoint = weights('CDD11_small.ckpt')
        state = checkpoint['state_dict']
        if not state or not all(k.startswith('net.') for k in state):
            raise RuntimeError('Unexpected MIRAGE checkpoint keys')
        model.load_state_dict({k[4:]: v for k, v in state.items()}, strict=True)
        model.to(device).eval()
        def forward(x, pil):
            result = model(x)
            return result[0] if isinstance(result, (tuple, list)) else result
        return forward
    restorer = load_module('audit_onerestore', work / name / 'model/OneRestore.py').OneRestore()
    restorer.load_state_dict(weights('onerestore_cdd-11.tar'), strict=True)
    state = weights('embedder_model.tar')
    # All initialization tensors are subsequently overwritten by strict loading.
    # Avoid downloading GloVe/ResNet weights unrelated to the supplied checkpoint.
    shim = types.ModuleType('utils.utils_word_embedding')
    shim.initialize_wordembedding_matrix = lambda method, vocab: (state['embedder.weight'].clone(), state['embedder.weight'].shape[1])
    original = models.resnet.resnet18
    with patch.dict(sys.modules, {'utils.utils_word_embedding': shim}), patch.object(
            models.resnet, 'resnet18', lambda **kwargs: original(weights=None)):
        module = load_module('audit_embedder', work / name / 'model/Embedder.py')
        embedder = module.Embedder(['clear'] + GROUPS)
    embedder.load_state_dict(state, strict=True)
    restorer.to(device).eval()
    embedder.to(device).eval()
    embedder.train_type = embedder.train_type.to(device)
    transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    def forward(x, pil):
        embedding, _, _ = embedder(transform(pil).unsqueeze(0).to(device), 'image_encoder')
        return restorer(x, embedding)
    return forward


def smoke(name, work, scenes):
    import torch
    import torch.nn.functional as F
    from skimage.metrics import structural_similarity
    if not torch.cuda.is_available():
        raise RuntimeError('Real-model smoke requires Kaggle GPU; CPU tests are separate')
    if not 1 <= scenes <= 5:
        raise ValueError('Smoke is limited to 1..5 discovery scenes')
    device = torch.device('cuda:0')
    torch.manual_seed(20260923)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((work / 'manifest.json').read_text())
    root = Path(manifest['root'])
    selected = [r for r in manifest['scenes'] if r['partition'] == 'discovery'][:scenes]
    output = work / 'results' / name
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    save_json(output / 'status.json', {'status': 'running', 'model': name})
    forward = load_model(name, work, device)
    with torch.inference_mode():
        for row in selected:
            with Image.open(root / 'clear' / row['file']) as image:
                gt = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
            for group in SMOKE_GROUPS:
                with Image.open(root / group / row['file']) as image:
                    pil = image.convert('RGB')
                arr = np.asarray(pil, dtype=np.float32) / 255
                x = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0).to(device)
                h, w = x.shape[-2:]
                # Common protocol: right/bottom replicate-pad to 8; unpad for scoring.
                padded = F.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode='replicate')
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                pred = forward(padded, pil)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                if pred.shape != padded.shape or not torch.isfinite(pred).all():
                    raise RuntimeError('Invalid output tensor')
                raw = pred[0, :, :h, :w].permute(1, 2, 0).cpu().numpy()
                restored = raw.clip(0, 1)
                mse = float(np.mean((restored - gt)**2, dtype=np.float64))
                rows.append({'scene': row['scene'], 'group': group, 'psnr': -10*math.log10(max(mse, 1e-15)),
                             'ssim': float(structural_similarity(gt, restored, data_range=1., channel_axis=-1)),
                             'seconds': seconds, 'peak_vram_bytes': torch.cuda.max_memory_allocated(),
                             'output_min': float(raw.min()), 'output_max': float(raw.max()), 'size': [w, h]})
                panel = np.concatenate([arr, restored, gt], axis=1)
                Image.fromarray((panel.clip(0, 1)*255).round().astype('uint8')).save(output / f"{row['scene']}_{group}.png")
                save_json(output / 'metrics.json', rows)
                print(name, rows[-1], flush=True)
    save_json(output / 'status.json', {'status': 'complete', 'model': name, 'images': len(rows),
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'manifest_sha256': sha256(work / 'manifest.json'),
              'protocol': 'smoke_only_fp32_rgb_clamp01_ssim_skimage_default_pad8; timings include cold start, not benchmark'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'smoke'])
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--model', choices=list(REPOS), default='onerestore')
    parser.add_argument('--scenes', type=int, default=3)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.work.resolve())
    else:
        smoke(args.model, args.work.resolve(), args.scenes)


if __name__ == '__main__':
    main()
