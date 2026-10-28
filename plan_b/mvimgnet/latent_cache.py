"""Precomputed SD-VAE latents of every cached canonical view (build_latent_cache).

The VAE is frozen and its encode is deterministic (posterior mode), so each view's latent
is computed once in fp32 (TF32 off) instead of on every training step:

  <root>/manifest.json               identity: view cache, VAE weights, encode code, numerics
  <root>/objects/<class>/<object>.npz
      image_ids  int64 (n,)           views with status 'ok' in the view cache
      z          float32 (n,4,32,32)  DiffusersVAE.encode(view/255) = posterior.mode() * scaling_factor

lookup() returns the raw latents z; the caller turns them into patch tokens exactly as
encode_target_latents/encode_context_latents do.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import view_cache

FORMAT = 'mrdt_latent_cache_v1'
ENCODE_SOURCE = Path(__file__).resolve().parents[2] / 'rescene' / 'codec' / 'vae_latent_delta.py'
LATENT_SHAPE = (4, 32, 32)

_ROOT = None


def vae_identity(vae_id):
    vae_id = Path(vae_id)
    weights = vae_id / 'diffusion_pytorch_model.safetensors'  # what _load_autoencoder_kl loads first
    if not weights.is_file():
        weights = vae_id / 'diffusion_pytorch_model.bin'
    config = json.loads((vae_id / 'config.json').read_text())
    return {'weights_file': weights.name, 'weights_sha256': view_cache.sha256_file(weights),
            'config_sha256': view_cache.sha256_file(vae_id / 'config.json'),
            'scaling_factor': config.get('scaling_factor', 0.18215)}


def expected_identity(view_cache_root, vae_id):
    """Everything that determines the stored values (numerics are recorded separately)."""
    return {'format': FORMAT,
            'view_cache_manifest_sha256': view_cache.sha256_file(Path(view_cache_root) / 'manifest.json'),
            'vae': vae_identity(vae_id),
            'encode_source_sha256': view_cache.sha256_file(ENCODE_SOURCE),
            'latent': 'DiffusersVAE.encode(rgb_uint8/255) = posterior.mode()*scaling_factor, float32 (4,32,32)'}


def object_path(root, uid):
    return Path(root) / 'objects' / (uid + '.npz')


def enable(root, view_cache_root, vae_id, *, allow_incomplete=False):
    """Serve lookup() from root (inherited by forked DataLoader workers)."""
    global _ROOT
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    for key, value in expected_identity(view_cache_root, vae_id).items():
        if manifest.get(key) != value:
            raise ValueError(f'Latent cache {root} does not match this view cache/VAE/code: {key}')
    if not manifest.get('complete') and not allow_incomplete:
        raise ValueError(f'Latent cache {root} is incomplete; finish build_latent_cache first')
    _ROOT = root
    return {'root': str(root), 'manifest_sha256': view_cache.sha256_file(root / 'manifest.json'),
            'objects': manifest.get('objects'), 'views': manifest.get('views'), 'numerics': manifest.get('numerics')}


def active():
    return _ROOT is not None


def lookup(uid, image_ids):
    """float32 tensor (len(image_ids), 4, 32, 32) of the cached latents."""
    with np.load(object_path(_ROOT, uid), allow_pickle=False) as data:
        ids, z = data['image_ids'], data['z']
    row = {int(i): n for n, i in enumerate(ids)}
    missing = [i for i in image_ids if int(i) not in row]
    if missing:
        raise ValueError(f'Views {missing} of {uid} are not in the latent cache')
    return torch.from_numpy(np.ascontiguousarray(z[[row[int(i)] for i in image_ids]]))
