"""Exact on-disk cache of canonical 256x256 views (roll_canonical_object.load_canonical_view).

Canonicalization is deterministic per (view, object basis), so it is computed once by
build_view_cache and stored losslessly, one zip per object:

  <root>/manifest.json                     identity: catalog, preprocessing policy, source hashes
  <root>/objects/<class>/<object>.zip
      meta.json            basis + per-view status and transform info
      <image_id>.webp      RGB, WebP lossless (decodes to the identical uint8 array)
      <image_id>.fg.npy    foreground (float32), deflate
      <image_id>.valid.npy valid region (float32), deflate

load_view() returns exactly what load_canonical_view() returns, and re-raises the same
deterministic data errors (MissingCanonicalForeground / ValueError) with the same message.
"""
import hashlib
import io
import json
import os
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from .canonical_object import MissingCanonicalForeground
from .roll_canonical_object import PREPROCESS_POLICY

# Bump FORMAT whenever the stored layout or encoding changes (this file is not hashed,
# so reader-only edits do not invalidate an existing cache).
FORMAT = 'mrdt_view_cache_v1'
WEBP_OPTIONS = {'format': 'WEBP', 'lossless': True, 'method': 2}
# The code that computes the cached pixels; changing it makes a cache stale.
SOURCE_FILES = ['roll_canonical_object.py', 'canonical_object.py', 'colmap_io.py']

_ROOT = None
_META = OrderedDict()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def expected_identity(catalog_dir):
    here = Path(__file__).parent
    return {'format': FORMAT, 'preprocessing_policy': PREPROCESS_POLICY,
            'catalog_readiness_sha256': sha256_file(Path(catalog_dir) / 'readiness.json'),
            'sources': {name: sha256_file(here / name) for name in SOURCE_FILES},
            'webp': WEBP_OPTIONS}


def object_path(root, uid):
    return Path(root) / 'objects' / (uid + '.zip')


def basis_record(basis):
    return {k: np.asarray(basis[k], dtype=np.float64).tolist() for k in ['origin', 'x', 'y', 'up']}


def encode_object(uid, basis, views, results):
    """results[image_id] = (color, fg, valid, info) or an exception from load_canonical_view."""
    meta = {'format': FORMAT, 'uid': uid, 'basis': basis_record(basis), 'views': {}}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as z:
        for view in views:
            image_id = view['image_id']
            result = results[image_id]
            record = {'path': view['path']}
            if isinstance(result, Exception):
                record.update(status='missing_foreground' if isinstance(result, MissingCanonicalForeground) else 'error',
                              error_type=type(result).__name__, message=str(result))
            else:
                color, fg, valid, info = result
                image = io.BytesIO()
                Image.fromarray(color).save(image, **WEBP_OPTIONS)
                z.writestr(f'{image_id}.webp', image.getvalue(), compress_type=zipfile.ZIP_STORED)
                for name, array in [('fg', fg), ('valid', valid)]:
                    data = io.BytesIO()
                    np.save(data, array, allow_pickle=False)
                    z.writestr(f'{image_id}.{name}.npy', data.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
                record.update(status='ok', info=info)
            meta['views'][str(image_id)] = record
        z.writestr('meta.json', json.dumps(meta), compress_type=zipfile.ZIP_DEFLATED)
    return buffer.getvalue()


def enable(root, catalog_dir, *, allow_incomplete=False):
    """Route roll_direction_dataset.tensor_view through the cache (inherited by forked workers)."""
    global _ROOT
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    expected = expected_identity(catalog_dir)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f'View cache {root} does not match this catalog/code: {key}')
    if not manifest.get('complete') and not allow_incomplete:
        raise ValueError(f'View cache {root} is incomplete; finish build_view_cache first')
    _ROOT = root
    _META.clear()
    return {'root': str(root), 'manifest_sha256': sha256_file(root / 'manifest.json'),
            'objects': manifest.get('objects'), 'complete': bool(manifest.get('complete'))}


def active():
    return _ROOT is not None


def _meta(uid, archive):
    key = (os.getpid(), uid)
    if key not in _META:
        _META[key] = json.loads(archive.read('meta.json'))
        while len(_META) > 16:
            _META.popitem(last=False)
    _META.move_to_end(key)
    return _META[key]


def object_basis(row):
    """The object's reference basis as stored in the cache (identical to
    roll_canonical_object.reference_basis(row), which may need the original COLMAP
    images.bin when the reference view was removed by the mask audit)."""
    uid = row['instance_uid']
    with zipfile.ZipFile(object_path(_ROOT, uid)) as archive:
        meta = _meta(uid, archive)
    if meta['uid'] != uid:
        raise ValueError(f'View cache uid mismatch for {uid}')
    return {key: np.asarray(meta['basis'][key], dtype=np.float64) for key in ['origin', 'x', 'y', 'up']}


def basis_for(row):
    """Reference basis for row: from the cache when enabled, else computed from the catalog."""
    if active():
        return object_basis(row)
    from .roll_canonical_object import reference_basis
    return reference_basis(row)


def load_view(view, basis):
    """Drop-in replacement for load_canonical_view(view, basis)."""
    path = Path(view['path'])
    uid = '/'.join(path.parts[-4:-2])
    with zipfile.ZipFile(object_path(_ROOT, uid)) as archive:
        meta = _meta(uid, archive)
        if meta['uid'] != uid or meta['basis'] != basis_record(basis):
            raise ValueError(f'View cache basis/uid mismatch for {uid}')
        record = meta['views'].get(str(view['image_id']))
        if record is None or record['path'] != view['path']:
            raise ValueError(f'View {view["image_id"]} of {uid} is not in the view cache')
        if record['status'] == 'missing_foreground':
            raise MissingCanonicalForeground(record['message'])
        if record['status'] == 'error':
            raise ValueError(record['message'])
        image_id = view['image_id']
        with Image.open(io.BytesIO(archive.read(f'{image_id}.webp'))) as image:
            color = np.asarray(image.convert('RGB'))
        fg = np.load(io.BytesIO(archive.read(f'{image_id}.fg.npy')), allow_pickle=False)
        valid = np.load(io.BytesIO(archive.read(f'{image_id}.valid.npy')), allow_pickle=False)
    return color, fg, valid, record['info']
