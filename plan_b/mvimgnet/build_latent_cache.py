"""Build the SD-VAE latent cache (latent_cache.py) from a complete view cache.

One process per GPU encodes its share of the objects (object index % --num-shards), in
fp32 with TF32 disabled. Resumable: finished objects are skipped. Run --finalize once
every shard has finished to mark the manifest complete.

    python -m plan_b.mvimgnet.build_latent_cache --view-cache /var/tmp/<user>/mrdt_cache \
        --catalog <catalog dir> --vae-id <sd-vae-ft-mse> --out /var/tmp/<user>/mrdt_latents --shard 0 --num-shards 4
    python -m plan_b.mvimgnet.build_latent_cache ... --finalize
"""
import argparse
import io
import json
import os
import shutil
import sqlite3
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import latent_cache, view_cache


def catalog_uids(catalog):
    connection = sqlite3.connect(Path(catalog / 'catalog.sqlite').resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    uids = [json.loads(row)['instance_uid'] for (row,) in connection.execute('SELECT row_json FROM instances ORDER BY rowid')]
    connection.close()
    return uids


class ObjectViews(Dataset):
    """uint8 RGB (n,3,256,256) of an object's 'ok' views, decoded exactly as view_cache.load_view."""

    def __init__(self, view_root, uids):
        self.view_root, self.uids = view_root, uids

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, index):
        uid = self.uids[index]
        with zipfile.ZipFile(view_cache.object_path(self.view_root, uid)) as archive:
            meta = json.loads(archive.read('meta.json'))
            assert meta['uid'] == uid
            ids = sorted(int(i) for i, record in meta['views'].items() if record['status'] == 'ok')
            images = []
            for image_id in ids:
                with Image.open(io.BytesIO(archive.read(f'{image_id}.webp'))) as image:
                    images.append(np.asarray(image.convert('RGB')))
        rgb = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2) if ids else torch.zeros(0, 3, 256, 256, dtype=torch.uint8)
        return uid, torch.tensor(ids, dtype=torch.int64), rgb


def write_object(root, uid, ids, z):
    target = latent_cache.object_path(root, uid)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f'.tmp{os.getpid()}')
    with temporary.open('wb') as f:
        np.savez(f, image_ids=ids.numpy(), z=z.numpy())
    os.replace(temporary, target)


def numerics(device):
    import diffusers
    return {'dtype': 'float32', 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32, 'gpu': torch.cuda.get_device_name(device),
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'diffusers': diffusers.__version__}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--view-cache', type=Path, required=True)
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--vae-id', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--batch', type=int, default=64, help='Views per VAE encode call')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--min-free-gb', type=float, default=15)
    p.add_argument('--limit', type=int, help='Only the first N objects (testing)')
    p.add_argument('--finalize', action='store_true')
    args = p.parse_args()
    log = lambda m: print(time.strftime('%Y-%m-%d %H:%M:%S'), f'[shard {args.shard}]', m, flush=True)

    view_manifest = json.loads((args.view_cache / 'manifest.json').read_text())
    assert view_manifest.get('complete'), 'view cache must be complete'
    for key, value in view_cache.expected_identity(args.catalog).items():
        assert view_manifest.get(key) == value, f'view cache does not match the catalog: {key}'
    identity = latent_cache.expected_identity(args.view_cache, args.vae_id)
    uids = catalog_uids(args.catalog)
    if args.limit:
        uids = uids[:args.limit]
    manifest_path = args.out / 'manifest.json'

    if args.finalize:
        manifest = json.loads(manifest_path.read_text())
        for key, value in identity.items():
            assert manifest.get(key) == value, f'cache was built for different {key}'
        missing = [u for u in uids if not latent_cache.object_path(args.out, u).exists()]
        views = 0
        for uid in uids:
            if not missing:
                with np.load(latent_cache.object_path(args.out, uid)) as data:
                    views += len(data['image_ids'])
        complete = args.limit is None and not missing
        manifest.update(complete=complete, objects=len(uids), views=views if not missing else None, finished_at=time.time())
        manifest_path.write_text(json.dumps(manifest, indent=2))
        log(f'finalize: {len(missing)} objects missing, views {views}, complete={complete}')
        return

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda', torch.cuda.current_device())
    run_numerics = numerics(device)
    args.out.mkdir(parents=True, exist_ok=True)
    os.chmod(args.out, 0o700)
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key, value in identity.items():
            assert old.get(key) == value, f'Existing cache was built for different {key}; use a new --out'
        assert old['numerics'] == run_numerics, 'numerics differ from the existing cache; use a new --out'
    elif args.shard == 0:
        manifest_path.write_text(json.dumps({**identity, 'numerics': run_numerics, 'complete': False,
                                             'started_at': time.time()}, indent=2))
    else:
        while not manifest_path.exists():
            time.sleep(1)
        assert json.loads(manifest_path.read_text())['numerics'] == run_numerics

    from rescene.codec.vae_latent_delta import load_vae
    vae = load_vae(str(args.vae_id), device=device)
    todo = [u for i, u in enumerate(uids) if i % args.num_shards == args.shard
            and not latent_cache.object_path(args.out, u).exists()]
    log(f'{len(todo)} objects to encode -> {args.out}')
    loader = DataLoader(ObjectViews(args.view_cache, todo), batch_size=None, num_workers=args.workers,
                        prefetch_factor=4, persistent_workers=False)
    pending, pending_views = [], 0
    done = views = 0
    started = last = time.time()

    def flush():
        nonlocal pending, pending_views, done, views
        if pending_views:
            rgb = torch.cat([r for _, _, r in pending]).to(device, non_blocking=True)
            # Same arithmetic as tensor_view (.float()/255) followed by DiffusersVAE.encode.
            z = torch.cat([vae.encode(rgb[i:i + args.batch].float() / 255) for i in range(0, rgb.shape[0], args.batch)]).float().cpu()
        offset = 0
        for uid, ids, rgb_obj in pending:
            n = rgb_obj.shape[0]
            write_object(args.out, uid, ids, z[offset:offset + n] if n else torch.zeros(0, *latent_cache.LATENT_SHAPE))
            offset += n
        done += len(pending); views += pending_views
        pending, pending_views = [], 0

    with torch.no_grad():
        for uid, ids, rgb in loader:
            pending.append((uid, ids, rgb)); pending_views += rgb.shape[0]
            if pending_views >= args.batch:
                flush()
            if time.time() - last > 60:
                free = shutil.disk_usage(args.out).free / 1e9
                rate = done / (time.time() - started)
                log(f'{done}/{len(todo)} objects, {views} views, {rate:.1f} obj/s, '
                    f'ETA {(len(todo) - done) / max(rate, 1e-9) / 3600:.2f} h, free {free:.0f} GB')
                last = time.time()
                if free < args.min_free_gb:
                    raise SystemExit(f'Free space {free:.0f} GB < --min-free-gb {args.min_free_gb}; stopped (resumable)')
        flush()
    log(f'finished: {done} objects, {views} views')


if __name__ == '__main__':
    main()
