"""Build the exact canonical-view cache (view_cache.py) for every view of a catalog.

The source images sit on a single HDD, where many concurrent readers thrash the disk. One
reader thread therefore streams each object's RGB/mask files sequentially (warming the page
cache, bounded by --prefetch objects), and a process pool runs the unchanged
load_canonical_view on the cached files and writes one zip per object.

    python -m plan_b.mvimgnet.build_view_cache --catalog <catalog dir> --out /var/tmp/<user>/mrdt_cache

Resumable: finished objects are skipped. Stops when free space under --out drops below
--min-free-gb. manifest.json gets complete=true only after every object was written.
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

from . import view_cache


def build_object(args):
    row_json, root = args
    import cv2
    cv2.setNumThreads(1)
    from .canonical_object import MissingCanonicalForeground  # noqa: F401  (ValueError subclass)
    from .roll_canonical_object import load_canonical_view, reference_basis
    row = json.loads(row_json)
    uid = row['instance_uid']
    target = view_cache.object_path(root, uid)
    if target.exists():
        return uid, 'skipped', 0, 0
    basis = reference_basis(row)
    results = {}
    for view in row['views']:
        try:
            results[view['image_id']] = load_canonical_view(view, basis)
        except ValueError as error:  # deterministic data errors (incl. MissingCanonicalForeground)
            results[view['image_id']] = error
    data = view_cache.encode_object(uid, basis, row['views'], results)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f'.tmp{os.getpid()}')
    temporary.write_bytes(data)
    os.replace(temporary, target)
    failed = sum(isinstance(r, Exception) for r in results.values())
    return uid, 'written', len(results) - failed, failed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--catalog', type=Path, required=True, help='Catalog dir with catalog.sqlite/readiness.json')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=28)
    p.add_argument('--prefetch', type=int, default=256, help='Objects read ahead into the page cache')
    p.add_argument('--min-free-gb', type=float, default=50)
    p.add_argument('--limit', type=int, help='Only the first N objects (testing)')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.chmod(args.out, 0o700)
    identity = view_cache.expected_identity(args.catalog)
    manifest_path = args.out / 'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key, value in identity.items():
            assert old.get(key) == value, f'Existing cache was built for different {key}; use a new --out'
    manifest_path.write_text(json.dumps({**identity, 'complete': False, 'started_at': time.time()}, indent=2))

    log = lambda m: print(time.strftime('%Y-%m-%d %H:%M:%S'), m, flush=True)
    connection = sqlite3.connect(Path(args.catalog / 'catalog.sqlite').resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    rows = [row for (row,) in connection.execute('SELECT row_json FROM instances ORDER BY rowid')]
    connection.close()
    # Disk order: archives were extracted one at a time, objects contiguously.
    rows.sort(key=lambda r: json.loads(r)['instance_path'])
    if args.limit:
        rows = rows[:args.limit]
    total_views = sum(len(json.loads(r)['views']) for r in rows)
    log(f'{len(rows)} objects, {total_views} views -> {args.out}')

    window = threading.Semaphore(args.prefetch)

    def tasks():
        for row_json in rows:
            window.acquire()
            row = json.loads(row_json)
            if not view_cache.object_path(args.out, row['instance_uid']).exists():
                # Sequential read warms the page cache for the workers. All RGB first, then all
                # masks: they are stored in separate runs on disk, so alternating would seek.
                for key in ('path', 'mask_path'):
                    for view in row['views']:
                        with open(view[key], 'rb') as f:
                            while f.read(8 << 20):
                                pass
            yield row_json, args.out

    done = written = ok_views = failed_views = 0
    started = last = time.time()
    with mp.get_context('fork').Pool(args.workers) as pool:
        for uid, state, ok, failed in pool.imap_unordered(build_object, tasks(), chunksize=1):
            window.release()
            done += 1; written += state == 'written'; ok_views += ok; failed_views += failed
            if time.time() - last > 60 or done == len(rows):
                free = shutil.disk_usage(args.out).free / 1e9
                rate = written / max(1e-9, time.time() - started)
                eta = (len(rows) - done) / rate / 3600 if rate else float('nan')
                log(f'{done}/{len(rows)} objects ({written} written), views ok {ok_views}, '
                    f'no-foreground/errors {failed_views}, {rate:.1f} obj/s, ETA {eta:.1f} h, free {free:.0f} GB')
                last = time.time()
                if free < args.min_free_gb:
                    pool.terminate()
                    raise SystemExit(f'Free space {free:.0f} GB < --min-free-gb {args.min_free_gb}; stopped (resumable)')
    complete = args.limit is None and all(view_cache.object_path(args.out, json.loads(r)['instance_uid']).exists() for r in rows)
    manifest_path.write_text(json.dumps({**identity, 'complete': complete, 'objects': len(rows),
                                         'views': total_views, 'finished_at': time.time()}, indent=2))
    log(f'finished: complete={complete}')


if __name__ == '__main__':
    main()
