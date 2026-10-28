"""Extract downloaded MVImgNet2.0 archives once, verify them, and write the receipts
that index_verified_full_bin15 / audit_full_masks expect (all_assets_verification/).

Per archive (only after the downloader's quickXorHash marker exists in RAW/.verified/):
  1. Stream gzip+tar once into DATA/extracted/<stem>.partial, hashing every regular file
     as it is written; draining the gzip stream to EOF also checks the gzip CRC.
  2. Rename to DATA/extracted/<stem>, then re-read every file in parallel: SHA256 must match
     the tar bytes, and every RGB/mask is fully decoded with PIL.
  3. Write manifests/<stem>.jsonl.gz, instances/<stem>.json, archives/<stem>.json.
  4. With --delete-archives, delete the archive only when all checks passed.

The aggregate report.json / preparation_config.json are written only by --finalize, so the
archive set stays frozen while a catalog is being built from it.

    python -m plan_b.mvimgnet.extract_verify_archives --raw data/MVImgNet2.0/raw \
        --data data/MVImgNet2.0/prepared --category-json <official repo>/mvimgnet2_category.json \
        --delete-archives --watch
    python -m plan_b.mvimgnet.extract_verify_archives --raw ... --data ... --finalize
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

from PIL import Image

from .prepare_full_bin15 import dump, safe_member

VERSION = 1
SPLIT_POLICY = 'SHA256 mvimgnet2-full-v1 uid,90/5/5; pilot overrides'  # must match prepare_full_bin15.split_for
EXPECTED_ARCHIVES = [f'mvi2_{i:02}' for i in range(41)]


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def kind_of(parts):
    if len(parts) == 4 and parts[2] in ('images', 'masks'):
        return parts[2]
    if len(parts) >= 3 and parts[2] == 'sparse':
        return 'sparse'
    return 'other'


def extract(archive, partial):
    """Single pass over the archive; returns {relative path: (bytes, sha256)}."""
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    files = {}
    with gzip.open(archive, 'rb') as gz, tarfile.open(fileobj=gz, mode='r|') as tar:
        for member in tar:
            path = safe_member(member.name)
            if member.isdir():
                continue
            if not member.isreg():
                raise ValueError(f'Unsupported archive member type {member.type!r}: {member.name}')
            relative = path.as_posix()
            if relative in files:
                raise ValueError('Duplicate archive member: ' + relative)
            target = partial / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            size = 0
            source = tar.extractfile(member)
            with target.open('wb') as sink:
                for block in iter(lambda: source.read(4 << 20), b''):
                    h.update(block)
                    sink.write(block)
                    size += len(block)
            if size != member.size:
                raise ValueError(f'Short read for {relative}: {size}/{member.size}')
            files[relative] = (size, h.hexdigest())
        # tarfile stops at the end-of-archive blocks; read the gzip stream to EOF
        # so GzipFile validates the CRC32/ISIZE trailer.
        while gz.read(16 << 20):
            pass
    return files


def verify_file(args):
    root, relative, size, digest = args
    path = Path(root) / relative
    parts = PurePosixPath(relative).parts
    kind = kind_of(parts)
    stat = path.stat()
    row = {'kind': kind, 'path': relative, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
           'sha256': digest, 'byte_verified': stat.st_size == size and sha256_file(path) == digest,
           'image_decoded': None}
    if kind in ('images', 'masks'):
        try:
            with Image.open(path) as im:
                info = {'size': list(im.size), 'mode': im.mode, 'format': im.format,
                        'exif_orientation': im.getexif().get(274, 1)}
                im.load()  # full decode, not just the header
            row.update(image=info, image_decoded=True)
        except (OSError, ValueError, SyntaxError) as error:
            row.update(image_decoded=False, image_error=f'{type(error).__name__}: {error}')
    return row


def process_archive(stem, raw, data, workers, delete):
    archive = raw / (stem + '.tar.gz')
    marker = json.loads((raw / '.verified' / (stem + '.tar.gz.json')).read_text())
    receipts = data / 'all_assets_verification'
    receipt_path = receipts / 'archives' / (stem + '.json')
    final = data / 'extracted' / stem
    started = time.time()
    if final.exists():
        # A previous run finished extraction but not verification: start clean.
        shutil.rmtree(final)
    if archive.stat().st_size != marker['bytes']:
        raise ValueError(f'{archive} size differs from its download marker')
    log(f'{stem}: extracting')
    files = extract(archive, data / 'extracted' / (stem + '.partial'))
    os.replace(data / 'extracted' / (stem + '.partial'), final)
    log(f'{stem}: extracted {len(files)} files in {time.time() - started:.0f}s; verifying')
    jobs = [(str(final), rel, size, digest) for rel, (size, digest) in sorted(files.items())]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(verify_file, jobs, chunksize=256))
    instances = sorted({'/'.join(PurePosixPath(r['path']).parts[:2]) for r in rows
                        if len(PurePosixPath(r['path']).parts) >= 3})
    count = lambda kind: sum(r['kind'] == kind for r in rows)
    byte_failures = [r['path'] for r in rows if not r['byte_verified']]
    image_errors = [r for r in rows if r['image_decoded'] is False]
    manifest = receipts / 'manifests' / (stem + '.jsonl.gz')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(manifest.with_suffix('.tmp'), 'wt') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    os.replace(manifest.with_suffix('.tmp'), manifest)
    dump(receipts / 'instances' / (stem + '.json'), instances)
    ok = not byte_failures and not image_errors
    receipt = {'version': VERSION, 'state': 'verified' if ok else 'verification_failed', 'archive': stem,
               'archive_bytes': marker['bytes'], 'archive_quickXorHash': marker['quickXorHash'],
               'gzip_crc_verified': True, 'instances': len(instances), 'regular_files': len(rows),
               'byte_verified_files': len(rows) - len(byte_failures), 'images': count('images'),
               'masks': count('masks'), 'image_errors': len(image_errors),
               'byte_failures': byte_failures[:100], 'image_error_examples': image_errors[:100],
               'manifest_sha256': sha256_file(manifest), 'extracted_root': str(final),
               'started_at': started, 'finished_at': time.time()}
    dump(receipt_path, receipt)
    if not ok:
        raise RuntimeError(f'{stem}: {len(byte_failures)} byte failures, {len(image_errors)} image errors; '
                           f'archive kept, see {receipt_path}')
    if delete:
        archive.unlink()
        log(f'{stem}: archive deleted after verification')
    log(f'{stem}: verified {len(instances)} objects, {receipt["images"]} images, {receipt["masks"]} masks '
        f'in {time.time() - started:.0f}s')
    return receipt


def finalize(data, raw, category_json):
    """Freeze the currently verified archives into report.json + preparation_config.json."""
    receipts = data / 'all_assets_verification'
    done = []
    for stem in EXPECTED_ARCHIVES:
        path = receipts / 'archives' / (stem + '.json')
        if path.exists() and json.loads(path.read_text())['state'] == 'verified':
            done.append(json.loads(path.read_text()))
    if not done:
        raise SystemExit('No verified archives yet')
    uids = [u for r in done for u in json.loads((receipts / 'instances' / (r['archive'] + '.json')).read_text())]
    assert len(uids) == len(set(uids)), 'Object UID appears in more than one archive'
    total = lambda key: sum(r[key] for r in done)
    report = {'version': VERSION, 'state': 'verified', 'archive_count': len(done),
              'archives': [r['archive'] for r in done], 'image_errors': total('image_errors'),
              'regular_files': total('regular_files'), 'byte_verified_files': total('byte_verified_files'),
              'images': total('images'), 'masks': total('masks'), 'unique_object_folders': len(uids),
              'missing_archives': [s for s in EXPECTED_ARCHIVES if s not in {r['archive'] for r in done}],
              'finalized_at': time.time()}
    dump(receipts / 'report.json', report)
    config = {'version': VERSION, 'scope': 'all_41_archives' if len(done) == 41 else f'{len(done)}_of_41_archives',
              'instance_limit': None, 'archive_limit': None,
              'archives': [{'path': str(raw / (r['archive'] + '.tar.gz')), 'bytes': r['archive_bytes'],
                            'quickXorHash': r['archive_quickXorHash']} for r in done],
              'preserved_pilot_split': {}, 'split_policy': SPLIT_POLICY,
              'source_sha256': sha256_file(__file__)}
    dump(data / 'preparation_config.json', config)
    if Path(category_json).resolve() != (data / 'mvimgnet2_category.json').resolve():
        shutil.copyfile(category_json, data / 'mvimgnet2_category.json')
    log(f'finalized {len(done)} archives, {len(uids)} objects, {report["masks"]} masks')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw', type=Path, required=True, help='Download dir with mvi2_XX.tar.gz and .verified/')
    p.add_argument('--data', type=Path, required=True, help='Output root (extracted/, all_assets_verification/)')
    p.add_argument('--category-json', type=Path, help='mvimgnet2_category.json from the official MVImgNet2.0 repo')
    p.add_argument('--archives', nargs='*', help='Subset of stems, e.g. mvi2_00 (default: all 41)')
    p.add_argument('--workers', type=int, default=24, help='Processes for hash/decode verification')
    p.add_argument('--delete-archives', action='store_true', help='Delete each archive once fully verified')
    p.add_argument('--watch', action='store_true', help='Keep waiting for downloads until all are processed')
    p.add_argument('--finalize', action='store_true', help='Write report.json/preparation_config.json and exit')
    args = p.parse_args()
    args.raw, args.data = args.raw.resolve(), args.data.resolve()
    args.data.mkdir(parents=True, exist_ok=True)
    if args.finalize:
        category = args.category_json or args.data / 'mvimgnet2_category.json'
        finalize(args.data, args.raw, category)
        return
    import fcntl
    lock = (args.data / 'extract.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.category_json:
        shutil.copyfile(args.category_json, args.data / 'mvimgnet2_category.json')
    wanted = args.archives or EXPECTED_ARCHIVES
    unknown = set(wanted) - set(EXPECTED_ARCHIVES)
    if unknown:
        p.error(f'unknown archives: {sorted(unknown)}')
    receipts = args.data / 'all_assets_verification' / 'archives'
    failed, last_wait, last_wait_log = set(), None, 0.0
    while True:
        pending = [s for s in wanted if not ((receipts / (s + '.json')).exists() and
                   json.loads((receipts / (s + '.json')).read_text())['state'] == 'verified')]
        if args.delete_archives:  # verified earlier but interrupted before deletion
            for stem in set(wanted) - set(pending):
                if (args.raw / (stem + '.tar.gz')).exists():
                    (args.raw / (stem + '.tar.gz')).unlink()
                    log(f'{stem}: deleted leftover verified archive')
        ready = [s for s in pending if s not in failed and (args.raw / '.verified' / (s + '.tar.gz.json')).exists()
                 and (args.raw / (s + '.tar.gz')).exists()]
        for stem in ready:
            try:
                process_archive(stem, args.raw, args.data, args.workers, args.delete_archives)
            except Exception:
                # Keep processing other archives; this one is retried on the next run.
                log(f'{stem}: FAILED (archive kept, skipped for the rest of this run)\n{traceback.format_exc()}')
                failed.add(stem)
        if not ready:
            waiting = [s for s in pending if s not in failed]
            if not waiting:
                log(f'done; failed={sorted(failed)}' if failed else 'all requested archives verified')
                raise SystemExit(1 if failed else 0)
            if not args.watch:
                log(f'not downloaded yet: {waiting}; failed={sorted(failed)}')
                raise SystemExit(1 if failed else 0)
            if waiting != last_wait or time.time() - last_wait_log > 1800:
                log(f'waiting for {len(waiting)} archives: {" ".join(waiting)}' + (f'; failed={sorted(failed)}' if failed else ''))
                last_wait, last_wait_log = waiting, time.time()
            time.sleep(120)

if __name__ == '__main__':
    main()
