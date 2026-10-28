"""Index all verified extracted objects; never open or extract TAR payloads."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path, PurePosixPath

from .prepare_full_bin15 import dump, finalize, prepare_instance


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def index_archive(data, output, stem, preserved, categories, identity):
    state_path = output / 'archive_status' / (stem + '.json')
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        assert previous['identity'] == identity, 'Index source/config changed'
        if previous['state'] == 'completed':
            return previous
    receipt = json.loads((data / 'all_assets_verification/archives' / (stem + '.json')).read_text())
    assert receipt['state'] == 'verified' and receipt['gzip_crc_verified'] and receipt['image_errors'] == 0
    uids = json.loads((data / 'all_assets_verification/instances' / (stem + '.json')).read_text())
    assert len(uids) == receipt['instances'] and len(uids) == len(set(uids))
    state = dict(state='indexing', identity=identity, pid=os.getpid(), archive=stem,
                 instances_total=len(uids), instances_found=0, instances_usable=0,
                 instances_rejected=0, views_usable=0, started_at=time.time(),
                 gzip_crc_verified=True, verification_receipt_sha256=digest(data / 'all_assets_verification/archives' / (stem + '.json')))
    accepted = output / 'archive_manifests' / (stem + '.jsonl')
    rejected = output / 'archive_manifests' / (stem + '.rejected.jsonl')
    accepted.parent.mkdir(parents=True, exist_ok=True)
    last = 0
    try:
        with accepted.with_suffix('.tmp').open('w') as good, rejected.with_suffix('.tmp').open('w') as bad:
            for uid in uids:
                parts = PurePosixPath(uid)
                assert len(parts.parts) == 2 and '..' not in parts.parts and not parts.is_absolute()
                state['instances_found'] += 1
                try:
                    row = prepare_instance(data / 'extracted' / stem / uid, categories, preserved)
                except (ValueError, KeyError, OSError, EOFError) as error:
                    # Unexpected I/O/resource failures must stop preparation, not discard data.
                    if isinstance(error, OSError) and error.errno not in (None, 2):
                        raise
                    bad.write(json.dumps(dict(instance_uid=uid, reason=str(error), error_type=type(error).__name__)) + '\n')
                    state['instances_rejected'] += 1
                else:
                    good.write(json.dumps(row, allow_nan=False) + '\n')
                    state['instances_usable'] += 1
                    state['views_usable'] += row['num_valid_views']
                if time.monotonic() - last > 5:
                    state['updated_at'] = time.time()
                    dump(state_path, state)
                    last = time.monotonic()
        os.replace(accepted.with_suffix('.tmp'), accepted)
        os.replace(rejected.with_suffix('.tmp'), rejected)
        assert state['instances_found'] == state['instances_usable'] + state['instances_rejected'] == len(uids)
        state.update(state='completed', finished_at=time.time())
        dump(state_path, state)
        return state
    except Exception as error:
        state.update(state='failed', error=str(error), traceback=traceback.format_exc())
        dump(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--reuse-from', type=Path)
    parser.add_argument('--min-free-gb', type=float, default=50)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.output / 'index.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verification = json.loads((args.data / 'all_assets_verification/report.json').read_text())
    assert verification['state'] == 'verified' and verification['archive_count'] == len(verification['archives'])
    assert verification['image_errors'] == 0 and verification['byte_verified_files'] == verification['regular_files']
    assert shutil.disk_usage(args.output).free > args.min_free_gb * 1e9
    original = json.loads((args.data / 'preparation_config.json').read_text())
    archives = [Path(a['path']) for a in original['archives']]
    assert [p.name.removesuffix('.tar.gz') for p in archives] == verification['archives']
    # Archives may be deleted after verified extraction; their receipts carry the identity.
    for path, record in zip(archives, original['archives']):
        receipt = json.loads((args.data / 'all_assets_verification/archives' / (path.name.removesuffix('.tar.gz') + '.json')).read_text())
        assert receipt['state'] == 'verified' and receipt['archive_bytes'] == record['bytes']
    preserved = original['preserved_pilot_split']
    labels = args.data / 'mvimgnet2_category.json'
    categories = json.loads(labels.read_text())
    config = {**original, 'version': 2, 'workers': args.workers,
              'mode': 'index_existing_verified_files', 'extracted_data': str(args.data),
              'verification_report_sha256': digest(args.data / 'all_assets_verification/report.json'),
              'index_source_sha256': digest(__file__),
              'geometry_source_sha256': digest(Path(__file__).with_name('prepare_full_bin15.py')),
              'category_sha256': digest(labels),
              'parse_error_policy': 'Reject malformed object camera/pose metadata, including EOFError; fail on resource or unexpected program errors.',
              'reuse_from': str(args.reuse_from) if args.reuse_from else None}
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    cp = args.output / 'preparation_config.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == config, 'Resume configuration mismatch'
    dump(cp, config)
    shutil.copyfile(labels, args.output / labels.name)
    if args.reuse_from:
        old_config = json.loads((args.reuse_from / 'preparation_config.json').read_text())
        for key in ['archives', 'preserved_pilot_split', 'split_policy', 'geometry_source_sha256',
                    'verification_report_sha256', 'category_sha256']:
            assert old_config[key] == config[key], 'Incompatible completed index: ' + key
        old_identity = hashlib.sha256(json.dumps(old_config, sort_keys=True).encode()).hexdigest()
        for archive in archives:
            stem = archive.name.removesuffix('.tar.gz')
            source_state = args.reuse_from / 'archive_status' / (stem + '.json')
            target_state = args.output / 'archive_status' / (stem + '.json')
            if target_state.exists() or not source_state.exists():
                continue
            previous = json.loads(source_state.read_text())
            if previous['state'] != 'completed':
                continue
            assert previous['identity'] == old_identity
            assert previous['instances_found'] == previous['instances_total']
            assert previous['instances_found'] == previous['instances_usable'] + previous['instances_rejected']
            receipt = args.data / 'all_assets_verification/archives' / (stem + '.json')
            assert previous['verification_receipt_sha256'] == digest(receipt)
            fingerprints = {}
            for suffix in ['.jsonl', '.rejected.jsonl']:
                source = args.reuse_from / 'archive_manifests' / (stem + suffix)
                target = args.output / 'archive_manifests' / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                with source.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        h.update(chunk)
                fingerprints[source.name] = h.hexdigest()
                if not target.exists():
                    os.link(source, target)
                else:
                    assert os.path.samefile(source, target)
            previous.update(identity=identity, reused_from=str(args.reuse_from),
                            reused_source_identity=old_identity, reused_files_sha256=fingerprints)
            dump(target_state, previous)
    status_path = args.output / 'preparation_status.json'
    started = time.time()
    dump(status_path, dict(state='running', pid=os.getpid(), started_at=started, workers=args.workers))
    try:
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(index_archive, args.data, args.output, p.name.removesuffix('.tar.gz'), preserved, categories, identity) for p in archives]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                print(json.dumps(results[-1]), flush=True)
                dump(status_path, dict(state='running', pid=os.getpid(), started_at=started,
                     completed_archives=len(results), total_archives=len(archives), workers=args.workers))
        assert sum(r['instances_found'] for r in results) == verification['unique_object_folders']
        report = finalize(args.output, archives, preserved, {
            'version': 2, 'raw_objects': verification['unique_object_folders'],
            'verification_report_sha256': config['verification_report_sha256'],
            'index_config_identity': identity,
            'image_validation': 'All extracted RGB and masks fully decoded; all file bytes SHA256 verified against TAR members; all gzip EOF/CRC verified.',
            'scope': f'{len(archives)} verified archives and all their extracted object folders; no category/object/view cap. Camera-valid objects split 90/5/5 by UID with pilot split preserved.',
            'all_assets_root': str(args.data), 'index_started_at': started})
        dump(status_path, report)
        print(json.dumps(report), flush=True)
    except Exception as error:
        dump(status_path, dict(state='failed', error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
