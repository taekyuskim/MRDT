"""Prepare every MVImgNet2 archive, preserving the bin15 pilot geometry/splits."""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import struct
import tarfile
import threading
import time
import traceback
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageOps

from .audit import read_category_map
from .audit_followup import _az_el_r_norm, _ref_image_up_basis
from .bin15_camera import bin_pair, camera_feature_from_bins
from .cameras import processed_intrinsics
from .colmap_io import Image as ColmapImage, camera_center_world, read_cameras_binary

VERSION = 1
STOP = threading.Event()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    os.replace(temp, path)


def unpack(handle, fmt):
    count = struct.calcsize('<' + fmt)
    data = handle.read(count)
    if len(data) != count:
        raise ValueError('Truncated COLMAP binary')
    return struct.unpack('<' + fmt, data)


def pose_model(instance):
    """Skip 2D correspondences/tracks; keep the exact first 50k point XYZs."""
    sparse = Path(instance) / 'sparse/0'
    cameras = read_cameras_binary(sparse / 'cameras.bin')
    images = {}
    image_path = sparse / 'images.bin'
    with image_path.open('rb') as stream:
        count = unpack(stream, 'Q')[0]
        if count > image_path.stat().st_size // 73:
            raise ValueError('Invalid image count')
        for _ in range(count):
            values = unpack(stream, 'idddddddi')
            name = bytearray()
            while True:
                char = stream.read(1)
                if not char or len(name) > 8192:
                    raise ValueError('Invalid COLMAP image name')
                if char == b'\0':
                    break
                name.extend(char)
            points = unpack(stream, 'Q')[0]
            stream.seek(points * 24, 1)
            if stream.tell() > image_path.stat().st_size:
                raise ValueError('Truncated image correspondences')
            images[values[0]] = ColmapImage(values[0], np.array(values[1:5]),
                np.array(values[5:8]), values[8], name.decode('utf-8'), None, None)
    xyz = []
    point_path = sparse / 'points3D.bin'
    if point_path.is_file():
        with point_path.open('rb') as stream:
            count = unpack(stream, 'Q')[0]
            if count > point_path.stat().st_size // 51:
                raise ValueError('Invalid point count')
            for index in range(count):
                values = unpack(stream, 'QdddBBBdQ')
                if index < 50000:
                    xyz.append(values[1:4])
                stream.seek(values[-1] * 8, 1)
                if stream.tell() > point_path.stat().st_size:
                    raise ValueError('Truncated point tracks')
    return cameras, images, np.asarray(xyz, dtype=np.float64)


def split_for(uid, preserved):
    if uid in preserved:
        return preserved[uid]
    fraction = int.from_bytes(hashlib.sha256(('mvimgnet2-full-v1:' + uid).encode()).digest()[:8], 'big') / 2**64
    return 'train' if fraction < .9 else 'val' if fraction < .95 else 'test'


def prepare_instance(instance, categories, preserved):
    instance = Path(instance)
    uid = '/'.join(instance.parts[-2:])
    cameras, images, xyz = pose_model(instance)
    valid = []
    paths = {}
    sizes = {}
    invalid = []
    for image_id in sorted(images):
        rec = images[image_id]
        relative = PurePosixPath(rec.name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe COLMAP image path')
        path = instance / 'images' / relative.name
        try:
            cam = cameras[rec.camera_id]
            # Current model has a single distortion-k channel.
            if cam.model != 'SIMPLE_RADIAL':
                raise ValueError('unsupported_camera_model:' + cam.model)
            if not np.isfinite(np.r_[rec.qvec, rec.tvec, cam.params]).all() or np.linalg.norm(rec.qvec) < 1e-8:
                raise ValueError('nonfinite_or_degenerate_camera')
            with Image.open(path) as im:
                if im.getexif().get(274, 1) not in (None, 1):
                    raise ValueError('exif_rotation_requires_camera_remap')
                sizes[image_id] = im.size
                im.verify()
            valid.append(image_id)
            paths[image_id] = path
        except (OSError, ValueError, KeyError) as error:
            invalid.append({'image_id': image_id, 'reason': str(error)})
    if len(valid) < 4:
        raise ValueError('too_few_valid_views:' + json.dumps(invalid[:4]))
    centers = np.stack([camera_center_world(images[i].qvec, images[i].tvec) for i in valid])
    object_center = np.median(xyz if len(xyz) else centers.mean(axis=0, keepdims=True), axis=0)
    basis = _ref_image_up_basis(centers, object_center, images[valid[0]])
    axes = np.stack([basis['x'], basis['y'], basis['up']])
    if not np.isfinite(axes).all() or not np.allclose(axes @ axes.T, np.eye(3), atol=1e-5) or basis['reference_radius'] < 1e-8:
        raise ValueError('degenerate_reference_frame')
    views = []
    for i, center in zip(valid, centers):
        rec = images[i]
        cam = cameras[rec.camera_id]
        az, el, radius = _az_el_r_norm(center, basis)
        if not np.isfinite([az, el, radius]).all() or radius <= 0:
            raise ValueError('invalid_fixed_camera')
        az_bin, el_bin = bin_pair(az, el)
        k256, _, resize = processed_intrinsics(cam, *sizes[i], 256)
        features = camera_feature_from_bins(az_bin, el_bin, radius, k256, cam.params[-1])
        if not np.isfinite(features).all():
            raise ValueError('nonfinite_camera_features')
        views.append({'image_id': i, 'frame_name': rec.name, 'path': str(paths[i]),
            'az_bin': az_bin, 'el_bin': el_bin, 'azimuth_deg': az, 'elevation_deg': el,
            'radius_refnorm': radius, 'features': features.tolist(),
            'qvec': rec.qvec.tolist(), 'tvec': rec.tvec.tolist(),
            'camera_model': cam.model, 'camera_params': cam.params.tolist(), 'resize': resize})
    occupied = len({(v['az_bin'], v['el_bin']) for v in views})
    if occupied < 3:
        raise ValueError('too_few_occupied_bins:' + str(occupied))
    return {'instance_uid': uid, 'instance_id': instance.name, 'class_id': instance.parent.name,
        'class_name': categories.get(instance.parent.name), 'instance_path': str(instance),
        'split': split_for(uid, preserved), 'valid_image_ids': valid, 'num_valid_views': len(valid),
        'num_registered_images': len(images), 'occupied_bins': occupied, 'invalid_views': invalid,
        'reference_image_id': valid[0], 'object_center': object_center.tolist(),
        'reference_radius': basis['reference_radius'], 'text': None, 'text_available': False,
        'views': views}


def safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError('Unsafe archive member:' + name)
    return path


def keep_member(parts):
    return len(parts) >= 4 and (parts[2] == 'images' or (
        len(parts) == 5 and parts[2:4] == ('sparse', '0') and
        parts[4] in {'cameras.bin', 'images.bin', 'points3D.bin'}))


def process_archive(archive, output, categories, preserved, reserve_bytes):
    archive = Path(archive)
    stem = archive.name.removesuffix('.tar.gz')
    state_path = output / 'archive_status' / (stem + '.json')
    fingerprint = {'name': archive.name, 'bytes': archive.stat().st_size,
                   'mtime_ns': archive.stat().st_mtime_ns, 'version': VERSION}
    if state_path.exists():
        old = json.loads(state_path.read_text())
        if old.get('fingerprint') != fingerprint:
            raise RuntimeError('Source/config changed for ' + archive.name)
        if old.get('state') == 'completed':
            return old
    destination = output / 'extracted' / stem
    destination.mkdir(parents=True, exist_ok=True)
    state = {'state': 'extracting', 'fingerprint': fingerprint, 'members_seen': 0,
             'selected_bytes': 0, 'written_bytes': 0, 'files_written': 0,
             'skipped_members': 0, 'started_at': time.time()}
    last_report = 0
    def progress():
        nonlocal last_report
        state['updated_at'] = time.time()
        dump(state_path, state)
        last_report = time.monotonic()
    progress()
    try:
        # Drain gzip to EOF after TAR EOF to validate its CRC/footer as well.
        with archive.open('rb') as raw, gzip.GzipFile(fileobj=raw) as zipped:
            with tarfile.open(fileobj=zipped, mode='r|', bufsize=1024 * 1024) as tar:
                for member in tar:
                    if STOP.is_set():
                        raise RuntimeError('Stopped after another archive failed')
                    relative = safe_member(member.name)
                    state['members_seen'] += 1
                    if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                        raise ValueError('Unsupported archive member type:' + member.name)
                    if member.isfile() and keep_member(relative.parts):
                        target = destination.joinpath(*relative.parts)
                        if not target.resolve().is_relative_to(destination.resolve()):
                            raise ValueError('Archive path escaped output')
                        state['selected_bytes'] += member.size
                        if target.is_symlink():
                            raise ValueError('Unexpected output symlink')
                        if not (target.is_file() and target.stat().st_size == member.size):
                            if shutil.disk_usage(output).free < reserve_bytes + member.size:
                                raise RuntimeError('Disk reserve reached; resume after freeing space')
                            target.parent.mkdir(parents=True, exist_ok=True)
                            temporary = target.with_name(target.name + '.partial')
                            with tar.extractfile(member) as source, temporary.open('wb') as sink:
                                shutil.copyfileobj(source, sink, 1024 * 1024)
                            if temporary.stat().st_size != member.size:
                                raise ValueError('Extracted size mismatch')
                            os.replace(temporary, target)
                            state['files_written'] += 1
                            state['written_bytes'] += member.size
                    else:
                        state['skipped_members'] += 1
                    tar.members.clear()
                    if time.monotonic() - last_report > 5:
                        state['compressed_bytes_read'] = raw.tell()
                        progress()
            while zipped.read(1024 * 1024):
                if STOP.is_set():
                    raise RuntimeError('Stopped')
        state.update(state='indexing', gzip_crc_verified=True, compressed_bytes_read=archive.stat().st_size)
        progress()
        accepted_path = output / 'archive_manifests' / (stem + '.jsonl')
        rejected_path = output / 'archive_manifests' / (stem + '.rejected.jsonl')
        accepted_path.parent.mkdir(parents=True, exist_ok=True)
        state.update(instances_found=0, instances_usable=0, views_usable=0, instances_rejected=0)
        with accepted_path.with_suffix('.tmp').open('w') as accepted, rejected_path.with_suffix('.tmp').open('w') as rejected:
            for class_dir in sorted(destination.iterdir()):
                if not class_dir.is_dir():
                    continue
                for instance in sorted(class_dir.iterdir()):
                    if STOP.is_set():
                        raise RuntimeError('Stopped after another archive failed')
                    if not instance.is_dir():
                        continue
                    state['instances_found'] += 1
                    try:
                        row = prepare_instance(instance, categories, preserved)
                        accepted.write(json.dumps(row, allow_nan=False) + '\n')
                        state['instances_usable'] += 1
                        state['views_usable'] += row['num_valid_views']
                    except Exception as error:
                        if isinstance(error, (MemoryError, OSError)) and getattr(error, 'errno', None) in (12, 28):
                            raise
                        rejected.write(json.dumps({'instance_uid': '/'.join(instance.parts[-2:]),
                            'reason': str(error), 'error_type': type(error).__name__}) + '\n')
                        state['instances_rejected'] += 1
                    if time.monotonic() - last_report > 5:
                        progress()
        os.replace(accepted_path.with_suffix('.tmp'), accepted_path)
        os.replace(rejected_path.with_suffix('.tmp'), rejected_path)
        state.update(state='completed', finished_at=time.time())
        progress()
        return state
    except Exception as error:
        STOP.set()
        state.update(state='failed', error=str(error), traceback=traceback.format_exc())
        progress()
        raise


def finalize(output, archives, preserved, report_fields=None):
    temporary = output / 'catalog.build.sqlite'
    connection = sqlite3.connect(temporary)
    connection.executescript('DROP TABLE IF EXISTS instances; CREATE TABLE instances('
        'uid TEXT PRIMARY KEY, split TEXT NOT NULL, ordinal INTEGER NOT NULL, row_json TEXT NOT NULL);'
        'CREATE UNIQUE INDEX split_ordinal ON instances(split, ordinal);')
    counts = Counter()
    views = Counter()
    categories = set()
    seen_preserved = set()
    manifest_root = output / 'manifests'
    manifest_root.mkdir(exist_ok=True)
    streams = {s: (manifest_root / (s + '_instances.jsonl.tmp')).open('w') for s in ['train', 'val', 'test']}
    rejected_reasons = Counter()
    try:
        for archive in archives:
            stem = archive.name.removesuffix('.tar.gz')
            state = json.loads((output / 'archive_status' / (stem + '.json')).read_text())
            assert state['state'] == 'completed' and state['gzip_crc_verified']
            with (output / 'archive_manifests' / (stem + '.jsonl')).open() as source:
                for line in source:
                    row = json.loads(line)
                    uid, split = row['instance_uid'], row['split']
                    if uid in preserved:
                        assert split == preserved[uid]
                        seen_preserved.add(uid)
                    connection.execute('INSERT INTO instances VALUES(?,?,?,?)', (uid, split, counts[split], line))
                    summary = {k: v for k, v in row.items() if k != 'views'}
                    streams[split].write(json.dumps(summary) + '\n')
                    counts[split] += 1
                    views[split] += row['num_valid_views']
                    categories.add(row['class_id'])
                    if sum(counts.values()) % 1000 == 0:
                        connection.commit()
            with (output / 'archive_manifests' / (stem + '.rejected.jsonl')).open() as source:
                for line in source:
                    rejected_reasons[json.loads(line)['reason'].split(':', 1)[0]] += 1
        connection.commit()
        assert all(counts[s] for s in streams), 'Empty split'
        assert seen_preserved == set(preserved), 'A preserved pilot object was rejected or missing'
        assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    finally:
        connection.close()
        for stream in streams.values():
            stream.close()
    os.replace(temporary, output / 'catalog.sqlite')
    for split in streams:
        os.replace(manifest_root / (split + '_instances.jsonl.tmp'), manifest_root / (split + '_instances.jsonl'))
    report = {'state': 'ready', 'version': VERSION, 'archive_count': len(archives),
        'all_archives_crc_verified': True, 'instances_by_split': dict(counts), 'views_by_split': dict(views),
        'classes': len(categories), 'rejections_by_reason': dict(rejected_reasons),
        'preserved_pilot_instances': sorted(seen_preserved), 'completed_at': time.time(),
        'camera_convention': 'reference-image-up; first valid image; median first50000 sparse points; bin15',
        'image_validation': 'PIL header/verify and gzip CRC; no exhaustive pixel decode',
        'scope': 'All 41 input archives; no category/object/view cap. RGB and sparse model assets extracted; masks remain in original archives.'}
    report.update(report_fields or {})
    dump(output / 'readiness.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/data1/haksoo/data/MVImgNet2.0'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--reserve-gb', type=float, default=500)
    args = parser.parse_args()
    archives = sorted((args.root / 'raw').glob('mvi2_*.tar.gz'))
    assert [p.name for p in archives] == [f'mvi2_{i:02}.tar.gz' for i in range(41)], 'Expected all 41 archives'
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.output / 'prepare.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    preserved = {}
    for split in ['train', 'val', 'test']:
        for line in (args.root / 'processed_plan_b/manifests' / (split + '_instances.jsonl')).read_text().splitlines():
            uid = json.loads(line)['instance_uid']
            assert uid not in preserved
            preserved[uid] = split
    categories = read_category_map(args.root / 'repo/mvimgnet_category.txt')
    config = {'version': VERSION, 'scope': 'all_41_archives', 'instance_limit': None,
        'archive_limit': None, 'archives': [{'path': str(p), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in archives],
        'preserved_pilot_split': preserved, 'split_policy': 'SHA256 mvimgnet2-full-v1 uid,90/5/5; pilot overrides',
        'workers': args.workers, 'reserve_bytes': int(args.reserve_gb * 1e9),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    config_path = args.output / 'preparation_config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Resume configuration mismatch'
    dump(config_path, config)
    dump(args.output / 'preparation_status.json', {'state': 'running', 'pid': os.getpid(), 'started_at': time.time()})
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_archive, p, args.output, categories, preserved, config['reserve_bytes']) for p in archives]
            for future in concurrent.futures.as_completed(futures):
                print(json.dumps(future.result()), flush=True)
        report = finalize(args.output, archives, preserved)
        dump(args.output / 'preparation_status.json', report)
        print(json.dumps(report), flush=True)
    except Exception as error:
        dump(args.output / 'preparation_status.json', {'state': 'failed', 'error': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
