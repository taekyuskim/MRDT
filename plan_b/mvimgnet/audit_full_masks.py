"""Audit pairing and every mask, using existing verified source manifests."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from PIL import Image
from .prepare_full_bin15 import dump


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def audit_archive(data, output, stem, identity):
    state_path=output/'archives'/(stem+'.json')
    if state_path.exists():
        old=json.loads(state_path.read_text())
        assert old['identity']==identity
        if old['state']=='completed':return old
    receipt=json.loads((data/'all_assets_verification/archives'/(stem+'.json')).read_text())
    assert receipt['state']=='verified' and receipt['image_errors']==0
    manifest=data/'all_assets_verification/manifests'/(stem+'.jsonl.gz')
    assert file_digest(manifest)==receipt['manifest_sha256']
    rgb={}; masks=[]
    with gzip.open(manifest,'rt') as f:
        for line in f:
            row=json.loads(line)
            if row['kind'] not in ('images','masks'):continue
            assert row['byte_verified'] and row['image_decoded']
            parts=Path(row['path']).parts
            assert len(parts)==4 and parts[2]==row['kind']
            key='/'.join(parts[:2])+'/'+parts[3]
            if row['kind']=='images':
                assert key not in rgb
                rgb[key]=row
            else:masks.append((key,row))
    assert len(rgb)==len(masks)
    target=output/'views'/(stem+'.jsonl');target.parent.mkdir(parents=True,exist_ok=True)
    state=dict(state='auditing',identity=identity,archive=stem,total_masks=len(masks),checked=0,
               rejected=0,full_foreground=0,started_at=time.time(),reasons={},categories={})
    last=0;seen=set();reasons=Counter();categories=Counter()
    try:
        with target.with_suffix('.tmp').open('w') as sink:
            for key,record in masks:
                assert key not in seen and key in rgb
                seen.add(key);image=rgb[key]
                path=data/'extracted'/stem/record['path']
                stat=path.stat()
                assert (stat.st_size,stat.st_mtime_ns)==(record['bytes'],record['mtime_ns']), 'Verified mask changed'
                item=dict(key=key,mask_path=str(path),mask_sha256=record['sha256'],valid=True)
                try:
                    if record['image']['size']!=image['image']['size']:
                        raise ValueError('mask_rgb_size_mismatch')
                    with Image.open(path) as im:
                        if im.mode!='L':raise ValueError('unsupported_mask_mode:'+im.mode)
                        if im.getexif().get(274,1) not in (None,1):raise ValueError('mask_exif_rotation')
                        histogram=im.histogram()
                        area=sum(histogram[128:]);pixels=im.width*im.height
                        item.update(foreground_pixels=area,pixels=pixels,foreground_fraction=area/pixels)
                        if not area:raise ValueError('empty_foreground_mask')
                        state['full_foreground']+=int(area==pixels)
                except (ValueError,OSError,EOFError) as error:
                    if isinstance(error,OSError) and error.errno not in (None,2):raise
                    item.update(valid=False,reason=str(error))
                    reasons[str(error).split(':')[0]]+=1;state['rejected']+=1
                sink.write(json.dumps(item)+'\n')
                state['checked']+=1;categories[key.split('/')[0]]+=1
                if time.monotonic()-last>10:
                    state.update(reasons=dict(reasons),categories=dict(categories),updated_at=time.time())
                    dump(state_path,state);last=time.monotonic()
        assert seen==set(rgb)
        os.replace(target.with_suffix('.tmp'),target)
        state.update(state='completed',finished_at=time.time(),reasons=dict(reasons),categories=dict(categories),
                     source_receipt_sha256=file_digest(data/'all_assets_verification/archives'/(stem+'.json')),
                     audit_sha256=file_digest(target))
        dump(state_path,state);return state
    except Exception as error:
        state.update(state='failed',error=str(error),traceback=traceback.format_exc());dump(state_path,state);raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(args.output/'audit.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verification=json.loads((args.data/'all_assets_verification/report.json').read_text())
    stems=verification['archives']
    config={'version':1,'source_report_sha256':file_digest(args.data/'all_assets_verification/report.json'),
            'script_sha256':file_digest(__file__),'threshold':128,'foreground':'white','workers':args.workers}
    identity=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if (args.output/'config.json').exists():assert json.loads((args.output/'config.json').read_text())==config
    dump(args.output/'config.json',config)
    started=time.time();dump(args.output/'status.json',dict(state='running',pid=os.getpid(),started_at=started))
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(audit_archive,args.data,args.output,stem,identity) for stem in stems]
            results=[]
            for future in as_completed(futures):
                results.append(future.result());print(json.dumps(results[-1]),flush=True)
                dump(args.output/'status.json',dict(state='running',pid=os.getpid(),started_at=started,completed_archives=len(results)))
        reasons=Counter()
        for record in results:reasons.update(record['reasons'])
        report=dict(state='ready',archive_count=len(stems),checked_masks=sum(r['checked'] for r in results),
                    rejected_masks=sum(r['rejected'] for r in results),full_foreground_masks=sum(r['full_foreground'] for r in results),
                    rejections_by_reason=dict(reasons),identity=identity,started_at=started,finished_at=time.time(),
                    semantic_accuracy='Human inspection is sampled; full decoding/pairing/empty-mask checks do not guarantee segmentation correctness.')
        assert report['checked_masks']==verification['masks']
        dump(args.output/'report.json',report);dump(args.output/'status.json',report)
    except Exception as error:
        dump(args.output/'status.json',dict(state='failed',error=str(error),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
