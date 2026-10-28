"""Join verified camera metadata and exhaustive mask audit without changing poses."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from .prepare_full_bin15 import dump, finalize
from .audit_full_masks import file_digest
from .masked_bin15_dataset import PREPROCESS_POLICY


def add_masks(row, audited):
    good=[]; bad=[]
    for view in row['views']:
        # COLMAP names may include an images/ prefix; the verified audit is
        # keyed by the actual extracted object UID and filename.
        image_path=Path(view['path'])
        assert image_path.parent.name=='images'
        assert image_path.parent.parent==Path(row['instance_path'])
        assert '/'.join(image_path.parts[-4:-2])==row['instance_uid']
        assert Path(view['frame_name']).name==image_path.name
        key=row['instance_uid']+'/'+image_path.name
        mask=audited[key]
        expected=image_path.parent.parent/'masks'/image_path.name
        assert Path(mask['mask_path'])==expected, 'Frame/mask mismatch: '+key
        if mask['valid']:
            good.append({**view, 'mask_path':mask['mask_path'], 'mask_sha256':mask['mask_sha256'],
                         'foreground_fraction':mask['foreground_fraction']})
        else:bad.append({'image_id':view['image_id'],'reason':mask['reason']})
    bins=len({(v['az_bin'],v['el_bin']) for v in good})
    result={**row,'views':good,'valid_image_ids':[v['image_id'] for v in good],
            'num_valid_views':len(good),'occupied_bins':bins,
            'invalid_views':row['invalid_views']+bad,'mask_rejected_views':bad,
            'preprocessing_policy':PREPROCESS_POLICY}
    # Keep the original reference camera basis even if its mask was rejected.
    reason='mask_too_few_valid_views' if len(good)<4 else ('mask_too_few_occupied_bins' if bins<3 else None)
    return result, reason


def main():
    p=argparse.ArgumentParser();p.add_argument('--camera',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(args.output/'build.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    camera=json.loads((args.camera/'readiness.json').read_text());audit=json.loads((args.audit/'report.json').read_text())
    assert camera['state']=='ready' and audit['state']=='ready' and camera['archive_count']==audit['archive_count']
    original=json.loads((args.camera/'preparation_config.json').read_text())
    config={**original,'mode':'camera_mask_join','camera_readiness_sha256':file_digest(args.camera/'readiness.json'),
            'mask_audit_report_sha256':file_digest(args.audit/'report.json'), 'mask_audit_identity':audit['identity'],
            'preprocessing_policy':PREPROCESS_POLICY,'masked_dataset_sha256':file_digest(Path(__file__).with_name('masked_bin15_dataset.py')),
            'builder_sha256':file_digest(__file__)}
    identity=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    cfg=args.output/'preparation_config.json'
    if cfg.exists():assert json.loads(cfg.read_text())==config
    dump(cfg,config);shutil.copyfile(args.camera/'mvimgnet2_category.json',args.output/'mvimgnet2_category.json')
    dump(args.output/'preparation_status.json',dict(state='building',pid=os.getpid(),started_at=time.time()))
    before=Counter();after=Counter();rejected_views=Counter();mask_object_reasons=Counter()
    try:
        for i,archive in enumerate(original['archives']):
            stem=Path(archive['path']).name.removesuffix('.tar.gz'); state=json.loads((args.audit/'archives'/(stem+'.json')).read_text())
            assert state['state']=='completed' and state['identity']==audit['identity']
            audit_path=args.audit/'views'/(stem+'.jsonl')
            assert file_digest(audit_path)==state['audit_sha256']
            audited={}
            with audit_path.open() as f:
                for line in f:
                    v=json.loads(line);assert v['key'] not in audited;audited[v['key']]=v
            assert len(audited)==state['checked']
            folder=args.output/'archive_manifests';folder.mkdir(exist_ok=True)
            source=args.camera/'archive_manifests'
            accepted=folder/(stem+'.jsonl'); rejected=folder/(stem+'.rejected.jsonl')
            found=usable=removed=views=0
            with accepted.with_suffix('.tmp').open('w') as good,rejected.with_suffix('.tmp').open('w') as bad:
                with (source/(stem+'.rejected.jsonl')).open() as f:
                    for line in f:
                        bad.write(line);found+=1;removed+=1
                with (source/(stem+'.jsonl')).open() as f:
                    for line in f:
                        row=json.loads(line);category=row['class_id'];before[category]+=1;found+=1
                        updated,reason=add_masks(row,audited)
                        rejected_views[category]+=len(updated['mask_rejected_views'])
                        if reason:
                            bad.write(json.dumps({'instance_uid':row['instance_uid'],'class_id':category,
                                'reason':reason,'mask_rejected_views':updated['mask_rejected_views']})+'\n')
                            removed+=1;mask_object_reasons[reason]+=1
                        else:
                            good.write(json.dumps(updated,allow_nan=False)+'\n');usable+=1;views+=updated['num_valid_views'];after[category]+=1
            os.replace(accepted.with_suffix('.tmp'),accepted);os.replace(rejected.with_suffix('.tmp'),rejected)
            original_state=json.loads((args.camera/'archive_status'/(stem+'.json')).read_text())
            assert found==original_state['instances_found']
            dump(args.output/'archive_status'/(stem+'.json'),dict(state='completed',identity=identity,archive=stem,
                instances_found=found,instances_usable=usable,instances_rejected=removed,views_usable=views,gzip_crc_verified=True))
            dump(args.output/'preparation_status.json',dict(state='building',pid=os.getpid(),completed_archives=i+1))
        coverage={'camera_objects_by_category':dict(before),'masked_objects_by_category':dict(after),
                  'mask_rejected_views_by_category':dict(rejected_views),'mask_object_rejections':dict(mask_object_reasons),
                  'dropped_categories':sorted(set(before)-set(after))}
        dump(args.output/'mask_coverage.json',coverage)
        assert not coverage['dropped_categories'], 'Mask filtering eliminated a complete category; inspect coverage'
        archives=[Path(a['path']) for a in original['archives']]
        report=finalize(args.output,archives,original['preserved_pilot_split'],{
            'version':3,'raw_objects':camera['raw_objects'],'preprocessing_policy':PREPROCESS_POLICY,'mask_audit_identity':audit['identity'],
            'mask_audit_report_sha256':config['mask_audit_report_sha256'],'all_masks_audited':True,
            'checked_masks':audit['checked_masks'],'rejected_masks':audit['rejected_masks'],
            'masked_dataset_sha256':config['masked_dataset_sha256'],'index_config_identity':identity,
            'image_validation':'All RGB/mask decoded and SHA256 byte verified; exhaustive same-frame mask pairing/size/threshold/empty checks.',
            'scope':f"{camera['archive_count']} verified archives, no category/object cap. Camera and mask validity filtering is explicit and counted.",
            'camera_convention':'Unchanged original full-frame reference-image-up camera basis; mask filtering does not redefine it.',
            'mask_coverage_sha256':file_digest(args.output/'mask_coverage.json')})
        assert report['classes']==camera['classes']
        dump(args.output/'preparation_status.json',report);print(json.dumps(report),flush=True)
    except Exception as error:
        # A late failed check must not leave a consumable readiness flag.
        dump(args.output/'readiness.json',dict(state='failed',error=str(error)))
        dump(args.output/'preparation_status.json',dict(state='failed',error=str(error),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
