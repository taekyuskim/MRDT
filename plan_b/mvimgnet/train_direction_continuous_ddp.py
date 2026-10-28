"""Continuous direction-only full training with preserved bin-based view selection."""
import argparse
import contextlib
import copy
import datetime
import hashlib
import json
import math
import os
import signal
import time
from unittest.mock import patch
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from .direction_bin15_model import DirectionBin15RDT
from .continuous_direction_dataset import (ContinuousDirectionDataset as DirectionFullBin15Dataset,
    INVALID_VIEW_POLICY, PREPROCESS_POLICY, CONDITIONING_POLICY, ANGLE_SOURCE, SELECTION_BIN_DEG)
from .bin15_dataset import mvimgnet_bin15_collate
from .epoch_sampling import RankSamples, SAMPLING_POLICY
from .train_bin15_pilot import _pilot_rdt_config, rdt_config_for, make_train_batch, save_sample_panel, apply_precision
from .prepare_full_bin15 import dump
from . import view_cache, latent_cache
from .val_loss import ValLoss
from .class_text import TextBank
from models.ema_model import EMAModel
from rescene.models.encoders import build_image_encoder
from rescene.codec.vae_latent_delta import load_vae

# Frozen SigLIP/VAE location: $MRDT_MODEL_ROOT if set, else <repo>/pretrained.
_MODEL_ROOT=os.environ.get('MRDT_MODEL_ROOT',str(Path(__file__).resolve().parents[2]/'pretrained'))
FEATURE_MODELS={'siglip_model': f'{_MODEL_ROOT}/siglip-so400m-patch14-384', 'vae_id': f'{_MODEL_ROOT}/sd-vae-ft-mse'}
SOURCE_NAMES=['train_direction_continuous_ddp.py','continuous_direction_dataset.py','roll_canonical_object.py','roll_direction_dataset.py','colmap_io.py','direction_bin15_dataset.py','direction_bin15_model.py',
              'canonical_object.py','bin15_model.py','bin15_dataset.py','bin15_camera.py',
              'full_bin15_dataset.py','epoch_sampling.py','train_bin15_pilot.py','view_cache.py','val_loss.py','class_text.py','latent_cache.py']
EMA_SOURCE=Path(__file__).resolve().parents[2]/'models'/'ema_model.py'
# RDT backbone files live outside this directory; hash them into the run config too.
MODEL_SOURCES=[Path(__file__).resolve().parents[2]/'models'/n for n in ['rdt/model.py','rdt/blocks.py','rdt_runner.py']]+\
    [Path(__file__).resolve().parents[2]/'rescene'/n for n in ['models/encoders.py','codec/vae_latent_delta.py']]  # frozen SigLIP/VAE wrappers


class LossModule(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,batch):return self.model.compute_loss(batch)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def state_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        t=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(tuple(t.shape)).encode());h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def micro_batches(batch,n):
    """Split a collated batch into n equal consecutive micro-batches (gradient accumulation)."""
    size=len(batch['metadata'])
    if size%n:raise ValueError('batch size must be divisible by --grad-accum')
    k=size//n
    return [{group:(values[i*k:(i+1)*k] if group=='metadata' else {key:value[i*k:(i+1)*k] for key,value in values.items()})
             for group,values in batch.items()} for i in range(n)]


def learning_rate(step,total,peak=5e-5,warmup=50,schedule='cosine'):
    """Linear warmup, then cosine decay to 0 at `total` or a constant `peak` (RDT's own
    pretrain/finetune scripts use constant; the run can then be stopped at any milestone)."""
    if step<warmup:return peak*(step+1)/warmup
    if schedule=='constant':return peak
    return peak/2*(1+math.cos(math.pi*min(1,(step-warmup)/(total-warmup))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--scratch',action='store_true');source.add_argument('--resume',type=Path)
    p.add_argument('--steps',type=int,default=100000);p.add_argument('--smoke-steps',type=int,default=0)
    p.add_argument('--memory-cap-gib',type=float,default=14)
    p.add_argument('--batch-per-gpu',type=int,default=8)
    p.add_argument('--workers-per-gpu',type=int,default=8)
    p.add_argument('--grad-accum',type=int,default=1,
                   help='Split each per-GPU batch into this many micro-batches and accumulate gradients (same update, less memory)')
    p.add_argument('--reshard-resume',action='store_true',
                   help='Resume with a different GPU count / per-GPU batch / grad accumulation / memory cap (same global batch, so every step\n'
                        'still sees the same global sample set); the change is recorded in config["reshard_history"]')
    p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--view-cache',type=Path,help='Exact canonical-view cache built by build_view_cache (fast storage)')
    p.add_argument('--latent-cache',type=Path,help='Precomputed fp32 SD-VAE latents built by build_latent_cache (needs --view-cache)')
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32',
                   help='bf16: generator forward/backward under bf16 autocast and a bf16 SigLIP tower; weights/optimizer/EMA/loss fp32')
    p.add_argument('--warmup',type=int,default=50,help='Linear warmup steps before the main schedule')
    p.add_argument('--mlp-ratio',type=float,default=4.0,help='RDT block FFN width / hidden size (RDT original 1; DiT 4)')
    p.add_argument('--rdt-depth',type=int,default=14,help='Number of RDT blocks (pilot config 14)')
    p.add_argument('--lr-schedule',choices=['constant','cosine'],default='constant',
                   help='constant: fixed lr after warmup (stop at any milestone); cosine: decay to 0 at --steps')
    p.add_argument('--tf32',action=argparse.BooleanOptionalAction,default=True,
                   help='Allow TF32 matmul/conv where the GPU supports it (Ampere+); plain fp32 elsewhere')
    p.add_argument('--ema',action=argparse.BooleanOptionalAction,default=True,help='Keep an EMA copy of the generator')
    p.add_argument('--ema-power',type=float,default=0.75,help='EMA warmup power (models/ema_model.py)')
    p.add_argument('--ema-max-decay',type=float,default=0.9999)
    p.add_argument('--val-every',type=int,default=1000,help='Validation-loss interval in steps; 0 disables')
    p.add_argument('--val-samples',type=int,default=1024)
    p.add_argument('--val-timesteps',type=int,default=10)
    p.add_argument('--val-seed',type=int,default=0)
    p.add_argument('--min-contexts',type=int,default=1,help='Fewest context views per training sample (max 5)')
    p.add_argument('--cond-drop-prob',type=float,default=0.1,help='Drop all context images+cameras (CFG); target direction kept')
    p.add_argument('--image-only-drop-prob',type=float,default=0.,
                   help='class text only: extra share of samples with context images+cameras dropped but text kept')
    p.add_argument('--text-only-drop-prob',type=float,default=0.,
                   help='class text only: extra share of samples with text dropped but context images kept')
    p.add_argument('--panel-cfg-scale',type=float,default=3.0,help='CFG scale for sample panels (1 when no dropout)')
    p.add_argument('--cross-attn',choices=['alternate','image'],default='alternate',
                   help="alternate: RDT's language/image alternation (null language token); image: all blocks see images")
    p.add_argument('--context-pos-embed',choices=['slotless_2d','slot_1d'],default='slotless_2d')
    p.add_argument('--target-pos-embed',choices=['2d_sincos','1d'],default='2d_sincos')
    p.add_argument('--loss-weighting',choices=['min_snr','none'],default='min_snr')
    p.add_argument('--min-snr-gamma',type=float,default=5.0)
    p.add_argument('--text-cond',choices=['null','class'],default='null',
                   help="null: one learned null language token; class: class-prompt T5-XXL embeddings (needs --text-bank)")
    p.add_argument('--text-bank',type=Path,help='Precomputed class-prompt embeddings from plan_b.mvimgnet.class_text')
    p.add_argument('--cond-injection',choices=['adaln_single','token'],default='adaln_single',
                   help="adaln_single: timestep + target direction modulate every block (PixArt-alpha); token: RDT's prepended tokens")
    p.add_argument('--context-latents',choices=['joint','none'],default='joint',
                   help='joint: clean SD-VAE latents of the context views join self-attention (SigLIP cross-attention kept)')
    args=p.parse_args()
    if args.steps<=args.warmup or args.warmup<1:raise ValueError('Total schedule must exceed warmup')
    rdt_arch={'hidden_size':_pilot_rdt_config()['rdt']['hidden_size'],'depth':args.rdt_depth,
              'num_heads':_pilot_rdt_config()['rdt']['num_heads'],'mlp_ratio':args.mlp_ratio}
    if args.latent_cache and not args.view_cache:raise ValueError('--latent-cache is keyed to a view cache; pass --view-cache')
    if (args.text_cond=='class')!=(args.text_bank is not None):raise ValueError('--text-bank is required exactly when --text-cond class')
    text_bank=TextBank(args.text_bank) if args.text_bank else None
    # Read the bank now: the scratch construction below forbids torch.load (which TextBank uses).
    max_text_tokens=text_bank.meta['max_tokens'] if text_bank else 32
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    # Variable context counts give variable SigLIP batch shapes; without expandable segments the
    # caching allocator fragments (measured peak reserved 20.7 -> 10.9 GiB, same speed).
    # Must be set before the first CUDA allocation; an explicit user setting wins.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    per_gpu=args.batch_per_gpu;global_batch=per_gpu*world
    assert per_gpu>=1 and args.workers_per_gpu>=1 and args.lr>0 and args.grad_accum>=1 and per_gpu%args.grad_accum==0
    torch.set_num_threads(2);torch.cuda.set_device(rank)
    # Permission only: TF32 is used where the hardware has it, otherwise ordinary fp32.
    torch.backends.cuda.matmul.allow_tf32=args.tf32;torch.backends.cudnn.allow_tf32=args.tf32
    tf32_hardware=torch.cuda.get_device_capability(rank)[0]>=8
    assert 0<args.memory_cap_gib<=torch.cuda.get_device_properties(rank).total_memory/1024**3
    torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib*1024**3/torch.cuda.get_device_properties(rank).total_memory,rank)
    device=torch.device('cuda',rank)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30))
    readiness=json.loads((args.data_root/'readiness.json').read_text())
    assert readiness['state']=='ready'
    assert readiness['all_masks_audited'] and readiness['all_archives_crc_verified']
    preparation=json.loads((args.data_root/'preparation_config.json').read_text())
    assert preparation['instance_limit'] is None
    # Any number of verified archives is accepted (archives may be deleted after
    # verified extraction); resume is pinned to this exact catalog via readiness.json.
    sources={name:digest(Path(__file__).with_name(name)) for name in SOURCE_NAMES}
    # Enabled before the DataLoader forks its workers; identical pixels, so not a config change.
    cache_info=view_cache.enable(args.view_cache,args.data_root) if args.view_cache else None
    # Precomputed latents are fp32 without TF32, so not bitwise the on-the-fly values: recorded in config.
    latent_info=latent_cache.enable(args.latent_cache,args.view_cache,FEATURE_MODELS['vae_id']) if args.latent_cache else None
    identity=digest(args.data_root/'readiness.json')
    output=args.output
    if rank==0:
        output.mkdir(parents=True,exist_ok=bool(args.resume))
        (output/'checkpoints').mkdir(exist_ok=True);(output/'samples').mkdir(exist_ok=True)
    dist.barrier()
    if (output/'STOP').exists():raise RuntimeError('STOP request still present; inspect before resuming')
    torch.manual_seed(0)
    def construct():
        return DirectionBin15RDT(rdt_config=rdt_config_for({'rdt_arch':rdt_arch}),img_token_dim=1152,lang_token_dim=4096,
                                tokens_per_image=64,k_max=5,dtype=torch.float32,cross_attn=args.cross_attn,
                                context_pos_embed=args.context_pos_embed,target_pos_embed=args.target_pos_embed,
                                cond_drop_prob=args.cond_drop_prob,loss_weighting=args.loss_weighting,
                                min_snr_gamma=args.min_snr_gamma,text_cond=args.text_cond,
                                max_text_tokens=max_text_tokens,
                                cond_injection=args.cond_injection,context_latents=args.context_latents,
                                image_only_drop_prob=args.image_only_drop_prob,text_only_drop_prob=args.text_only_drop_prob).to(device)
    if args.scratch:
        # Fail immediately if any generator initialization path tries to load weights.
        with patch.object(torch,'load',side_effect=RuntimeError('Scratch generator must not load checkpoint')), \
             patch.object(torch.nn.Module,'load_state_dict',side_effect=RuntimeError('Scratch generator must not import weights')):
            model=construct()
        checkpoint=None
    else:
        model=construct()
        checkpoint=torch.load(str(args.resume),map_location='cpu',weights_only=True,mmap=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0)
    config={'preprocessing_policy':PREPROCESS_POLICY,'conditioning_policy':CONDITIONING_POLICY,
        'conditioning_angle_source':ANGLE_SOURCE,'selection_bin_deg':SELECTION_BIN_DEG,
        'camera_feature_dim':4,'camera_fields':['sin_az','cos_az','sin_el','cos_el'],
        'sources':sources,'data_identity':identity,'data_root':str(args.data_root),
        'dataset_counts':readiness['instances_by_split'],'classes':readiness['classes'],'archive_count':readiness['archive_count'],
        'world_size':world,'batch_per_gpu':per_gpu,'global_batch':global_batch,'grad_accum':args.grad_accum,'dtype':'fp32','precision':args.precision,
        'vae_latents':'latent_cache_fp32_no_tf32' if args.latent_cache else 'on_the_fly','workers_per_gpu':args.workers_per_gpu,
        'memory_cap_gib_per_rank':args.memory_cap_gib,'steps':args.steps,'lr':args.lr,'warmup':args.warmup,
        'tf32_allowed':args.tf32,'tf32_hardware':tf32_hardware,'cuda_alloc_conf':os.environ['PYTORCH_CUDA_ALLOC_CONF'],
        'ema':{'power':args.ema_power,'max_decay':args.ema_max_decay,'inv_gamma':1.0,'update_after_step':0,
               'source_sha256':digest(EMA_SOURCE)} if args.ema else None,
        'val':{'every':args.val_every,'samples':args.val_samples,'timesteps':args.val_timesteps,'seed':args.val_seed}
              if args.val_every else None,
        'sampling_policy':SAMPLING_POLICY,'object_seed':0,'prediction_type':'sample',
        'latest_every':1000,'milestone_every':5000,'sample_seed':1234,'sample_every':1000,
        'fill':.8,'background_rgb':[128,128,128],'context_encoding':'present_slots_only',
        'loss':'x0 latent MSE'+(f' x min(SNR,{args.min_snr_gamma})' if args.loss_weighting=='min_snr' else ' (unweighted)'),
        'loss_weighting':args.loss_weighting,'min_snr_gamma':args.min_snr_gamma,
        'min_contexts':args.min_contexts,'max_contexts':5,
        'cond_drop_prob':args.cond_drop_prob,'cond_drop_targets':'context_images+context_cameras (target direction kept)',
        'panel_cfg_scale':args.panel_cfg_scale if args.cond_drop_prob>0 else 1.0,
        'cross_attn':args.cross_attn,'context_pos_embed':args.context_pos_embed,'target_pos_embed':args.target_pos_embed,
        'text_cond':args.text_cond,'rdt_arch':rdt_arch,
        'cond_injection':args.cond_injection,'context_latents':args.context_latents,
        'text_bank':{'path':str(args.text_bank),'sha256':digest(args.text_bank),
                     **{k:text_bank.meta[k] for k in ['template_version','template','normalization','typo_fixes','name_overrides','max_tokens','encoder']}}
                    if text_bank else None,
        'model_sources':{str(f.relative_to(EMA_SOURCE.parents[1])):digest(f) for f in MODEL_SOURCES},
        'lr_schedule':args.lr_schedule,
        'schedule':f'scratch, {args.warmup} warmup then '+('constant lr' if args.lr_schedule=='constant' else f'cosine to step {args.steps}')+', fresh optimizer',
        'generator_initialization':'scratch_seed0_no_checkpoint', 'feature_models':FEATURE_MODELS,
        'invalid_view_policy':INVALID_VIEW_POLICY}
    if args.image_only_drop_prob or args.text_only_drop_prob:
        config['cond_drop_split']={'all':args.cond_drop_prob,'image_only':args.image_only_drop_prob,
            'text_only':args.text_only_drop_prob,'none':1-args.cond_drop_prob-args.image_only_drop_prob-args.text_only_drop_prob,
            'cfg_unconditional':'all dropped (images, cameras, text)'}
    if args.resume:
        # Topology keys may change only with --reshard-resume and an unchanged global batch. This file's own hash is
        # exempt then too, since the option was added to it after runs had started (nothing else in it changed).
        reshard={'world_size','batch_per_gpu','grad_accum','memory_cap_gib_per_rank'} if args.reshard_resume else set()
        if args.reshard_resume:
            assert checkpoint['config']['global_batch']==global_batch,'Reshard must keep the global batch'
        changed={}
        for key,value in config.items():
            saved=checkpoint['config'][key]
            if key=='sources' and args.reshard_resume:
                diff={n for n in value if saved.get(n)!=value[n]}|(set(saved)-set(value))
                assert diff<={'train_direction_continuous_ddp.py'}, 'Resume configuration mismatch: sources '+str(sorted(diff))
                if diff:changed[key]={'train_direction_continuous_ddp.py':[saved.get('train_direction_continuous_ddp.py'),value.get('train_direction_continuous_ddp.py')]}
            elif key in reshard and saved!=value:changed[key]=[saved,value]
            else:assert saved==value, 'Resume configuration mismatch: '+key
        # keys recorded only when an option is used must not silently disappear on resume
        assert ('cond_drop_split' in checkpoint['config'])==('cond_drop_split' in config),'Resume configuration mismatch: cond_drop_split'
        model.load_state_dict(checkpoint['model'],strict=True);optimizer.load_state_dict(checkpoint['optimizer'])
        config=checkpoint['config'];start=int(checkpoint['step']);saved_rng=checkpoint['rng_by_rank']
        if changed:
            for key in reshard:config[key]={'world_size':world,'batch_per_gpu':per_gpu,'grad_accum':args.grad_accum,
                                            'memory_cap_gib_per_rank':args.memory_cap_gib}[key]
            if 'sources' in changed:config['sources']=dict(config['sources'],**{n:v[1] for n,v in changed['sources'].items()})
            config['reshard_history']=config.get('reshard_history',[])+[{'step':start,'changes':changed}]
            # Ranks that did not exist before start from fresh seeds; existing ranks keep their RNG streams.
            saved_rng=saved_rng[:world]+[None]*(world-len(saved_rng))
        if args.ema:assert 'ema' in checkpoint,'Checkpoint has no EMA state'
    else:
        initial_hash=state_digest(model.state_dict())
        config.update(initial_checkpoint=None,base_checkpoint_step=0,initial_checkpoint_sha256=None,
            generator_initial_state_sha256=initial_hash,optimizer_initial_state_entries=len(optimizer.state),
            generator_weight_loading_blocked_during_construction=True,
            generator_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            frozen_pretrained_components=['SigLIP','VAE'])
        assert not optimizer.state
        start=0;saved_rng=None
    ema=ema_model=None
    if args.ema:
        # A separate copy: EMAModel overwrites the module it is given.
        ema_model=copy.deepcopy(model)
        ema=EMAModel(ema_model,update_after_step=0,inv_gamma=1.0,power=args.ema_power,max_value=args.ema_max_decay)
        if args.resume:
            ema_model.load_state_dict(checkpoint['ema'],strict=True);ema.optimization_step=int(checkpoint['ema_step'])
    del checkpoint
    wrapped=DDP(LossModule(model),device_ids=[rank],find_unused_parameters=True)
    encoder=build_image_encoder('siglip',device=device,siglip_model=FEATURE_MODELS['siglip_model'],tokens_per_image=64,dim=1152).eval()
    apply_precision(model,encoder,args.precision)
    if ema_model is not None:ema_model.precision=args.precision
    vae=load_vae(FEATURE_MODELS['vae_id'],device=device).eval()
    # Frozen-encoder features for the fixed val set; built before the RNG state is set
    # and consumes no torch RNG, so training randomness is unaffected.
    val=ValLoss(args.data_root/'catalog.sqlite',samples=args.val_samples,timesteps=args.val_timesteps,seed=args.val_seed,
                encoder=encoder,vae=vae,device=device,rank=rank,world=world,k_min=args.min_contexts,
                text_bank=text_bank,context_latents=args.context_latents=='joint') if args.val_every else None
    if saved_rng and saved_rng[rank] is not None:
        torch.set_rng_state(saved_rng[rank]['cpu']);torch.cuda.set_rng_state(saved_rng[rank]['cuda'],device)
    else:torch.manual_seed(1234+rank)
    stop_step=min(args.steps,start+args.smoke_steps) if args.smoke_steps else args.steps
    if start>=stop_step:raise ValueError('Checkpoint has reached the requested final step')
    dataset=DirectionFullBin15Dataset(args.data_root/'catalog.sqlite','train',seed=0,samples_per_epoch=args.steps*global_batch,
                                      k_min=args.min_contexts,text_bank=text_bank)
    loader=DataLoader(dataset,batch_size=per_gpu,sampler=RankSamples(start,stop_step,per_gpu,rank,world),num_workers=args.workers_per_gpu,
        pin_memory=True,prefetch_factor=2,collate_fn=mvimgnet_bin15_collate,generator=torch.Generator().manual_seed(20261004+rank))
    config['objects_per_epoch']=len(dataset.rows)
    panels={}
    if rank==0:
        for split in ['train','val']:
            selected=DirectionFullBin15Dataset(args.data_root/'catalog.sqlite',split,deterministic=True,seed=0,text_bank=text_bank)
            panels[split]=mvimgnet_bin15_collate([selected[0]])
        # Keep evaluation input choices identical to the baseline extra panels.
        candidates=[]
        for index in sorted(set(int(i*(len(selected.rows)-1)/511) for i in range(512))):
            row,_,tid,_=selected._choose(index);view=next(v for v in row['views'] if v['image_id']==tid)
            candidates.append((view['foreground_fraction'],row['class_id'],index,row['instance_uid']))
        candidates.sort();used=set()
        for number in range(min(8,len(candidates))):
            pivot=int((number+.5)*len(candidates)/8)
            options=sorted(candidates,key=lambda v:abs(candidates.index(v)-pivot))
            choice=next((v for v in options if v[1] not in used),options[0]);used.add(choice[1])
            panels[f'val_q{number}_{choice[1]}']=mvimgnet_bin15_collate([selected[choice[2]]])
        dump(output/'fixed_evaluation_inputs.json',{name:batch['metadata'] for name,batch in panels.items()})
        dump(output/'config_used.json',config)
        if cache_info:dump(output/'view_cache_used.json',cache_info)
        if latent_info:dump(output/'latent_cache_used.json',latent_info)
    # Rank0's data-only fixed panel setup must not change the model RNG stream.
    dist.barrier();model.train();started=time.time();stop_signal=[False]
    def request_stop(*_):stop_signal[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)

    def save(step):
        rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device).cpu()}
        gathered=[None]*world;dist.all_gather_object(gathered,rng)
        if rank==0:
            state={'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,'rng_by_rank':gathered}
            if ema:state.update(ema=ema_model.state_dict(),ema_step=ema.optimization_step)
            targets=[output/'checkpoints/latest.pt']
            if step%5000==0 or step==args.steps:targets.append(output/'checkpoints'/f'step_{step:06d}.pt')
            for target in targets:
                temporary=target.with_suffix('.tmp');torch.save(state,temporary);os.replace(temporary,target)
        dist.barrier()

    def sample(step):
        if rank==0:
            with torch.random.fork_rng(devices=[rank]):
                for name,batch in panels.items():
                    if '_q' in name and step%5000:continue
                    save_sample_panel(model,encoder,vae,batch,device,output/'samples'/f'step_{step:06d}_{name}.png',seed=1234,
                                      cfg_scale=config['panel_cfg_scale'])
                    if ema:
                        save_sample_panel(ema_model,encoder,vae,batch,device,output/'samples'/f'step_{step:06d}_{name}_ema.png',seed=1234,
                                          cfg_scale=config['panel_cfg_scale'])
                        ema_model.eval()
        dist.barrier()

    def validate(step):
        result={'step':step}
        result['raw']=val.evaluate(model)
        if ema:result['ema']=val.evaluate(ema_model)
        if rank==0:
            with (output/'val_loss.jsonl').open('a') as log:log.write(json.dumps(result)+'\n')
            print(json.dumps({'step':step,'val_loss':result['raw']['val_loss'],
                              'val_loss_ema':result['ema']['val_loss'] if ema else None}),flush=True)
        return result

    # Save baseline examples for the new input distribution before optimizing.
    if start==0:
        save(0)
        sample(0)
        if val:validate(0)
    elif config.get('conditioning_transition_step')==start:
        # Preserve a same-weight baseline under the new conditioning before updates.
        sample(start)
    stopped=False;iterator=iter(loader)
    for step in range(start+1,stop_step+1):
        data_error=None
        try:batch=next(iterator)
        except Exception as error:
            batch=None;data_error=f'{type(error).__name__}: {error}'
        loaded=torch.tensor(int(data_error is None),device=device)
        dist.all_reduce(loaded,op=dist.ReduceOp.MIN)
        if not loaded.item():
            errors=[None]*world;dist.all_gather_object(errors,data_error)
            save(step-1)
            if rank==0:dump(output/'training_status.json',{'state':'failed_data','step':step-1,
                'next_step':step,'errors_by_rank':errors,'checkpoint_saved':True,'camera_feature_dim':4})
            raise RuntimeError('Data loading failed; last completed update saved: '+repr(errors))
        fallback_examples=[]
        for meta in batch['metadata']:
            if meta.get('rejected_views') or meta.get('view_replacements'):
                with (output/f'rejected_views_rank{rank}.jsonl').open('a') as log:
                    log.write(json.dumps({'step':step,'instance_uid':meta['instance_uid'],
                        'policy':INVALID_VIEW_POLICY,'rejected':meta['rejected_views'],
                        'replacements':meta['view_replacements']})+'\n')
            for frame,transform in zip(meta['context_frame_ids']+[meta['target_frame_id']],
                                       meta['context_transforms']+[meta['target_transform']]):
                if transform and transform.get('distortion_status','').startswith('identity_fallback'):
                    fallback_examples.append({'instance_uid':meta['instance_uid'],'frame':frame,
                        'source_k':transform['source_distortion_k']})
        if fallback_examples:
            with (output/f'preprocess_fallback_rank{rank}.jsonl').open('a') as log:
                log.write(json.dumps({'step':step,'count':len(fallback_examples),'views':fallback_examples})+'\n')
        lr=learning_rate(step-1,args.steps,args.lr,args.warmup,args.lr_schedule)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        # Equal-size micro-batches: the mean of their mean losses is the full-batch mean loss.
        # Gradients are all-reduced only on the last micro-batch (DDP no_sync).
        chunks=micro_batches(batch,args.grad_accum);loss=0.
        for index,chunk in enumerate(chunks):
            with wrapped.no_sync() if index<len(chunks)-1 else contextlib.nullcontext():
                part=wrapped(make_train_batch(chunk,encoder,vae,device,context_latents=args.context_latents=='joint'))
                if len(chunks)>1:part=part/len(chunks)
                part.backward()
            loss=loss+part.detach()
        finite=torch.isfinite(loss).to(torch.int32);dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item():raise RuntimeError('Nonfinite loss')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if ema:ema.step(model)
        stop=torch.tensor(int(stop_signal[0] or (rank==0 and (output/'STOP').exists())),device=device)
        dist.all_reduce(stop,op=dist.ReduceOp.MAX);stopped=bool(stop.item())
        if step%25==0 or step==start+1 or step==stop_step or stopped:
            average=loss.detach().clone();dist.all_reduce(average)
            if rank==0:
                record={'state':'training','step':step,'target_step':args.steps,'base_checkpoint_step':config['base_checkpoint_step'],
                    'loss':float(average/world),'lr':lr,'elapsed_seconds':time.time()-started,'full_data_samples':step*global_batch,
                    'object_epochs':step*global_batch/len(dataset.rows),'camera_feature_dim':4,
                    'cuda_peak_reserved_gib_rank0':torch.cuda.max_memory_reserved(device)/1024**3}
                dump(output/'training_status.json',record);print(json.dumps(record),flush=True)
        if step%1000==0 or step==stop_step or stopped:
            save(step)
            if not stopped:sample(step)
        if val and (step%args.val_every==0 or step==stop_step or stopped):validate(step)
        if stopped:break
    if rank==0:
        dump(output/'training_status.json',{'state':'stopped' if stopped else ('smoke_completed' if args.smoke_steps else 'completed'),
            'step':step,'target_step':args.steps,'base_checkpoint_step':config['base_checkpoint_step'],'camera_feature_dim':4})
    dist.destroy_process_group()


if __name__=='__main__':main()
