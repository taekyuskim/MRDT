"""Novel-view evaluation: PSNR, SSIM, LPIPS (AlexNet, VGG), FID, KID.

Protocol (README.md, "Evaluation"):
  * pairs: one per object of --split (or the first --max-objects of a fixed permutation);
    target view and context views drawn with the training sampler rules under a fixed seed
    (target bin never shared with a context bin); the target is the same for every k;
  * k = number of context views, evaluated separately for each value in --contexts;
  * generation: EMA weights (default), DPM-Solver++ --steps, fixed --sample-seed, CFG scale;
  * metrics on 256x256 RGB in [0,1] against the identically preprocessed ground truth,
    on the full image and on the ground-truth foreground (mask > 0.5);
  * FID/KID compare generated vs ground-truth target images of the same pairs.
Choose generation settings (e.g. CFG scale) on --split val; report --split test.

    torchrun --nproc_per_node=4 -m plan_b.mvimgnet.eval_nvs --checkpoint ... \
        --catalog <catalog dir> --view-cache /var/tmp/<user>/mrdt_cache --split val --max-objects 1000 --out <dir>
"""
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import models  # noqa: F401  (bind bundled packages first)
import rescene  # noqa: F401
from .bin15_camera import occupied_pair_bins
from .bin15_dataset import mvimgnet_bin15_collate
from .bin15_model import patch_tokens_to_latent
from .canonical_object import MissingCanonicalForeground
from .continuous_direction_dataset import ContinuousDirectionDataset
from .direction_bin15_model import DirectionBin15RDT
from .train_bin15_pilot import _pilot_rdt_config, rdt_config_for, model_options_for, encode_context_images, encode_context_latents, apply_precision
from . import view_cache
from .class_text import TextBank


class FixedContextDataset(ContinuousDirectionDataset):
    """Training sampler rules with exactly k context views (k distinct non-target bins)."""

    def __init__(self, catalog, split, *, k, seed, text_bank=None):
        super().__init__(catalog, split, seed=seed, k_min=k, text_bank=text_bank)
        self.k_fixed = k

    def _choose(self, index):
        row = self.rows[self.object_order[index]]
        rng = self._rng(index)
        fixed_rows = self.fixed[row['instance_uid']]
        bins = occupied_pair_bins(fixed_rows[int(i)] for i in row['valid_image_ids'])
        usable = {b: ids for b, ids in bins.items() if ids}
        if len(usable) < self.k_fixed + 1:
            raise MissingCanonicalForeground(f'only {len(usable)} occupied bins')
        target_bin = rng.choice(list(usable))  # drawn first: same target for every k
        target_id = rng.choice(usable[target_bin])
        context_bins = [b for b in usable if b != target_bin]
        context_ids = [rng.choice(usable[b]) for b in rng.sample(context_bins, self.k_fixed)]
        return row, context_ids, target_id, target_bin

    def __getitem__(self, index):
        try:
            item = super().__getitem__(index)
        except MissingCanonicalForeground as error:  # too few bins / no usable foreground
            return {'skipped': index, 'reason': str(error)}
        item['metadata']['eval_index'] = index
        return item


def collate(items):
    kept = [i for i in items if 'skipped' not in i]
    return (mvimgnet_bin15_collate(kept) if kept else None), [i for i in items if 'skipped' in i]


def masked_mean(values, mask):
    """values (B,C,H,W) or (B,1,H,W); mask (B,1,H,W) bool -> per-image mean over masked pixels."""
    m = mask.expand_as(values).to(values.dtype)
    return (values * m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp(min=1)


def psnr(mse):
    return 10 * torch.log10(1.0 / mse.clamp(min=1e-10))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--catalog', type=Path, required=True, help='Catalog dir (catalog.sqlite, readiness.json)')
    p.add_argument('--view-cache', type=Path)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--contexts', type=int, nargs='+', default=[1, 3, 5])
    p.add_argument('--max-objects', type=int, help='Default: every object of the split')
    p.add_argument('--pair-seed', type=int, default=2026)
    p.add_argument('--sample-seed', type=int, default=1234)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--cfg-scale', type=float, help='Default: 3 if trained with condition dropout, else 1')
    p.add_argument('--weights', choices=['auto', 'ema', 'raw'], default='auto')
    p.add_argument('--precision', choices=['fp32', 'bf16'], help='Default: same as the training run')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--memory-cap-gib', type=float,
                   help='PyTorch allocator cap per GPU (e.g. when sharing a GPU with a training run)')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--kid-subset', type=int, default=1000)
    p.add_argument('--text-bank', type=Path, help="Class-prompt embeddings (default: the training run's path)")
    p.add_argument('--siglip-model')
    p.add_argument('--vae-model')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()

    distributed = 'WORLD_SIZE' in os.environ
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(rank)
    if args.memory_cap_gib:
        torch.cuda.set_per_process_memory_fraction(
            args.memory_cap_gib * 1024**3 / torch.cuda.get_device_properties(rank).total_memory, rank)
    device = torch.device('cuda', rank)
    if distributed:
        dist.init_process_group('nccl')

    checkpoint = torch.load(str(args.checkpoint), map_location='cpu', weights_only=True, mmap=True)
    config = checkpoint['config']
    tf32 = bool(config.get('tf32_allowed', False))
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    weights = ('ema' if 'ema' in checkpoint else 'raw') if args.weights == 'auto' else args.weights
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else (3.0 if config.get('cond_drop_prob', 0) > 0 else 1.0)
    model = DirectionBin15RDT(rdt_config=rdt_config_for(config), img_token_dim=1152, lang_token_dim=4096,
                              tokens_per_image=64, k_max=5, dtype=torch.float32, **model_options_for(config),
                              max_text_tokens=(config.get('text_bank') or {}).get('max_tokens', 32)).to(device).eval()
    text_bank = None
    if config.get('text_cond') == 'class':
        text_bank = TextBank(args.text_bank or config['text_bank']['path'])
        if text_bank.meta['template_version'] != config['text_bank']['template_version']:
            raise ValueError('Text bank does not match the training run')
    model.load_state_dict(checkpoint['ema' if weights == 'ema' else 'model'], strict=True)
    step = int(checkpoint['step'])
    del checkpoint

    from rescene.models.encoders import build_image_encoder
    from rescene.codec.vae_latent_delta import load_vae
    import lpips
    from torchmetrics.functional.image import structural_similarity_index_measure
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    feature = config['feature_models']
    encoder = build_image_encoder('siglip', device=device, siglip_model=args.siglip_model or feature['siglip_model'],
                                  tokens_per_image=64, dim=1152).eval()
    vae = load_vae(args.vae_model or feature['vae_id'], device=device).eval()
    precision = args.precision or config.get('precision', 'fp32')
    apply_precision(model, encoder, precision)
    lpips_nets = {name: lpips.LPIPS(net=name, spatial=True, verbose=False).to(device).eval() for name in ['alex', 'vgg']}
    if args.view_cache:
        view_cache.enable(args.view_cache, args.catalog)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)

    summary = {'checkpoint': str(args.checkpoint), 'checkpoint_step': step, 'weights': weights, 'cfg_scale': cfg_scale,
               'steps': args.steps, 'split': args.split, 'pair_seed': args.pair_seed, 'sample_seed': args.sample_seed,
               'tf32_allowed': tf32, 'precision': precision, 'lpips': 'spatial maps; alex (primary) and vgg', 'ssim': 'gaussian 11x11, sigma 1.5, data_range 1',
               'foreground': 'ground-truth target mask > 0.5', 'results': {}}
    for k in args.contexts:
        dataset = FixedContextDataset(args.catalog / 'catalog.sqlite', args.split, k=k, seed=args.pair_seed, text_bank=text_bank)
        n = min(len(dataset.rows), args.max_objects or len(dataset.rows))
        indices = list(range(rank, n, world))
        loader = DataLoader(indices, batch_size=args.batch, num_workers=args.workers, collate_fn=lambda ids: collate([dataset[i] for i in ids]))
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        kid = KernelInceptionDistance(subset_size=min(args.kid_subset, n), normalize=True).to(device)
        keys = ['psnr', 'psnr_fg', 'ssim', 'ssim_fg', 'lpips_alex', 'lpips_alex_fg', 'lpips_vgg', 'lpips_vgg_fg']
        sums = torch.zeros(len(keys), dtype=torch.float64, device=device)
        count = torch.zeros((), dtype=torch.float64, device=device)
        skipped = []
        rows_out = []
        started = time.time()
        for batch, skip in loader:
            skipped += skip
            if batch is None:
                continue
            mi = batch['model_inputs']
            with torch.no_grad():
                inputs = {key: mi[key].to(device) for key in ['context_camera_features', 'target_camera_features', 'context_present',
                                                               'text_embeds', 'text_mask'] if torch.is_tensor(mi.get(key))}
                # Frozen encoders in chunks of --batch images keep the peak low (eval may share a GPU with training).
                inputs['slot_feats'] = encode_context_images(batch, encoder, device, chunk=args.batch)
                if model.context_latents == 'joint':
                    inputs['context_latents'] = encode_context_latents(batch, vae, device, chunk=args.batch)
                # One fixed initial noise per evaluation pair, so results do not depend on batch size or GPU count.
                noise = torch.stack([torch.randn((64, 64), generator=torch.Generator().manual_seed(
                    args.sample_seed * 1_000_003 + int(meta['eval_index']))) for meta in batch['metadata']])
                tokens = model.predict(inputs, num_steps=args.steps, cfg_scale=cfg_scale, initial_noise=noise)
                gen = vae.decode(patch_tokens_to_latent(tokens.float())).clamp(0, 1)
                gt = batch['supervision']['target_image'].to(device).clamp(0, 1)
                fg = batch['supervision']['target_foreground_mask'].to(device) > 0.5
                se = (gen - gt) ** 2
                ssim_full, ssim_map = structural_similarity_index_measure(
                    gen, gt, data_range=1.0, gaussian_kernel=True, sigma=1.5, kernel_size=11,
                    reduction='none', return_full_image=True)
                values = {'psnr': psnr(se.flatten(1).mean(1)), 'psnr_fg': psnr(masked_mean(se, fg)),
                          'ssim': ssim_full, 'ssim_fg': masked_mean(ssim_map, fg)}
                for name, net in lpips_nets.items():
                    lp = net(gen, gt, normalize=True)  # (B,1,H,W), inputs in [0,1]
                    values[f'lpips_{name}'] = lp.flatten(1).mean(1)
                    values[f'lpips_{name}_fg'] = masked_mean(lp, fg)
                fid.update(gt, real=True); fid.update(gen, real=False)
                kid.update(gt, real=True); kid.update(gen, real=False)
            stacked = torch.stack([values[key].double() for key in keys])  # (K, B)
            sums += stacked.sum(1)
            count += stacked.shape[1]
            for j, meta in enumerate(batch['metadata']):
                rows_out.append({'eval_index': meta['eval_index'], 'instance_uid': meta['instance_uid'],
                                 'class_id': meta['class_id'], 'target_image_id': meta['target_image_id'],
                                 'context_image_ids': ' '.join(str(i) for i in meta['context_image_ids'] if i is not None),
                                 **{key: float(stacked[i, j]) for i, key in enumerate(keys)}})
        if distributed:
            dist.all_reduce(sums)
            dist.all_reduce(count)
        fid_value = float(fid.compute())  # torchmetrics syncs states across ranks
        kid.subset_size = min(kid.subset_size, int(count.item()))
        torch.manual_seed(args.sample_seed)  # KID draws random subsets
        kid_mean, kid_std = (float(v) for v in kid.compute())
        skipped_all = [None] * world if distributed else [skipped]
        if distributed:
            dist.all_gather_object(skipped_all, skipped)
        with (args.out / f'per_image_k{k}_rank{rank}.csv').open('w', newline='') as f:
            if rows_out:
                writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
                writer.writeheader()
                writer.writerows(rows_out)
        if rank == 0:
            result = {key: float(sums[i] / count) for i, key in enumerate(keys)}
            result.update(fid=fid_value, kid_mean=kid_mean, kid_std=kid_std, pairs=int(count.item()),
                          objects_considered=n, skipped=sum(len(s) for s in skipped_all),
                          skip_examples=[s for part in skipped_all for s in part][:20], seconds=time.time() - started)
            summary['results'][f'k{k}'] = result
            print(json.dumps({'k': k, **{key: round(v, 4) if isinstance(v, float) else v
                                       for key, v in result.items() if key != 'skip_examples'}}), flush=True)
    if rank == 0:
        summary['cuda_peak_reserved_gib_rank0'] = torch.cuda.max_memory_reserved(device) / 1024**3
        summary['cuda_peak_allocated_gib_rank0'] = torch.cuda.max_memory_allocated(device) / 1024**3
        (args.out / 'metrics.json').write_text(json.dumps(summary, indent=2))
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
