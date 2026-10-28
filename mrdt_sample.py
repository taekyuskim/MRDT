"""Generate a target view from catalog context images and continuous angles."""
import argparse
import json
from pathlib import Path


def usable_contexts(row, ids, basis, *, repair, parser):
    """Drop/replace contexts whose mask yields no canonical foreground.

    Mirrors the training repair (same object, same 15-degree bin first, never a bin
    another context already uses). Explicit --context-ids are never replaced.
    """
    from plan_b.mvimgnet.canonical_object import MissingCanonicalForeground
    from plan_b.mvimgnet.roll_direction_dataset import tensor_view
    views = {view['image_id']: view for view in row['views']}
    camera_bin = lambda i: (views[i]['az_bin'], views[i]['el_bin'])
    candidates = sorted(int(i) for i in row['valid_image_ids'])
    rejected, selected, replacements = set(), [], []

    def loads(image_id):
        try:
            tensor_view(views[image_id], basis)
            return True
        except MissingCanonicalForeground as error:
            if not repair:
                parser.error(f'context image {image_id} has no usable foreground ({error}); choose another ID')
            rejected.add(image_id)
            return False

    for slot, original in enumerate(ids):
        forbidden = {camera_bin(i) for i in selected} | {camera_bin(i) for i in ids[slot + 1:]}
        ordered = [original] + sorted((i for i in candidates if i != original),
                                      key=lambda i: (camera_bin(i) != camera_bin(original), i))
        choice = next((i for i in ordered if i not in rejected and i not in selected and
                       (i == original or camera_bin(i) not in forbidden) and loads(i)), None)
        if choice is None:
            replacements.append({'slot': slot, 'from_image_id': original, 'to_image_id': None})
            continue
        if choice != original:
            replacements.append({'slot': slot, 'from_image_id': original, 'to_image_id': choice})
        selected.append(choice)
    if not selected:
        parser.error('No context view of this object has a usable foreground')
    return selected, replacements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='Required unless --prepare-only')
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--context-ids', type=int, nargs='+', help='Optional 1–5 catalog image IDs')
    parser.add_argument('--azimuth', type=float, required=True, help='Degrees in the object reference basis')
    parser.add_argument('--elevation', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--siglip-model')
    parser.add_argument('--vae-model')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cfg-scale', type=float, default=None,
                        help='Classifier-free guidance scale; default 3 if the model was trained with condition dropout, else 1')
    parser.add_argument('--text-bank', type=Path, help="Class-prompt embeddings (default: the training run's path)")
    parser.add_argument('--weights', choices=['auto', 'ema', 'raw'], default='auto',
                        help='auto: EMA weights when the checkpoint has them')
    parser.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=None,
                        help='Default: same as the checkpoint training run')
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default=None,
                        help='Default: same as the checkpoint training run')
    parser.add_argument('--view-cache', type=Path, help='Canonical-view cache dir from build_view_cache (optional)')
    parser.add_argument('--prepare-only', action='store_true', help='Save context inputs without loading weights or using GPU')
    args = parser.parse_args()
    if not args.prepare_only and args.checkpoint is None:
        parser.error('--checkpoint is required for generation')
    if args.steps < 1 or args.index < 0:
        parser.error('--steps must be positive and --index nonnegative')
    import models
    import rescene
    import torch
    from plan_b.mvimgnet.continuous_direction_dataset import (
        ContinuousDirectionDataset, context_inputs, CONDITIONING_POLICY, PREPROCESS_POLICY)
    from plan_b.mvimgnet.roll_canonical_object import reference_basis
    from plan_b.mvimgnet.direction_bin15_model import DirectionBin15RDT
    from plan_b.mvimgnet.train_bin15_pilot import _pilot_rdt_config, rdt_config_for, model_options_for, _tensor_to_pil, encode_context_images, encode_context_latents, apply_precision
    from plan_b.mvimgnet.bin15_model import patch_tokens_to_latent
    from rescene.models.encoders import build_image_encoder
    from rescene.codec.vae_latent_delta import load_vae

    if args.view_cache:
        from plan_b.mvimgnet import view_cache
        view_cache.enable(args.view_cache, args.catalog.parent)
    dataset = ContinuousDirectionDataset(args.catalog, args.split, deterministic=True, seed=0)
    if args.index >= len(dataset.rows):
        parser.error('--index exceeds the selected split')
    if args.context_ids:
        row = dataset.rows[args.index]
        ids = args.context_ids
    else:
        row, ids, _, _ = dataset._choose(args.index)
    if len(ids) != len(set(ids)):
        parser.error('--context-ids must be distinct')
    views = {view['image_id']: view for view in row['views']}
    if any(i not in views for i in ids):
        parser.error('A context image ID is absent from this object')
    from plan_b.mvimgnet import view_cache
    basis = view_cache.basis_for(row)
    ids, replacements = usable_contexts(row, ids, basis, repair=not args.context_ids, parser=parser)
    # Only the selected contexts are opened; target RGB/mask are never loaded.
    mi, transforms = context_inputs([views[i] for i in ids], args.azimuth, args.elevation, basis=basis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for slot in range(len(ids)):
        _tensor_to_pil(mi['context_images'][slot]).save(args.output.with_name(args.output.stem + f'_context{slot}.png'))
    metadata = {'instance_uid': row['instance_uid'], 'context_image_ids': ids,
                'context_angles_deg': [[views[i]['azimuth_deg'], views[i]['elevation_deg']] for i in ids],
                'target_angles_deg': [args.azimuth, args.elevation],
                'conditioning_policy': CONDITIONING_POLICY, 'preprocessing_policy': PREPROCESS_POLICY,
                'seed': args.seed, 'steps': args.steps, 'prepare_only': args.prepare_only,
                'context_transforms': transforms, 'context_replacements': replacements}
    if not args.prepare_only:
        checkpoint = torch.load(str(args.checkpoint), map_location='cpu', weights_only=True, mmap=True)
        config = checkpoint['config']
        if config.get('conditioning_policy') != CONDITIONING_POLICY or config.get('preprocessing_policy') != PREPROCESS_POLICY:
            raise ValueError('Checkpoint is not the current continuous-direction/roll-aligned model')
        if config.get('camera_feature_dim') != 4 or config.get('prediction_type') != 'sample':
            raise ValueError('Checkpoint has an incompatible camera/output contract')
        feature = config['feature_models']
        device = torch.device(args.device)
        # Architecture options come from the training run (older checkpoints: original model).
        model = DirectionBin15RDT(rdt_config=rdt_config_for(config), img_token_dim=1152,
            lang_token_dim=4096, tokens_per_image=64, k_max=5, dtype=torch.float32, **model_options_for(config),
            max_text_tokens=(config.get('text_bank') or {}).get('max_tokens', 32)).to(device).eval()
        cfg_scale = args.cfg_scale if args.cfg_scale is not None else (3.0 if config.get('cond_drop_prob', 0) > 0 else 1.0)
        if cfg_scale != 1.0 and not config.get('cond_drop_prob', 0) > 0:
            parser.error('This checkpoint was trained without condition dropout; CFG is not meaningful')
        weights = ('ema' if 'ema' in checkpoint else 'raw') if args.weights == 'auto' else args.weights
        if weights == 'ema' and 'ema' not in checkpoint:
            parser.error('This checkpoint has no EMA weights')
        model.load_state_dict(checkpoint[weights if weights == 'ema' else 'model'], strict=True)
        tf32 = config.get('tf32_allowed', False) if args.tf32 is None else args.tf32
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        metadata.update(checkpoint_step=int(checkpoint['step']), weights=weights, tf32_allowed=tf32, cfg_scale=cfg_scale)
        del checkpoint
        encoder = build_image_encoder('siglip', device=device,
            siglip_model=args.siglip_model or feature['siglip_model'], tokens_per_image=64, dim=1152).eval()
        vae = load_vae(args.vae_model or feature['vae_id'], device=device).eval()
        precision = args.precision or config.get('precision', 'fp32')
        apply_precision(model, encoder, precision)
        metadata['precision'] = precision
        with torch.no_grad():
            batch = {key: mi[key].unsqueeze(0).to(device) for key in
                     ['context_camera_features', 'target_camera_features', 'context_present']}
            if config.get('text_cond') == 'class':  # class prompt of this catalog object
                from plan_b.mvimgnet.class_text import TextBank
                bank = TextBank(args.text_bank or config['text_bank']['path'])
                if bank.meta['template_version'] != config['text_bank']['template_version']:
                    raise ValueError('Text bank does not match the training run')
                embeds, mask = bank.lookup(row['class_id'])
                batch.update(text_embeds=embeds[None].to(device), text_mask=mask[None].to(device))
                metadata['text_prompt'] = bank.meta['classes'][str(row['class_id'])]['prompt']
            contexts = {'model_inputs': {key: mi[key].unsqueeze(0) for key in ['context_images', 'context_present']}}
            batch['slot_feats'] = encode_context_images(contexts, encoder, device)
            if model.context_latents == 'joint':
                batch['context_latents'] = encode_context_latents(contexts, vae, device)
            tokens = model.predict(batch, num_steps=args.steps, seed=args.seed, cfg_scale=cfg_scale)
            generated = vae.decode(patch_tokens_to_latent(tokens.float()))[0]
            _tensor_to_pil(generated).save(args.output)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'prepared_contexts': len(ids), 'generated': not args.prepare_only, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
