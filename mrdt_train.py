"""Run the unchanged continuous-direction trainer with explicit encoder paths."""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=
        'Other trainer options are passed through unchanged: --warmup, --tf32/--no-tf32, --ema/--no-ema, '
        '--ema-power, --ema-max-decay, --val-every, --val-samples, --val-timesteps, --val-seed, '
        '--min-contexts, --cond-drop-prob, --panel-cfg-scale, --cross-attn, --context-pos-embed, '
        '--target-pos-embed, --loss-weighting, --min-snr-gamma, --lr-schedule, --text-cond, --text-bank, '
        '--cond-injection, --context-latents, --grad-accum, --latent-cache, --precision '
        '(see python -m plan_b.mvimgnet.train_direction_continuous_ddp --help).')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--scratch', action='store_true')
    source.add_argument('--resume')
    parser.add_argument('--siglip-model', help='Local SigLIP directory; default: $MRDT_MODEL_ROOT/siglip-so400m-patch14-384 (MRDT_MODEL_ROOT defaults to <repo>/pretrained)')
    parser.add_argument('--vae-model', help='Local VAE directory; default: $MRDT_MODEL_ROOT/sd-vae-ft-mse')
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--memory-cap-gib', type=float, default=14)
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument('--batch-per-gpu', type=int, default=8, help='Default 8 x 4 GPUs = global batch 32')
    parser.add_argument('--workers-per-gpu', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5, help='Peak learning rate')
    parser.add_argument('--view-cache', help='Canonical-view cache dir from build_view_cache (optional)')
    args, passthrough = parser.parse_known_args()
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # see trainer
    # Bind bundled packages before legacy modules add their original server path.
    import models
    import rescene
    from plan_b.mvimgnet import train_direction_continuous_ddp as trainer
    if args.siglip_model:
        trainer.FEATURE_MODELS['siglip_model'] = args.siglip_model
    if args.vae_model:
        trainer.FEATURE_MODELS['vae_id'] = args.vae_model
    sys.argv = [sys.argv[0], '--data-root', args.data_root, '--output', args.output,
                '--steps', str(args.steps), '--memory-cap-gib', str(args.memory_cap_gib),
                '--smoke-steps', str(args.smoke_steps), '--batch-per-gpu', str(args.batch_per_gpu),
                '--workers-per-gpu', str(args.workers_per_gpu), '--lr', repr(args.lr)]
    sys.argv += ['--scratch'] if args.scratch else ['--resume', args.resume]
    sys.argv += ['--view-cache', args.view_cache] if args.view_cache else []
    sys.argv += passthrough
    trainer.main()


if __name__ == '__main__':
    main()
