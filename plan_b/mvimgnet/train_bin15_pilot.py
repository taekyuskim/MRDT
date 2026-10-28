"""Train MVImgNet2.0 Stage-1 bin15 camera-conditioned pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from rescene.codec.vae_latent_delta import load_vae
from rescene.models.encoders import build_image_encoder

from .bin15_dataset import MVImgNetBin15Stage1Dataset, mvimgnet_bin15_collate
from .bin15_model import Bin15CameraRDT, latent_to_patch_tokens, patch_tokens_to_latent


ROOT = Path("/data1/haksoo/data/MVImgNet2.0")
PROCESSED = ROOT / "processed_plan_b"
FOLLOWUP = ROOT / "reports_plan_b/mvimgnet2_audit_followup"


def _tiny_rdt_config(prediction_type: str = "sample") -> dict:
    return {
        "lang_adaptor": "mlp2x_gelu",
        "img_adaptor": "mlp2x_gelu",
        "state_adaptor": "mlp3x_gelu",
        "rdt": {"hidden_size": 64, "depth": 2, "num_heads": 4},
        "noise_scheduler": {
            "num_train_timesteps": 16,
            "num_inference_timesteps": 4,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": prediction_type,
            "clip_sample": False,
        },
    }


def _pilot_rdt_config() -> dict:
    return {
        "lang_adaptor": "mlp2x_gelu",
        "img_adaptor": "mlp2x_gelu",
        "state_adaptor": "mlp3x_gelu",
        "rdt": {"hidden_size": 1024, "depth": 14, "num_heads": 32},
        "noise_scheduler": {
            "num_train_timesteps": 1000,
            "num_inference_timesteps": 20,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "sample",
            "clip_sample": False,
        },
    }


def rdt_config_for(run_config: dict | None = None) -> dict:
    """The pilot RDT config with the architecture options recorded in a run config
    (older runs: original depth 14, mlp ratio 1)."""
    cfg = _pilot_rdt_config()
    arch = (run_config or {}).get("rdt_arch") or {}
    cfg["rdt"]["depth"] = int(arch.get("depth", cfg["rdt"]["depth"]))
    cfg["rdt"]["mlp_ratio"] = float(arch.get("mlp_ratio", 1.0))
    return cfg


def model_options_for(run_config: dict | None = None) -> dict:
    """Bin15CameraRDT keyword options recorded in a run config (older runs: original defaults)."""
    c = run_config or {}
    return dict(cross_attn=c.get("cross_attn", "alternate"),
                context_pos_embed=c.get("context_pos_embed", "slot_1d"),
                target_pos_embed=c.get("target_pos_embed", "1d"),
                cond_drop_prob=c.get("cond_drop_prob", 0.0),
                loss_weighting=c.get("loss_weighting", "none"),
                min_snr_gamma=c.get("min_snr_gamma", 5.0),
                text_cond=c.get("text_cond", "null"),
                cond_injection=c.get("cond_injection", "token"),
                context_latents=c.get("context_latents", "none"))


def apply_precision(model, image_encoder, precision: str) -> None:
    """bf16: the generator's network calls and the frozen SigLIP tower run under bf16 autocast;
    all weights, the optimizer, EMA and the loss stay fp32. fp32 changes nothing."""
    if precision not in ("fp32", "bf16"):
        raise ValueError(f"Unsupported precision={precision!r}")
    model.precision = precision
    if hasattr(image_encoder, "set_precision"):
        image_encoder.set_precision(precision)
    elif precision != "fp32":
        raise ValueError(f"{type(image_encoder).__name__} has no bf16 mode")


def _to_device_batch(batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    mi = batch["model_inputs"]
    out = {
        "context_camera_features": mi["context_camera_features"].to(device),
        "target_camera_features": mi["target_camera_features"].to(device),
        "context_present": mi["context_present"].to(device),
    }
    if torch.is_tensor(mi.get("text_embeds")):  # class-text conditioning (class_text.TextBank)
        out["text_embeds"] = mi["text_embeds"].to(device)
        out["text_mask"] = mi["text_mask"].to(device)
    return out


@torch.no_grad()
def encode_context_images(batch: Dict, image_encoder, device: torch.device, chunk: int = 32) -> torch.Tensor:
    # Encode present slots only; the model replaces absent-slot features with
    # its missing-image token, so their (zero) features are never used.
    imgs = batch["model_inputs"]["context_images"]
    b, k, c, h, w = imgs.shape
    present = batch["model_inputs"]["context_present"].bool().reshape(b * k).cpu()
    if not present.any():
        raise ValueError("every sample needs at least one present context slot")
    flat = imgs.reshape(b * k, c, h, w)[present]
    # Chunked: SigLIP's eager attention materialises (n, 16, 729, 729) scores; 80 images at
    # once need ~6.7 GiB transient vs ~2.7 GiB in chunks of 32 (measured, ~4% slower).
    feats = torch.cat([image_encoder.encode(flat[i:i + chunk].to(device)).float() for i in range(0, flat.shape[0], chunk)])
    out = feats.new_zeros(b * k, feats.shape[1], feats.shape[2])
    out[present.to(device)] = feats
    return out.reshape(b, k, feats.shape[1], feats.shape[2])


@torch.no_grad()
def encode_target_latents(batch: Dict, vae, device: torch.device) -> torch.Tensor:
    target = batch["supervision"]["target_image"].to(device)
    z = vae.encode(target)
    return latent_to_patch_tokens(z).float()


@torch.no_grad()
def encode_context_latents(batch: Dict, vae, device: torch.device, chunk: int = 32) -> torch.Tensor:
    """SD-VAE patch tokens of the present context views, (B, K, 64, 64); absent slots zero."""
    imgs = batch["model_inputs"]["context_images"]
    b, k, c, h, w = imgs.shape
    present = batch["model_inputs"]["context_present"].bool().reshape(b * k).cpu()
    flat = imgs.reshape(b * k, c, h, w)[present]
    # Chunked: VAE encoder activations at 256x256 are ~34 MB per image.
    tokens = torch.cat([latent_to_patch_tokens(vae.encode(flat[i:i + chunk].to(device))).float()
                        for i in range(0, flat.shape[0], chunk)])
    out = tokens.new_zeros(b * k, tokens.shape[1], tokens.shape[2])
    out[present.to(device)] = tokens
    return out.reshape(b, k, tokens.shape[1], tokens.shape[2])


def make_train_batch(batch: Dict, image_encoder, vae, device: torch.device, context_latents: bool = False) -> Dict[str, torch.Tensor]:
    out = _to_device_batch(batch, device)
    out["slot_feats"] = encode_context_images(batch, image_encoder, device)
    # Precomputed latents (latent_cache) when the dataset provides them, else encode now.
    cached_target = batch["supervision"].get("target_latent_z")
    out["action_gt"] = (latent_to_patch_tokens(cached_target.to(device)).float() if torch.is_tensor(cached_target)
                        else encode_target_latents(batch, vae, device))
    if context_latents:
        cached_context = batch["model_inputs"].get("context_latent_z")
        if torch.is_tensor(cached_context):
            b, k = cached_context.shape[:2]
            out["context_latents"] = latent_to_patch_tokens(cached_context.to(device).flatten(0, 1)).float().reshape(b, k, 64, 64)
        else:
            out["context_latents"] = encode_context_latents(batch, vae, device)
    forbidden = {"target_image", "target_latent", "clean_latent", "raw_T_rel", "angle_residual"}
    if forbidden.intersection(out):
        raise AssertionError(f"forbidden conditioning keys present: {forbidden.intersection(out)}")
    return out


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = torch.nan_to_num(t.detach().cpu(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).round().astype("uint8"))


def _label(img: Image.Image, text: str, height: int = 52) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + height), "white")
    out.paste(img.convert("RGB"), (0, height))
    draw = ImageDraw.Draw(out)
    draw.text((5, 5), text[:120], fill=(0, 0, 0))
    return out


def _concat_h(imgs: List[Image.Image]) -> Image.Image:
    w = sum(i.width for i in imgs)
    h = max(i.height for i in imgs)
    out = Image.new("RGB", (w, h), "white")
    x = 0
    for im in imgs:
        out.paste(im, (x, 0))
        x += im.width
    return out


@torch.no_grad()
def save_sample_panel(model, image_encoder, vae, batch: Dict, device: torch.device, path: Path, *, seed: int, nested_ks=(None,), cfg_scale: float = 1.0) -> None:
    model.eval()
    panels = []
    for nested_k in nested_ks:
        local = {
            "model_inputs": {k: (v.clone() if torch.is_tensor(v) else list(v)) for k, v in batch["model_inputs"].items()},
            "supervision": batch["supervision"],
            "metadata": batch["metadata"],
        }
        if nested_k is not None:
            present = torch.zeros_like(local["model_inputs"]["context_present"])
            present[:, : int(nested_k)] = batch["model_inputs"]["context_present"][:, : int(nested_k)]
            local["model_inputs"]["context_present"] = present
        train_batch = make_train_batch(local, image_encoder, vae, device,
                                       context_latents=getattr(model, "context_latents", "none") == "joint")
        pred = model.predict(train_batch, num_steps=20, seed=seed, cfg_scale=cfg_scale)
        z_pred = patch_tokens_to_latent(pred.float())
        gen = vae.decode(z_pred).detach().cpu()
        target = local["supervision"]["target_image"].detach().cpu()
        gt_dec = vae.decode(patch_tokens_to_latent(encode_target_latents(local, vae, device))).detach().cpu()
        b0 = 0
        ctx_tiles = []
        for i in range(local["model_inputs"]["context_images"].shape[1]):
            if not bool(local["model_inputs"]["context_present"][b0, i]):
                continue
            desc = local["metadata"][b0]["context_descriptions"][i]
            ctx_tiles.append(_label(_tensor_to_pil(local["model_inputs"]["context_images"][b0, i]), f"C{i+1} {desc}", 42))
        row = ctx_tiles + [
            _label(_tensor_to_pil(target[b0]), "GT target"),
            _label(_tensor_to_pil(gt_dec[b0]), "GT VAE decode"),
            _label(_tensor_to_pil(gen[b0]), f"Generated K={nested_k or 'all'}"),
        ]
        panels.append(_concat_h(row))
    width = max(p.width for p in panels)
    height = sum(p.height for p in panels)
    out = Image.new("RGB", (width, height), "white")
    y = 0
    for p in panels:
        out.paste(p, (0, y))
        y += p.height
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    model.train()


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("step", 0))


def build_scheduler(optimizer, *, steps: int, warmup: int):
    def lr_lambda(step):
        if step < warmup:
            return float(step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run(args) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False if not args.resume else True)
    (out / "checkpoints").mkdir(exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)

    debug = args.debug_components
    rdt_cfg = _tiny_rdt_config() if debug else _pilot_rdt_config()
    img_dim = 64 if debug else 1152
    lang_dim = 64 if debug else 4096
    tokens_per_image = 64
    k_max = args.k_max

    image_encoder = build_image_encoder(
        "debug" if debug else "siglip",
        device=device,
        siglip_model=args.siglip_model,
        tokens_per_image=tokens_per_image,
        dim=img_dim,
    )
    vae = load_vae("debug" if debug else args.vae_id, device=device)
    model = Bin15CameraRDT(
        rdt_config=rdt_cfg,
        img_token_dim=img_dim,
        lang_token_dim=lang_dim,
        tokens_per_image=tokens_per_image,
        k_max=k_max,
        dtype=torch.float32,
    ).to(device)
    pretrained_report = None
    if not debug and not args.skip_pretrained:
        pretrained_report = model.install_pretrained(args.rdt_pretrained)

    train_ds = MVImgNetBin15Stage1Dataset(
        PROCESSED / "manifests/train_instances.jsonl",
        args.fixed_camera_csv,
        "train",
        k_max=k_max,
        samples_per_epoch=max(args.steps * args.batch_size, args.batch_size),
        seed=args.seed,
    )
    val_ds = MVImgNetBin15Stage1Dataset(
        PROCESSED / "manifests/val_instances.jsonl",
        args.fixed_camera_csv,
        "val",
        k_max=k_max,
        samples_per_epoch=max(4, args.batch_size),
        seed=args.seed,
        deterministic=True,
    )
    train_debug_ds = MVImgNetBin15Stage1Dataset(
        PROCESSED / "manifests/train_instances.jsonl",
        args.fixed_camera_csv,
        "train_debug",
        k_max=k_max,
        samples_per_epoch=max(4, args.batch_size),
        seed=args.seed,
        deterministic=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=mvimgnet_bin15_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=mvimgnet_bin15_collate)
    debug_loader = DataLoader(train_debug_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=mvimgnet_bin15_collate)
    val_batch = next(iter(val_loader))
    debug_batch = next(iter(debug_loader))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = build_scheduler(optimizer, steps=args.steps, warmup=args.warmup)
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(Path(args.resume), model, optimizer, scheduler)

    arg_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg_used = {
        "source_root": str(ROOT),
        "processed_root": str(PROCESSED),
        "fixed_camera_csv": str(args.fixed_camera_csv),
        "split": "existing train/val/test instance split",
        "camera_feature": "sin/cos az_bin, sin/cos el_bin, log(radius_refnorm), normalized K256 fx/fy/cx/cy, distortion k",
        "conditioning": "bin15 centers only; no raw T_rel, no angle residual, no exact canonical ray",
        "model": "Bin15CameraRDT; no discrete view embedding",
        "init": "official RDT-170M partial load" if not debug else "debug tiny random init",
        "pretrained_report": pretrained_report,
        "training": arg_dict,
    }
    (out / "config_used.json").write_text(json.dumps(cfg_used, indent=2, sort_keys=True), encoding="utf-8")

    log = open(out / "train.log", "a", encoding="utf-8")
    print(json.dumps({"event": "start", "out": str(out), "start_step": start_step, "debug": debug}), file=log, flush=True)
    model.train()
    step = start_step
    while step < args.steps:
        train_ds.set_epoch(step)
        for batch in train_loader:
            step += 1
            train_batch = make_train_batch(batch, image_encoder, vae, device)
            loss = model.compute_loss(train_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if step % args.log_every == 0 or step == 1:
                msg = {"step": step, "loss": float(loss.detach().cpu()), "lr": scheduler.get_last_lr()[0]}
                print(json.dumps(msg), file=log, flush=True)
                print(json.dumps(msg), flush=True)
            if step % args.sample_every == 0 or (args.self_test and step == 1):
                save_sample_panel(model, image_encoder, vae, debug_batch, device, out / "samples" / f"step_{step:06d}_train_debug.png", seed=args.sample_seed)
                save_sample_panel(model, image_encoder, vae, val_batch, device, out / "samples" / f"step_{step:06d}_val.png", seed=args.sample_seed)
            if step in {500, 1000}:
                save_sample_panel(model, image_encoder, vae, val_batch, device, out / "samples" / f"step_{step:06d}_nested_k.png", seed=args.sample_seed, nested_ks=(2, 3, 5))
            if step in {500, 1000}:
                save_checkpoint(out / "checkpoints" / f"step_{step:06d}.pt", model, optimizer, scheduler, step, cfg_used)
            if args.self_test and step >= 1:
                break
            if step >= args.steps:
                break
        if args.self_test:
            break
    save_checkpoint(out / "checkpoints" / "final.pt", model, optimizer, scheduler, step, cfg_used)
    save_checkpoint(out / "checkpoints" / "latest.pt", model, optimizer, scheduler, step, cfg_used)
    print(json.dumps({"event": "done", "step": step, "out": str(out)}), file=log, flush=True)
    log.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--sample-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample-seed", type=int, default=1234)
    p.add_argument("--k-max", type=int, default=5)
    p.add_argument("--fixed-camera-csv", type=Path, default=FOLLOWUP / "fixed_camera_views_followup.csv")
    p.add_argument("--siglip-model", default="/data1/MFM_models/rescene/ckpt/siglip-so400m-patch14-384")
    p.add_argument("--vae-id", default="/data1/MFM_models/rescene/ckpt/sd-vae-ft-mse")
    p.add_argument("--rdt-pretrained", default="/data1/MFM_models/rescene/ckpt/rdt-170m")
    p.add_argument("--resume", default="")
    p.add_argument("--debug-components", action="store_true")
    p.add_argument("--skip-pretrained", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        args.debug_components = True
        args.skip_pretrained = True
        args.device = "cpu"
        args.steps = min(args.steps, 1)
        args.num_workers = 0
        args.batch_size = min(args.batch_size, 2)
    run(args)


if __name__ == "__main__":
    main()
