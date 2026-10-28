"""Shared construction/checkpoint logic for ReSCENE training and sampling."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

from rescene.codec import build_codec
from rescene.models.encoders import build_image_encoder, build_text_encoder
from rescene.models.rescene_rdt import ReSCENERDT

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}




def _rank0_print(*args, **kwargs) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs)


def _extract_tensor_state_dict(obj):
    """Return the tensor state dict from common checkpoint payload layouts."""
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state", "model_state_dict", "model", "module"):
            nested = obj.get(key)
            if isinstance(nested, dict):
                obj = nested
                break
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(obj)!r}")
    return {k: v for k, v in obj.items() if torch.is_tensor(v)}


def _load_one_weight_file(path: Path):
    """Load one .bin/.pt/.safetensors file on CPU."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _load_checkpoint_state_dict(ckpt_path: str):
    """Load local HF/PyTorch state dicts, including simple sharded repos."""
    path = Path(ckpt_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"RDT checkpoint path does not exist: {ckpt_path}")

    if path.is_file():
        return _extract_tensor_state_dict(_load_one_weight_file(path))

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if index_path.exists():
            index = json.loads(index_path.read_text())
            shards = sorted(set(index.get("weight_map", {}).values()))
            if not shards:
                raise RuntimeError(f"No shard files listed in {index_path}")
            state = {}
            for shard in shards:
                state.update(_extract_tensor_state_dict(_load_one_weight_file(path / shard)))
            return state

    candidates = []
    for name in ("model.safetensors", "pytorch_model.bin", "model.bin", "ckpt.pt", "checkpoint.pt"):
        p = path / name
        if p.exists():
            candidates.append(p)
    candidates.extend(sorted(path.glob("*.safetensors")))
    candidates.extend(sorted(path.glob("*.bin")))
    candidates.extend(sorted(path.glob("*.pt")))

    seen = set()
    candidates = [p for p in candidates if p.is_file() and not (p in seen or seen.add(p))]
    if not candidates:
        raise FileNotFoundError(
            f"No RDT weight file found under {ckpt_path}. Expected model.safetensors, "
            "pytorch_model.bin, sharded index json, *.bin, *.pt, or *.safetensors."
        )
    return _extract_tensor_state_dict(_load_one_weight_file(candidates[0]))


def _candidate_pretrained_keys(src_key: str):
    """Generate plausible RDTRunner-local key names for official/wrapped checkpoints."""
    keys = []
    k = src_key
    for prefix in ("module.", "_orig_mod."):
        if k.startswith(prefix):
            k = k[len(prefix):]
    keys.append(k)

    prefix_strips = (
        "model.runner.",
        "runner.",
        "policy.runner.",
        "policy.",
        "rdt.",
    )
    for prefix in prefix_strips:
        if k.startswith(prefix):
            keys.append(k[len(prefix):])

    # Some checkpoints may store the bare RDT module without the RDTRunner's "model." prefix.
    bare_tops = ("x_pos_embed", "lang_cond_pos_embed", "img_cond_pos_embed",
                 "t_embedder", "freq_embedder", "blocks", "final_layer")
    if k.split(".", 1)[0] in bare_tops:
        keys.append("model." + k)

    out, seen = [], set()
    for kk in keys:
        if kk not in seen:
            out.append(kk)
            seen.add(kk)
    return out


def load_rdt_pretrained_partial(runner: torch.nn.Module, ckpt_path: str) -> dict:
    """Load shape-compatible tensors from an official RDT/RDTRunner checkpoint.

    ReSCENE changes robot action prediction into edit-action prediction. That
    changes action_dim, state token dimension, image condition length, and
    position-embedding lengths. strict=False alone still errors on same-name
    shape mismatches, so we copy only same-name/same-shape tensors and report
    what was skipped.
    """
    state = _load_checkpoint_state_dict(ckpt_path)
    own = runner.state_dict()

    matched = {}
    skipped_shape = []
    skipped_name = []

    for src_key, value in state.items():
        loaded_or_shape_checked = False
        for dst_key in _candidate_pretrained_keys(src_key):
            if dst_key not in own:
                continue
            loaded_or_shape_checked = True
            if tuple(own[dst_key].shape) == tuple(value.shape):
                matched[dst_key] = value
            else:
                skipped_shape.append((dst_key, tuple(value.shape), tuple(own[dst_key].shape)))
            break
        if not loaded_or_shape_checked:
            skipped_name.append(src_key)

    incompatible = runner.load_state_dict(matched, strict=False)
    n_model = sum(1 for k in matched if k.startswith("model."))
    n_blocks = sum(1 for k in matched if k.startswith("model.blocks."))
    n_adaptors = sum(1 for k in matched if "adaptor" in k)

    _rank0_print(f"[rdt] pretrained path: {ckpt_path}")
    _rank0_print(
        f"[rdt] loaded tensors: {len(matched)} / current {len(own)} "
        f"(model={n_model}, blocks={n_blocks}, adaptors={n_adaptors})"
    )
    _rank0_print(f"[rdt] skipped by shape: {len(skipped_shape)} | skipped by name: {len(skipped_name)}")
    for name, src_shape, dst_shape in skipped_shape[:30]:
        _rank0_print(f"[rdt] shape mismatch: {name} ckpt={src_shape} current={dst_shape}")
    _rank0_print(f"[rdt] missing after partial load: {len(incompatible.missing_keys)}")
    _rank0_print(f"[rdt] unexpected after partial load: {len(incompatible.unexpected_keys)}")

    if len(matched) == 0:
        raise RuntimeError(f"No tensors loaded from RDT checkpoint: {ckpt_path}")
    if n_blocks == 0:
        _rank0_print("[rdt][WARN] No transformer block tensors loaded. Check model size/config.")

    return {
        "source_tensors": len(state),
        "current_tensors": len(own),
        "loaded_tensors": len(matched),
        "loaded_model_tensors": n_model,
        "loaded_block_tensors": n_blocks,
        "loaded_adaptor_tensors": n_adaptors,
        "skipped_shape": len(skipped_shape),
        "skipped_name": len(skipped_name),
    }


@torch.no_grad()
def configure_action_positional_embedding(
    runner: torch.nn.Module,
    spec,
    mode: str = "auto",
) -> Optional[dict]:
    """Configure positional embeddings for ReSCENE action tokens.

    The official RDT uses one-dimensional temporal action positions.  A
    ``patch2d`` codec instead represents an ``H_tokens x W_tokens`` spatial
    grid flattened in raster order.  In that case this function preserves the
    first three RDT positions (diffusion timestep, control frequency and
    metadata state) and replaces only the action-token positions with a fixed
    2-D sine/cosine initialization.  The full parameter remains trainable.

    ``mode='auto'`` selects ``2d_sincos`` for patch tokens and keeps the
    official initialization for legacy row tokens.
    """
    if mode is None:
        mode = "auto"
    mode = str(mode).lower()
    if mode == "auto":
        mode = "2d_sincos" if spec.is_spatial_patch_sequence else "pretrained_1d"
    if mode in {"none", "pretrained", "pretrained_1d", "1d"}:
        return None
    if mode not in {"2d_sincos", "sincos2d"}:
        raise ValueError(
            f"Unknown action positional embedding mode {mode!r}; expected "
            "'auto', 'pretrained_1d', or '2d_sincos'."
        )
    if not spec.is_spatial_patch_sequence:
        raise ValueError(
            "2d_sincos action positions require codec tokenization='patch2d'."
        )

    from models.rdt.blocks import get_nd_sincos_pos_embed_from_grid

    token_h, token_w = spec.token_grid_shape
    model = runner.model
    expected_len = 3 + token_h * token_w
    if model.x_pos_embed.shape[1] != expected_len:
        raise ValueError(
            f"RDT x_pos_embed length={model.x_pos_embed.shape[1]} but patch grid "
            f"{token_h}x{token_w} requires {expected_len} positions "
            "(3 special + spatial action tokens)."
        )

    # Official RDT multimodal positions reserve the first half for modality
    # identity and the second half for within-modality position. Preserve the
    # pretrained action-modality half and replace only its 1-D temporal half
    # with a factorized 2-D spatial embedding.
    spatial_dim = model.hidden_size // 2
    spatial_np = get_nd_sincos_pos_embed_from_grid(
        spatial_dim, (token_h, token_w)
    )
    spatial = torch.from_numpy(spatial_np).reshape(
        1, token_h * token_w, spatial_dim
    )
    spatial = spatial.to(
        device=model.x_pos_embed.device,
        dtype=model.x_pos_embed.dtype,
    )
    model.x_pos_embed.data[:, 3:, -spatial_dim:].copy_(spatial)
    _rank0_print(
        f"[rdt] action positional embedding: 2d_sincos "
        f"grid={token_h}x{token_w} tokens={token_h * token_w} "
        f"spatial_dim={spatial_dim} (pretrained modality half preserved)"
    )
    return {
        "mode": "2d_sincos",
        "token_grid_h": token_h,
        "token_grid_w": token_w,
        "num_action_tokens": token_h * token_w,
        "spatial_dim": spatial_dim,
    }


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_pipeline(cfg: dict, device: str = "cpu", build_model: bool = True,
                   load_pretrained_rdt: bool = True) -> Dict:
    """Build codec + frozen encoders (+ trainable ReSCENE model) from a config."""
    enc_cfg = cfg["encoders"]
    data_cfg = cfg["data"]

    codec_cfg = dict(cfg["codec"])
    codec_cfg.setdefault("image_size", data_cfg["image_size"])
    if codec_cfg["image_size"] != data_cfg["image_size"]:
        raise ValueError(
            f"codec.image_size ({codec_cfg['image_size']}) must equal "
            f"data.image_size ({data_cfg['image_size']})")
    codec = build_codec(codec_cfg, device=device)

    text_encoder = build_text_encoder(
        enc_cfg["text"], device=device, t5_model=enc_cfg.get("t5_model"),
        max_length=enc_cfg.get("max_lang_cond_len", 120),
        dim=enc_cfg.get("lang_token_dim", 64))
    image_encoder = build_image_encoder(
        enc_cfg["image"], device=device, siglip_model=enc_cfg.get("siglip_model"),
        tokens_per_image=enc_cfg["tokens_per_image"],
        dim=enc_cfg.get("img_token_dim", 64))
    if text_encoder.dim != enc_cfg["lang_token_dim"]:
        raise ValueError(f"encoders.lang_token_dim={enc_cfg['lang_token_dim']} but the "
                         f"{enc_cfg['text']} text encoder outputs dim={text_encoder.dim}")
    if image_encoder.dim != enc_cfg["img_token_dim"]:
        raise ValueError(f"encoders.img_token_dim={enc_cfg['img_token_dim']} but the "
                         f"{enc_cfg['image']} image encoder outputs dim={image_encoder.dim}")

    out = {"codec": codec, "text_encoder": text_encoder,
           "image_encoder": image_encoder, "cfg": cfg}
    if build_model:
        model = ReSCENERDT(
            rdt_config=cfg["model"],
            spec=codec.spec,
            lang_token_dim=enc_cfg["lang_token_dim"],
            img_token_dim=enc_cfg["img_token_dim"],
            tokens_per_image=enc_cfg["tokens_per_image"],
            max_normal_views=data_cfg.get("max_normal_views", 5),
            max_support=data_cfg.get("max_support", 2),
            max_lang_cond_len=enc_cfg.get("max_lang_cond_len", 120),
            dtype=DTYPES[cfg.get("dtype", "fp32")],
            dropout=cfg.get("dropout"),
            local_condition_dim=int(getattr(codec, "local_condition_dim", 0) or 0),
        )
        pretrained_rdt = cfg.get("model", {}).get("pretrained_rdt") or cfg.get("pretrained_rdt")
        if load_pretrained_rdt and pretrained_rdt:
            report = load_rdt_pretrained_partial(model.runner, pretrained_rdt)
            out["pretrained_rdt_report"] = report

        pos_mode = (
            cfg.get("model", {}).get("action_pos_embed")
            or codec_cfg.get("spatial_pos_embed")
            or "auto"
        )
        pos_report = configure_action_positional_embedding(
            model.runner, codec.spec, mode=pos_mode
        )
        if pos_report is not None:
            out["action_pos_embed_report"] = pos_report

        model = model.to(device)
        out["model"] = model
    return out


# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_batch_conditions(batch: Dict, text_encoder, image_encoder, device,
                            use_text_cache: bool = False) -> Dict:
    """Turn a collated dataset batch into the tensor dict ReSCENERDT expects
    (everything except ``action_gt``)."""
    B, S = batch["slot_images"].shape[:2]
    flat = batch["slot_images"].flatten(0, 1).to(device)          # (B*S, 3, H, W)
    feats = image_encoder.encode(flat)                            # (B*S, P, D)
    slot_feats = feats.reshape(B, S, feats.shape[1], feats.shape[2])

    if use_text_cache and all("cached_text_embedding" in ep for ep in batch["episode"]):
        embs = [torch.load(ep["cached_text_embedding"], map_location=device,
                           weights_only=False)["emb"].float()
                for ep in batch["episode"]]
        L = max(e.shape[0] for e in embs)
        lang_emb = torch.zeros(B, L, embs[0].shape[1], device=device)
        lang_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b, e in enumerate(embs):
            lang_emb[b, : e.shape[0]] = e
            lang_mask[b, : e.shape[0]] = True
    else:
        lang_emb, lang_mask = text_encoder.encode(batch["instruction"])
        lang_emb, lang_mask = lang_emb.to(device), lang_mask.to(device)

    return {
        "slot_feats": slot_feats.float(),
        "slot_present": batch["slot_present"].to(device),
        "slot_view_ids": batch["slot_view_ids"].to(device),
        "dataset_id": batch["dataset_id"].to(device),
        "target_view_id": batch["target_view_id"].to(device),
        "lang_emb": lang_emb,
        "lang_mask": lang_mask,
    }


@torch.no_grad()
def compute_norm_stats(codec, loader, device, num_batches: int = 8):
    """Estimate per-grid-channel mean/std of the (un-normalized) edit-action grid
    and install them into the codec."""
    codec._norm_mean = None  # make sure we measure raw values
    codec._norm_std = None
    sums = None
    sq = None
    count = 0
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        chunk = codec.encode_target(batch["target_normal"].to(device),
                                    batch["target_anomaly"].to(device),
                                    batch["target_mask"].to(device))
        grid = codec.chunk_to_grid(chunk)          # (B, C, H, W)
        flat = grid.transpose(0, 1).reshape(grid.shape[1], -1)  # (C, B*H*W)
        if sums is None:
            sums = flat.sum(dim=1)
            sq = (flat ** 2).sum(dim=1)
        else:
            sums += flat.sum(dim=1)
            sq += (flat ** 2).sum(dim=1)
        count += flat.shape[1]
    if count == 0:
        raise RuntimeError("norm-stat loader yielded no batches")
    mean = sums / count
    var = (sq / count - mean ** 2).clamp_min(1e-8)
    std = var.sqrt()
    # keep the mask channel un-shifted: it is already a well-scaled SDF in [-1,1]
    m = codec.spec.mask_channels
    mean[:m] = 0.0
    std[:m] = 1.0
    codec.set_normalization(mean.cpu(), std.cpu())
    return codec.normalization_state()


# ---------------------------------------------------------------------------
def save_checkpoint(path, model: ReSCENERDT, cfg: dict, norm_state, step: int,
                    optimizer=None, extra: Optional[dict] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": cfg,
        "model_state": model.state_dict(),
        "norm_state": norm_state,
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, device="cpu", build_encoders: bool = True) -> Dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    pipe = build_pipeline(cfg, device=device, build_model=True,
                          load_pretrained_rdt=False) if build_encoders \
        else {"cfg": cfg}
    if build_encoders:
        pipe["model"].load_state_dict(payload["model_state"])
        pipe["model"].eval()
        pipe["codec"].load_normalization_state(payload.get("norm_state"))
    pipe["step"] = payload.get("step", 0)
    pipe["payload"] = payload
    return pipe
