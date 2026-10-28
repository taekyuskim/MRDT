"""Bin15 camera-conditioned RDT wrapper for MVImgNet2.0 Stage-1."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import contextlib
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rdt.blocks import get_nd_sincos_pos_embed_from_grid
from models.rdt_runner import RDTRunner
from rescene.codec.base import EditActionSpec
from rescene.pipeline import configure_action_positional_embedding, load_rdt_pretrained_partial


def latent_to_patch_tokens(z: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    if z.ndim != 4 or tuple(z.shape[1:]) != (4, 32, 32):
        raise ValueError(f"expected latent (B,4,32,32), got {tuple(z.shape)}")
    b, c, h, w = z.shape
    p = int(patch_size)
    patches = z.unfold(2, p, p).unfold(3, p, p)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(b, (h // p) * (w // p), c * p * p)


def patch_tokens_to_latent(tokens: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (64, 64):
        raise ValueError(f"expected tokens (B,64,64), got {tuple(tokens.shape)}")
    b = tokens.shape[0]
    p = int(patch_size)
    grid = tokens.view(b, 8, 8, 4, p, p)
    grid = grid.permute(0, 3, 1, 4, 2, 5).contiguous()
    return grid.view(b, 4, 32, 32)


PRECISIONS = {"fp32": None, "bf16": torch.bfloat16}


class Bin15CameraRDT(nn.Module):
    """Use binned camera features without ReSCENE discrete view embeddings."""

    # Compute precision of the network calls (weights stay fp32): "fp32", or "bf16" autocast.
    # A plain attribute, not a constructor option, so checkpoints and init are unaffected.
    precision = "fp32"

    def _autocast(self):
        dtype = PRECISIONS[self.precision]
        return contextlib.nullcontext() if dtype is None else torch.autocast("cuda", dtype=dtype)

    def __init__(
        self,
        *,
        rdt_config: dict,
        img_token_dim: int,
        tokens_per_image: int = 64,
        k_max: int = 5,
        action_dim: int = 64,
        pred_horizon: int = 64,
        lang_token_dim: int = 4096,
        camera_feature_dim: int = 10,
        dtype=torch.float32,
        cross_attn: str = "alternate",
        context_pos_embed: str = "slot_1d",
        target_pos_embed: str = "1d",
        cond_drop_prob: float = 0.0,
        loss_weighting: str = "none",
        min_snr_gamma: float = 5.0,
        text_cond: str = "null",
        max_text_tokens: int = 32,
        cond_injection: str = "token",
        context_latents: str = "none",
        image_only_drop_prob: float = 0.0,
        text_only_drop_prob: float = 0.0,
    ) -> None:
        """Defaults reproduce the original model. Options:
        cross_attn: 'alternate' (RDT: even blocks attend the null language token) or 'image'.
        context_pos_embed: 'slot_1d' (slot index x 1-D token index) or 'slotless_2d' (8x8 per view).
        target_pos_embed: '1d' or '2d_sincos' (initial value of the target-token positions).
        cond_drop_prob: training-time drop of all context images + cameras (for CFG).
        loss_weighting: 'none' or 'min_snr' (weight min(SNR, gamma) on the x0 loss).
        text_cond: 'null' (one learned null language token) or 'class' (class-prompt T5-XXL
            embeddings, up to max_text_tokens, cross-attended by the language blocks).
        cond_injection: 'token' (RDT: timestep/frequency/target-direction tokens prepended to
            the sequence) or 'adaln_single' (PixArt-alpha: timestep + target-direction embedding
            modulate every block; the sequence holds only the target tokens).
        context_latents: 'none' or 'joint' (the clean SD-VAE latents of the present context
            views join the self-attention sequence, 64 tokens per view; SigLIP cross-attention
            is kept).
        image_only_drop_prob / text_only_drop_prob (text_cond='class'): extra, mutually exclusive
            training cases on top of cond_drop_prob (which drops images, cameras and text
            together): drop only the context images/cameras (text kept), or only the text
            (images kept). One uniform draw per sample picks the case.
        """
        super().__init__()
        if cross_attn not in ("alternate", "image"):
            raise ValueError(f"cross_attn must be 'alternate' or 'image', got {cross_attn!r}")
        if context_pos_embed not in ("slot_1d", "slotless_2d"):
            raise ValueError(f"context_pos_embed must be 'slot_1d' or 'slotless_2d', got {context_pos_embed!r}")
        if target_pos_embed not in ("1d", "2d_sincos"):
            raise ValueError(f"target_pos_embed must be '1d' or '2d_sincos', got {target_pos_embed!r}")
        if loss_weighting not in ("none", "min_snr"):
            raise ValueError(f"loss_weighting must be 'none' or 'min_snr', got {loss_weighting!r}")
        if text_cond not in ("null", "class"):
            raise ValueError(f"text_cond must be 'null' or 'class', got {text_cond!r}")
        if text_cond == "class" and cross_attn != "alternate":
            raise ValueError("text_cond='class' needs cross_attn='alternate' (language blocks)")
        if cond_injection not in ("token", "adaln_single"):
            raise ValueError(f"cond_injection must be 'token' or 'adaln_single', got {cond_injection!r}")
        if context_latents not in ("none", "joint"):
            raise ValueError(f"context_latents must be 'none' or 'joint', got {context_latents!r}")
        if cond_injection == "adaln_single" and target_pos_embed != "2d_sincos":
            raise ValueError("adaln_single uses a full-width 2-D sin/cos target position embedding")
        self.adaln = cond_injection == "adaln_single"
        self.context_latents = context_latents
        rdt_config = copy.deepcopy(rdt_config)
        rdt_config["rdt"]["cond_injection"] = cond_injection
        hidden = int(rdt_config["rdt"]["hidden_size"])
        self.text_cond = text_cond
        if not 0.0 <= cond_drop_prob < 1.0 or min_snr_gamma <= 0:
            raise ValueError("cond_drop_prob must be in [0, 1) and min_snr_gamma > 0")
        self.cond_drop_prob = float(cond_drop_prob)
        if image_only_drop_prob < 0 or text_only_drop_prob < 0 or \
                cond_drop_prob + image_only_drop_prob + text_only_drop_prob >= 1.0:
            raise ValueError("drop probabilities must be >= 0 and sum to < 1")
        if (image_only_drop_prob or text_only_drop_prob) and text_cond != "class":
            raise ValueError("image-only / text-only dropout needs text_cond='class'")
        self.image_only_drop_prob = float(image_only_drop_prob)
        self.text_only_drop_prob = float(text_only_drop_prob)
        self.loss_weighting = loss_weighting
        self.min_snr_gamma = float(min_snr_gamma)
        self.k_max = int(k_max)
        self.tokens_per_image = int(tokens_per_image)
        self.action_dim = int(action_dim)
        self.pred_horizon = int(pred_horizon)
        self.camera_feature_dim = int(camera_feature_dim)
        self.runner = RDTRunner(
            action_dim=self.action_dim,
            pred_horizon=self.pred_horizon,
            config=rdt_config,
            lang_token_dim=lang_token_dim,
            img_token_dim=img_token_dim,
            state_token_dim=self.action_dim,
            max_lang_cond_len=1 if text_cond == "null" else int(max_text_tokens),
            img_cond_len=self.k_max * self.tokens_per_image,
            img_pos_embed_config=self._context_pos_config(context_pos_embed, k_max, tokens_per_image),
            # class text: RDT's default 1-D sin/cos over the token sequence
            lang_pos_embed_config=[("null_text", 1)] if text_cond == "null" else None,
            dtype=dtype,
        )
        hidden_img = max(128, img_token_dim // 4)
        self.context_camera_mlp = nn.Sequential(
            nn.Linear(camera_feature_dim, hidden_img),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_img, img_token_dim),
        )
        if self.adaln:  # target direction -> condition vector added to the timestep embedding
            self.target_direction_embed = nn.Sequential(
                nn.Linear(camera_feature_dim, 256),
                nn.SiLU(),
                nn.Linear(256, hidden),
            )
        else:
            self.target_camera_mlp = nn.Sequential(
                nn.Linear(camera_feature_dim, 128),
                nn.GELU(approximate="tanh"),
                nn.Linear(128, self.action_dim),
            )
        if context_latents == "joint":
            self.context_latent_adaptor = self.runner.build_condition_adapter(
                rdt_config["state_adaptor"], in_features=self.action_dim, out_features=hidden)
            self.context_latent_camera_mlp = nn.Sequential(
                nn.Linear(camera_feature_dim, 256),
                nn.SiLU(),
                nn.Linear(256, hidden),
            )
            side = int(round(self.tokens_per_image ** 0.5))
            pos = get_nd_sincos_pos_embed_from_grid(hidden, (side, side)).reshape(side * side, hidden)
            # shared by every context slot (unordered set); trainable, sin/cos initialised
            self.context_latent_pos_embed = nn.Parameter(torch.from_numpy(pos).float().unsqueeze(0))
            self.context_latent_type_embed = nn.Parameter(torch.zeros(1, 1, hidden))
            nn.init.normal_(self.context_latent_type_embed, std=0.02)
        self.null_lang_token = nn.Parameter(torch.zeros(1, 1, lang_token_dim))
        self.missing_image_token = nn.Parameter(torch.zeros(1, 1, 1, img_token_dim))
        nn.init.normal_(self.null_lang_token, std=0.02)
        nn.init.normal_(self.missing_image_token, std=0.02)
        self.runner.model.cross_attn_mode = cross_attn
        if target_pos_embed == "2d_sincos" and not self.adaln:  # adaln: set by the RDT itself
            side = int(round(self.pred_horizon ** 0.5))
            spec = EditActionSpec(grid_channels=4, grid_h=side * 4, grid_w=side * 4, mask_channels=0,
                                  tokenization="patch2d", patch_size=4)
            configure_action_positional_embedding(self.runner, spec, mode="2d_sincos")

    @staticmethod
    def _context_pos_config(mode, k_max, tokens_per_image):
        if mode == "slot_1d":
            return [("image", (int(k_max), int(tokens_per_image)))]
        side = int(round(tokens_per_image ** 0.5))
        if side * side != tokens_per_image:
            raise ValueError("slotless_2d needs a square number of tokens per image")
        # A negative grid size disables that axis (models/rdt/blocks.py), so all
        # slots share one 2-D (row, col) embedding: contexts form an unordered set.
        return [("image", (-int(k_max), side, side))]

    @property
    def prediction_type(self) -> str:
        return self.runner.prediction_type

    def install_pretrained(self, ckpt_path: str) -> dict:
        report = load_rdt_pretrained_partial(self.runner, ckpt_path)
        spec = EditActionSpec(grid_channels=4, grid_h=32, grid_w=32, mask_channels=0, tokenization="patch2d", patch_size=4)
        pos = configure_action_positional_embedding(self.runner, spec, mode="2d_sincos")
        report["action_pos_embed"] = pos
        return report

    def _conditions(
        self,
        *,
        slot_feats: torch.Tensor,
        context_camera_features: torch.Tensor,
        target_camera_features: torch.Tensor,
        context_present: torch.Tensor,
        drop: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        drop_text: Optional[torch.Tensor] = None,
    ):
        """drop (B,) bool: replace every context slot by the null image token (CFG's
        unconditional branch). Those samples attend all null tokens, never an empty mask.
        drop_text (B,) bool: samples whose class text is replaced by the null token;
        None means the same samples as `drop` (images and text dropped together)."""
        b, k, p, d = slot_feats.shape
        if k != self.k_max or p != self.tokens_per_image:
            raise ValueError(f"slot_feats expected (B,{self.k_max},{self.tokens_per_image},D), got {tuple(slot_feats.shape)}")
        present = context_present.bool()
        cam = self.context_camera_mlp(context_camera_features.to(slot_feats.device, slot_feats.dtype))
        feats = slot_feats + cam[:, :, None, :]
        if drop is not None:
            drop = drop.to(device=present.device, dtype=torch.bool)
            present = present & ~drop[:, None]
        feats = torch.where(present[:, :, None, None], feats, self.missing_image_token.to(feats.dtype))
        img_tokens = feats.reshape(b, k * p, d)
        visible = present if drop is None else (present | drop[:, None])
        img_mask = visible[:, :, None].expand(b, k, p).reshape(b, k * p)
        target_dir = target_camera_features.to(slot_feats.device, slot_feats.dtype)
        if self.adaln:  # (B, D) condition vector
            state_tokens = self.target_direction_embed(target_dir)
        else:
            state_tokens = self.target_camera_mlp(target_dir).unsqueeze(1)
        if self.text_cond == "null":
            lang = self.null_lang_token.expand(b, 1, -1).to(slot_feats.device, slot_feats.dtype)
            lang_mask = torch.ones((b, 1), dtype=torch.bool, device=slot_feats.device)
        else:
            if text_embeds is None or text_mask is None:
                raise ValueError("text_cond='class' requires text_embeds and text_mask")
            lang = text_embeds.to(slot_feats.device, slot_feats.dtype)
            lang_mask = text_mask.to(slot_feats.device).bool()
            tdrop = drop if drop_text is None else drop_text.to(device=lang.device, dtype=torch.bool)
            if tdrop is not None:  # unconditional: only the null token (position 0) is visible
                null = torch.zeros_like(lang)
                null[:, 0] = self.null_lang_token[0, 0].to(lang.dtype)
                null_mask = torch.zeros_like(lang_mask)
                null_mask[:, 0] = True
                lang = torch.where(tdrop[:, None, None], null, lang)
                lang_mask = torch.where(tdrop[:, None], null_mask, lang_mask)
        return lang, lang_mask, img_tokens, img_mask, state_tokens

    def _context_latent_tokens(self, batch, drop=None):
        """Clean context-view latent tokens for joint self-attention: (B, K*P, D) and the
        key mask (B, K*P), False for absent slots and for dropped (unconditional) samples."""
        if self.context_latents != "joint":
            return None, None
        lat = batch.get("context_latents")
        if lat is None:
            raise ValueError("context_latents='joint' requires batch['context_latents'] (B,K,P,C)")
        b, k, p, _ = lat.shape
        if k != self.k_max or p != self.tokens_per_image:
            raise ValueError(f"context_latents expected (B,{self.k_max},{self.tokens_per_image},C), got {tuple(lat.shape)}")
        present = batch["context_present"].bool()
        if drop is not None:
            present = present & ~drop.to(device=present.device, dtype=torch.bool)[:, None]
        tok = self.context_latent_adaptor(lat)
        cam = self.context_latent_camera_mlp(batch["context_camera_features"].to(tok.device, tok.dtype))
        tok = tok + cam[:, :, None, :] + self.context_latent_pos_embed[:, None] + self.context_latent_type_embed
        tok = tok * present[:, :, None, None].to(tok.dtype)  # absent slots are masked keys anyway
        mask = present[:, :, None].expand(b, k, p).reshape(b, k * p)
        return tok.reshape(b, k * p, -1), mask

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Training loss. Validation passes fixed noise/timesteps and reduction='none'
        (per-sample MSE); the default call samples them exactly as before."""
        action_gt = batch["action_gt"]
        b = action_gt.shape[0]
        device = action_gt.device
        drop = drop_text = None
        split = self.image_only_drop_prob > 0 or self.text_only_drop_prob > 0
        if self.training and noise is None and split:
            # one draw per sample: [all | images only | text only | nothing dropped]
            u = torch.rand(b, device=device)
            p_all, p_img = self.cond_drop_prob, self.cond_drop_prob + self.image_only_drop_prob
            p_txt = p_img + self.text_only_drop_prob
            drop = u < p_img                                   # all, or images only
            drop_text = (u < p_all) | ((u >= p_img) & (u < p_txt))  # all, or text only
        elif self.training and self.cond_drop_prob > 0 and noise is None:
            drop = torch.rand(b, device=device) < self.cond_drop_prob
        # Network calls in the compute precision; noise/scheduler math and the loss stay fp32.
        with self._autocast():
            lang, lang_mask, img, img_mask, state = self._conditions(
                slot_feats=batch["slot_feats"],
                context_camera_features=batch["context_camera_features"],
                target_camera_features=batch["target_camera_features"],
                context_present=batch["context_present"],
                drop=drop,
                text_embeds=batch.get("text_embeds"),
                text_mask=batch.get("text_mask"),
                drop_text=drop_text,
            )
            if noise is None:
                noise = torch.randn_like(action_gt)
            if timesteps is None:
                timesteps = torch.randint(0, self.runner.num_train_timesteps, (b,), device=device).long()
            noisy = self.runner.noise_scheduler.add_noise(action_gt, noise, timesteps)
            action_mask = torch.ones(b, 1, self.action_dim, device=device, dtype=action_gt.dtype)
            extra, extra_mask = self._context_latent_tokens(batch, drop)
            if self.adaln:
                action = self.runner.state_adaptor(torch.cat([noisy, action_mask.expand(-1, noisy.shape[1], -1)], dim=2))
                pred = self.runner.model(
                    action, None, timesteps,
                    self.runner.lang_adaptor(lang), self.runner.img_adaptor(img),
                    lang_mask=lang_mask, img_mask=img_mask, cond=state, extra=extra, extra_mask=extra_mask,
                )
            else:
                raw = torch.cat([state, noisy], dim=1)
                raw = torch.cat([raw, action_mask.expand(-1, raw.shape[1], -1)], dim=2)
                lang_cond, img_cond, state_action = self.runner.adapt_conditions(lang, img, raw)
                pred = self.runner.model(
                    state_action,
                    torch.ones(b, device=device, dtype=action_gt.dtype),
                    timesteps,
                    lang_cond,
                    img_cond,
                    lang_mask=lang_mask,
                    img_mask=img_mask,
                    **({} if extra is None else dict(extra=extra, extra_mask=extra_mask)),
                )
        pred = pred.float()
        if self.runner.prediction_type == "sample":
            target = action_gt
        elif self.runner.prediction_type == "epsilon":
            target = noise
        else:
            raise ValueError(f"Unsupported prediction_type={self.runner.prediction_type}")
        if reduction == "none":
            # Unweighted per-sample MSE (used for the comparable validation loss).
            return F.mse_loss(pred, target, reduction="none").flatten(1).mean(1)
        if reduction != "mean":
            raise ValueError(f"Unsupported reduction={reduction}")
        if self.loss_weighting == "min_snr":
            if self.runner.prediction_type != "sample":
                raise ValueError("min_snr weighting is implemented for x0 (sample) prediction")
            alphas = self.runner.noise_scheduler.alphas_cumprod.to(device)[timesteps].to(torch.float64)
            weight = (alphas / (1 - alphas)).clamp(max=self.min_snr_gamma).to(pred.dtype)
            per_sample = F.mse_loss(pred, target, reduction="none").flatten(1).mean(1)
            return (weight * per_sample).mean()
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def predict(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        num_steps: int = 20,
        seed: Optional[int] = None,
        cfg_scale: float = 1.0,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample target tokens. cfg_scale != 1 applies classifier-free guidance,
        out = uncond + s * (cond - uncond), where the unconditional branch drops all
        context images/cameras (target direction kept), matching training dropout.
        For x0 prediction this equals guidance on epsilon at a fixed x_t."""
        cond = dict(
            slot_feats=batch["slot_feats"],
            context_camera_features=batch["context_camera_features"],
            target_camera_features=batch["target_camera_features"],
            context_present=batch["context_present"],
            text_embeds=batch.get("text_embeds"),
            text_mask=batch.get("text_mask"),
        )
        with self._autocast():
            lang, lang_mask, img, img_mask, state = self._conditions(**cond)
            extra, extra_mask = self._context_latent_tokens(batch)
        b = state.shape[0]
        device = state.device
        dtype = state.dtype if self.precision == "fp32" else torch.float32  # sampler state stays fp32
        guided = float(cfg_scale) != 1.0
        if guided:
            all_drop = torch.ones(b, dtype=torch.bool, device=device)
            with self._autocast():
                u = self._conditions(**cond, drop=all_drop)
                lang, lang_mask, img, img_mask, state = (torch.cat([c, x], 0) for c, x in zip((lang, lang_mask, img, img_mask, state), u))
                if extra is not None:
                    ue, um = self._context_latent_tokens(batch, all_drop)
                    extra, extra_mask = torch.cat([extra, ue], 0), torch.cat([extra_mask, um], 0)
        extra_kw = {} if extra is None else dict(extra=extra, extra_mask=extra_mask)
        if initial_noise is not None:  # per-sample noise, independent of batch layout
            noisy = initial_noise.to(device=device, dtype=dtype)
        elif seed is not None:
            g = torch.Generator(device=device).manual_seed(int(seed))
            noisy = torch.randn((b, self.pred_horizon, self.action_dim), device=device, dtype=dtype, generator=g)
        else:
            noisy = torch.randn((b, self.pred_horizon, self.action_dim), device=device, dtype=dtype)
        n = 2 * b if guided else b
        action_mask = torch.ones(n, 1, self.action_dim, device=device, dtype=dtype)
        with self._autocast():
            if self.adaln:
                lang_cond, img_cond = self.runner.lang_adaptor(lang), self.runner.img_adaptor(img)
            else:
                lang_cond, img_cond, state_hidden = self.runner.adapt_conditions(
                    lang, img, torch.cat([state, action_mask], dim=2)
                )
        self.runner.noise_scheduler_sample.set_timesteps(int(num_steps))
        for t in self.runner.noise_scheduler_sample.timesteps:
            x = torch.cat([noisy, noisy], 0) if guided else noisy
            with self._autocast():
                action_raw = torch.cat([x, action_mask.expand(-1, self.pred_horizon, -1)], dim=2)
                action_hidden = self.runner.state_adaptor(action_raw)
                if self.adaln:
                    model_output = self.runner.model(
                        action_hidden, None, t.unsqueeze(-1).to(device), lang_cond, img_cond,
                        lang_mask=lang_mask, img_mask=img_mask, cond=state, **extra_kw,
                    )
                else:
                    traj = torch.cat([state_hidden, action_hidden], dim=1)
                    model_output = self.runner.model(
                        traj,
                        torch.ones(n, device=device, dtype=dtype),
                        t.unsqueeze(-1).to(device),
                        lang_cond,
                        img_cond,
                        lang_mask=lang_mask,
                        img_mask=img_mask,
                        **extra_kw,
                    )
            model_output = model_output.float()
            if guided:
                out_c, out_u = model_output.chunk(2)
                model_output = out_u + float(cfg_scale) * (out_c - out_u)
            noisy = self.runner.noise_scheduler_sample.step(model_output, t, noisy).prev_sample.to(dtype)
        return noisy
