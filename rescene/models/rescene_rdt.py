"""ReSCENE wrapper around the official RDT runner.

Mapping (robot -> ReSCENE):

* robot action chunk        -> anomaly target chunk (mask + delta/full latent patches)
* proprioception state      -> metadata token (dataset id + target view id embeddings)
* multi-view robot obs      -> image slots: [target view | normal context views | support anomaly overlays]
* task instruction          -> pseudo/real anomaly edit instruction
* control frequency         -> constant 1 (kept only for API compatibility)

The official ``RDTRunner`` (adapters + diffusion transformer + DDPM/DPM-Solver
schedulers) is used unmodified; this module only prepares its conditional
inputs and adds ReSCENE-specific embeddings:

* per-slot type embeddings (target / normal context / support anomaly)
* view-id embeddings, dataset-id embeddings
* learned "missing slot" token (used both for absent inputs and for modality
  dropout during training)
* learned null-text token (text dropout / text-free inference)
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.rdt_runner import RDTRunner  # official, unmodified
from rescene.codec.base import EditActionSpec

SLOT_TARGET = 0
SLOT_NORMAL = 1
SLOT_SUPPORT = 2


class ReSCENERDT(nn.Module):
    def __init__(
        self,
        rdt_config: dict,
        spec: EditActionSpec,
        lang_token_dim: int,
        img_token_dim: int,
        tokens_per_image: int,
        max_normal_views: int = 5,
        max_support: int = 2,
        num_datasets: int = 2,
        max_views: int = 5,
        max_lang_cond_len: int = 128,
        dtype=torch.float32,
        dropout: Optional[dict] = None,
        local_condition_dim: int = 0,
    ):
        super().__init__()
        self.spec = spec
        self.max_normal_views = max_normal_views
        self.max_support = max_support
        self.max_views = max_views
        self.tokens_per_image = tokens_per_image
        self.num_slots = 1 + max_normal_views + max_support
        self.max_lang_cond_len = max_lang_cond_len
        dropout = dropout or {}
        self.p_drop_text = float(dropout.get("text", 0.1))
        self.p_drop_support = float(dropout.get("support", 0.3))
        self.p_drop_view = float(dropout.get("view", 0.1))

        self.runner = RDTRunner(
            action_dim=spec.chunk_dim,
            pred_horizon=spec.chunk_len,
            config=rdt_config,
            lang_token_dim=lang_token_dim,
            img_token_dim=img_token_dim,
            state_token_dim=spec.chunk_dim,
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=self.num_slots * tokens_per_image,
            lang_pos_embed_config=None,
            img_pos_embed_config=[("image", (self.num_slots, tokens_per_image))],
            dtype=dtype,
        )

        # Optional spatially aligned local condition used by the latent-only
        # mask-conditioned inpainting model.  Each action token receives
        # [base-latent patch | binary-mask patch | mask-SDF patch].
        self.local_condition_dim = int(local_condition_dim or 0)
        self.local_condition_adaptor = None
        if self.local_condition_dim > 0:
            hidden = int(rdt_config["rdt"]["hidden_size"])
            self.local_condition_adaptor = nn.Sequential(
                nn.Linear(self.local_condition_dim, hidden),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden, hidden),
            )
            for module in self.local_condition_adaptor.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        # ReSCENE-specific embeddings, all in image-feature space
        self.slot_type_emb = nn.Embedding(3, img_token_dim)
        self.view_emb = nn.Embedding(max_views + 1, img_token_dim)  # last = unknown
        self.dataset_emb = nn.Embedding(num_datasets + 1, img_token_dim)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, 1, img_token_dim))
        self.null_lang_token = nn.Parameter(torch.zeros(1, 1, lang_token_dim))
        # metadata ("state") token replacing robot proprioception
        self.meta_dataset_emb = nn.Embedding(num_datasets + 1, spec.chunk_dim)
        self.meta_view_emb = nn.Embedding(max_views + 1, spec.chunk_dim)
        for emb in (self.slot_type_emb, self.view_emb, self.dataset_emb,
                    self.meta_dataset_emb, self.meta_view_emb):
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.normal_(self.missing_token, std=0.02)
        nn.init.normal_(self.null_lang_token, std=0.02)

        # fixed slot-type ids for [target, normal*K, support*S]
        slot_types = torch.tensor(
            [SLOT_TARGET] + [SLOT_NORMAL] * max_normal_views
            + [SLOT_SUPPORT] * max_support, dtype=torch.long)
        self.register_buffer("slot_types", slot_types, persistent=False)

    # ------------------------------------------------------------------
    def _clamp_view_ids(self, view_ids: torch.Tensor) -> torch.Tensor:
        unknown = self.max_views
        return torch.where((view_ids >= 0) & (view_ids < self.max_views),
                           view_ids, torch.full_like(view_ids, unknown))

    def assemble_image_tokens(self, slot_feats: torch.Tensor,
                              slot_present: torch.Tensor,
                              slot_view_ids: torch.Tensor,
                              dataset_id: torch.Tensor) -> torch.Tensor:
        """slot_feats: (B, S, P, D_img) frozen-encoder features per image slot.
        slot_present: (B, S) bool. slot_view_ids: (B, S) long (-1 = unknown).
        dataset_id: (B,) long. Returns flattened image tokens (B, S*P, D_img)."""
        B, S, P, D = slot_feats.shape
        assert S == self.num_slots and P == self.tokens_per_image, (
            f"expected {self.num_slots} slots x {self.tokens_per_image} tokens, "
            f"got {S} x {P}")
        feats = torch.where(slot_present[:, :, None, None],
                            slot_feats, self.missing_token.to(slot_feats.dtype))
        feats = feats + self.slot_type_emb(self.slot_types)[None, :, None, :]
        feats = feats + self.view_emb(self._clamp_view_ids(slot_view_ids))[:, :, None, :]
        feats = feats + self.dataset_emb(dataset_id)[:, None, None, :]
        return feats.reshape(B, S * P, D)

    def metadata_state_token(self, dataset_id: torch.Tensor,
                             target_view_id: torch.Tensor) -> torch.Tensor:
        """(B,) x (B,) -> (B, 1, chunk_dim)."""
        v = self._clamp_view_ids(target_view_id)
        return (self.meta_dataset_emb(dataset_id) + self.meta_view_emb(v)).unsqueeze(1)

    def null_text(self, batch_size: int, device) -> tuple:
        emb = self.null_lang_token.expand(batch_size, 1, -1).to(device)
        mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        return emb, mask

    def _apply_text(self, lang_emb, lang_mask, drop: torch.Tensor):
        """Replace dropped samples' text with the learned null token."""
        B = lang_emb.shape[0]
        if lang_emb.shape[1] > self.max_lang_cond_len:
            lang_emb = lang_emb[:, : self.max_lang_cond_len]
            lang_mask = lang_mask[:, : self.max_lang_cond_len]
        null = self.null_lang_token.to(lang_emb.dtype)
        out_emb = lang_emb.clone()
        out_mask = lang_mask.clone()
        for b in range(B):
            if drop[b]:
                out_emb[b] = 0.0
                out_emb[b, 0] = null[0, 0]
                out_mask[b] = False
                out_mask[b, 0] = True
        return out_emb, out_mask

    # ------------------------------------------------------------------
    def _prepare_training_inputs(self, batch: Dict[str, torch.Tensor]):
        """Apply modality dropout and assemble train-time RDT conditions."""
        B = batch["action_gt"].shape[0]
        device = batch["action_gt"].device
        present = batch["slot_present"].clone()

        if self.training:
            # Independent modality dropout. Target slot 0 (the masked target
            # view for inpainting) is never dropped.
            drop_text = torch.rand(B, device=device) < self.p_drop_text
            drop_support = torch.rand(B, device=device) < self.p_drop_support
            k = self.max_normal_views
            drop_view = torch.rand(B, k, device=device) < self.p_drop_view
            present[:, 1:1 + k] &= ~drop_view
            present[:, 1 + k:] &= ~drop_support[:, None]
        else:
            drop_text = torch.zeros(B, dtype=torch.bool, device=device)

        lang_emb, lang_mask = self._apply_text(
            batch["lang_emb"], batch["lang_mask"], drop_text)
        img_tokens = self.assemble_image_tokens(
            batch["slot_feats"], present, batch["slot_view_ids"], batch["dataset_id"])
        state_tokens = self.metadata_state_token(
            batch["dataset_id"], batch["target_view_id"])
        return lang_emb, lang_mask, img_tokens, state_tokens

    def _local_condition_hidden(
        self,
        local_condition: torch.Tensor,
        *,
        device,
        dtype,
    ) -> torch.Tensor:
        if self.local_condition_adaptor is None:
            raise RuntimeError(
                "This checkpoint/config has no local_condition_adaptor, but "
                "latent-only mask inpainting requires one."
            )
        if local_condition.ndim != 3:
            raise ValueError(
                f"local_condition must be (B,T,D), got {tuple(local_condition.shape)}."
            )
        if local_condition.shape[1] != self.spec.chunk_len:
            raise ValueError(
                f"local_condition T={local_condition.shape[1]} but expected "
                f"{self.spec.chunk_len}."
            )
        if local_condition.shape[2] != self.local_condition_dim:
            raise ValueError(
                f"local_condition D={local_condition.shape[2]} but expected "
                f"{self.local_condition_dim}."
            )
        x = local_condition.to(device=device, dtype=dtype)
        return self.local_condition_adaptor(x)

    def _compute_latent_only_inpainting_loss(
        self,
        batch: Dict[str, torch.Tensor],
        lang_emb: torch.Tensor,
        lang_mask: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Diffuse only latent actions inside the supplied mask.

        The user/GT mask is a clean spatial condition through
        ``local_condition_adaptor``.  Outside the generation region, the
        noisy action is replaced by a forward-noised base-image latent.
        Only the masked region contributes to the denoising objective.
        """
        runner = self.runner
        action_gt = batch["action_gt"]
        base_action = batch["inpaint_base_action"].to(
            device=action_gt.device, dtype=action_gt.dtype
        )
        outside = batch["inpaint_outside_mask"].to(
            device=action_gt.device, dtype=action_gt.dtype
        ).clamp(0, 1)
        loss_weight = batch["inpaint_loss_weight"].to(
            device=action_gt.device, dtype=action_gt.dtype
        ).clamp_min(0)
        local_condition = batch["inpaint_local_condition"]
        if not (
            action_gt.shape == base_action.shape == outside.shape == loss_weight.shape
        ):
            raise ValueError(
                "Latent-only inpainting tensors must share (B,T,D): "
                f"target={tuple(action_gt.shape)} base={tuple(base_action.shape)} "
                f"outside={tuple(outside.shape)} weight={tuple(loss_weight.shape)}"
            )

        B = action_gt.shape[0]
        device = action_gt.device
        noise = torch.randn_like(action_gt)
        timesteps = torch.randint(
            0, runner.num_train_timesteps, (B,), device=device
        ).long()
        noisy_target = runner.noise_scheduler.add_noise(action_gt, noise, timesteps)
        # Use the same noise realization on the base latent so boundary noise
        # remains spatially coherent across generated/preserved regions.
        noisy_base = runner.noise_scheduler.add_noise(base_action, noise, timesteps)
        inside = (1.0 - outside).clamp(0, 1)
        noisy_action = noisy_target * inside + noisy_base * outside

        action_valid_mask = torch.ones(
            B, 1, self.spec.chunk_dim, device=device, dtype=action_gt.dtype
        )
        raw_traj = torch.cat([state_tokens, noisy_action], dim=1)
        expanded_valid = action_valid_mask.expand(-1, raw_traj.shape[1], -1)
        raw_traj = torch.cat([raw_traj, expanded_valid], dim=2)
        lang_cond, img_cond, hidden_traj = runner.adapt_conditions(
            lang_emb, img_tokens, raw_traj
        )
        local_hidden = self._local_condition_hidden(
            local_condition, device=device, dtype=hidden_traj.dtype
        )
        hidden_traj = hidden_traj.clone()
        hidden_traj[:, 1:] = hidden_traj[:, 1:] + local_hidden
        pred = runner.model(
            hidden_traj, torch.ones(B, device=device), timesteps,
            lang_cond, img_cond, lang_mask=lang_mask
        )

        if runner.prediction_type == "epsilon":
            target = noise
        elif runner.prediction_type == "sample":
            target = action_gt
        else:
            raise ValueError(
                f"Unsupported prediction type {runner.prediction_type!r}"
            )
        squared = (pred - target).float().pow(2) * loss_weight.float()
        denom = loss_weight.float().sum(dim=(1, 2)).clamp_min(1.0)
        return (squared.sum(dim=(1, 2)) / denom).mean()

    def _compute_inpainting_loss(
        self,
        batch: Dict[str, torch.Tensor],
        lang_emb: torch.Tensor,
        lang_mask: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Masked diffusion objective with clean mask and preserved background.

        The action tensor still has shape ``[mask patch | latent patch]`` for
        checkpoint compatibility. Mask dimensions are clean known inputs;
        outside-mask latent dimensions are forward-noised base-image values;
        only generated-region latent dimensions contribute to the loss.
        """
        runner = self.runner
        action_gt = batch["action_gt"]
        known_action = batch["inpaint_known_action"].to(
            device=action_gt.device, dtype=action_gt.dtype)
        clean_known = batch["inpaint_clean_known_mask"].to(
            device=action_gt.device, dtype=action_gt.dtype).clamp(0, 1)
        noised_known = batch["inpaint_noised_known_mask"].to(
            device=action_gt.device, dtype=action_gt.dtype).clamp(0, 1)
        loss_weight = batch["inpaint_loss_weight"].to(
            device=action_gt.device, dtype=action_gt.dtype).clamp_min(0)

        if not (
            action_gt.shape == known_action.shape == clean_known.shape
            == noised_known.shape == loss_weight.shape
        ):
            raise ValueError(
                "Inpainting tensors must share shape (B,T,D): "
                f"target={tuple(action_gt.shape)}, known={tuple(known_action.shape)}, "
                f"clean={tuple(clean_known.shape)}, noised={tuple(noised_known.shape)}, "
                f"weight={tuple(loss_weight.shape)}"
            )
        overlap = (clean_known * noised_known).amax()
        if float(overlap.detach()) > 0:
            raise ValueError("clean-known and noised-known inpainting masks overlap.")

        B = action_gt.shape[0]
        device = action_gt.device
        noise = torch.randn_like(action_gt)
        known_noise = torch.randn_like(action_gt)
        timesteps = torch.randint(
            0, runner.num_train_timesteps, (B,), device=device
        ).long()

        noisy_target = runner.noise_scheduler.add_noise(
            action_gt, noise, timesteps)
        noisy_known = runner.noise_scheduler.add_noise(
            known_action, known_noise, timesteps)

        unknown = (1.0 - clean_known - noised_known).clamp(0, 1)
        noisy_action = (
            noisy_target * unknown
            + known_action * clean_known
            + noisy_known * noised_known
        )

        action_valid_mask = torch.ones(
            B, 1, self.spec.chunk_dim, device=device, dtype=action_gt.dtype)
        # Follow the official RDTRunner path exactly after constructing the
        # partially observed action trajectory.
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        expanded_valid = action_valid_mask.expand(
            -1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat(
            [state_action_traj, expanded_valid], dim=2)
        lang_cond, img_cond, state_action_traj = runner.adapt_conditions(
            lang_emb, img_tokens, state_action_traj)
        pred = runner.model(
            state_action_traj,
            torch.ones(B, device=device),
            timesteps,
            lang_cond,
            img_cond,
            lang_mask=lang_mask,
        )

        if runner.prediction_type == "epsilon":
            target = noise
        elif runner.prediction_type == "sample":
            target = action_gt
        else:
            raise ValueError(
                f"Unsupported prediction type {runner.prediction_type!r}")

        squared = (pred - target).float().pow(2) * loss_weight.float()
        denom = loss_weight.float().sum(dim=(1, 2)).clamp_min(1.0)
        return (squared.sum(dim=(1, 2)) / denom).mean()

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute ordinary joint diffusion or mask-conditioned inpainting loss."""
        lang_emb, lang_mask, img_tokens, state_tokens = (
            self._prepare_training_inputs(batch)
        )
        if "inpaint_local_condition" in batch:
            return self._compute_latent_only_inpainting_loss(
                batch, lang_emb, lang_mask, img_tokens, state_tokens)
        if "inpaint_loss_weight" in batch:
            return self._compute_inpainting_loss(
                batch, lang_emb, lang_mask, img_tokens, state_tokens)

        B = batch["action_gt"].shape[0]
        device = batch["action_gt"].device
        action_gt = batch["action_gt"]
        action_mask = torch.ones(
            B, 1, self.spec.chunk_dim, device=device, dtype=action_gt.dtype)
        ctrl_freqs = torch.ones(B, device=device)
        return self.runner.compute_loss(
            lang_tokens=lang_emb, lang_attn_mask=lang_mask, img_tokens=img_tokens,
            state_tokens=state_tokens, action_gt=action_gt,
            action_mask=action_mask, ctrl_freqs=ctrl_freqs)

    def _prepare_prediction_inputs(self, batch: Dict[str, torch.Tensor]):
        """Build the frozen text/image conditions used by inference.

        The helper is shared by ordinary joint sampling and oracle-channel
        sampling.  ``batch`` is expected to be the already encoded condition
        dictionary produced by :func:`rescene.pipeline.encode_batch_conditions`.
        """
        B = batch["slot_feats"].shape[0]
        device = batch["slot_feats"].device
        if batch.get("lang_emb") is None:
            lang_emb, lang_mask = self.null_text(B, device)
        else:
            drop = torch.zeros(B, dtype=torch.bool, device=device)
            lang_emb, lang_mask = self._apply_text(
                batch["lang_emb"], batch["lang_mask"], drop)
        img_tokens = self.assemble_image_tokens(
            batch["slot_feats"], batch["slot_present"], batch["slot_view_ids"],
            batch["dataset_id"])
        state_tokens = self.metadata_state_token(
            batch["dataset_id"], batch["target_view_id"])
        action_mask = torch.ones(
            B, 1, self.spec.chunk_dim, device=device, dtype=state_tokens.dtype)
        ctrl_freqs = torch.ones(B, device=device)
        return lang_emb, lang_mask, img_tokens, state_tokens, action_mask, ctrl_freqs

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Sample an edit-action chunk (B, T, D_chunk). Conditions are used as
        given (no dropout) — missing modalities are expressed by the caller via
        slot_present=False and/or use_text=False."""
        (lang_emb, lang_mask, img_tokens, state_tokens,
         action_mask, ctrl_freqs) = self._prepare_prediction_inputs(batch)
        return self.runner.predict_action(
            lang_tokens=lang_emb, lang_attn_mask=lang_mask, img_tokens=img_tokens,
            state_tokens=state_tokens, action_mask=action_mask, ctrl_freqs=ctrl_freqs)

    @torch.no_grad()
    def predict_inpaint_latent_only(
        self,
        batch: Dict[str, torch.Tensor],
        base_action: torch.Tensor,
        outside_mask: torch.Tensor,
        local_condition: torch.Tensor,
    ) -> torch.Tensor:
        """Sample anomaly-latent actions with explicit mask conditioning.

        ``outside_mask`` is one outside the dilated generation region.  Those
        latent scalars are clamped to forward-noised base-image values at every
        reverse step and to the exact base latent at the end.
        """
        if base_action.ndim != 3 or outside_mask.ndim != 3:
            raise ValueError("base_action/outside_mask must both be (B,T,D).")
        if base_action.shape != outside_mask.shape:
            raise ValueError(
                f"base/outside shapes differ: {tuple(base_action.shape)} vs "
                f"{tuple(outside_mask.shape)}"
            )
        expected = (self.spec.chunk_len, self.spec.chunk_dim)
        if tuple(base_action.shape[1:]) != expected:
            raise ValueError(
                f"Base action shape {tuple(base_action.shape[1:])}; expected {expected}."
            )

        (lang_tokens, lang_attn_mask, img_tokens, state_tokens,
         action_valid_mask, ctrl_freqs) = self._prepare_prediction_inputs(batch)
        runner = self.runner
        B = base_action.shape[0]
        device = state_tokens.device
        dtype = state_tokens.dtype
        base_action = base_action.to(device=device, dtype=dtype)
        outside = outside_mask.to(device=device, dtype=dtype).clamp(0, 1)
        inside = (1.0 - outside).clamp(0, 1)
        action_valid_mask = action_valid_mask.to(dtype=dtype)

        # Static multimodal and metadata conditions.
        state_raw = torch.cat([state_tokens, action_valid_mask], dim=2)
        lang_cond, img_cond, state_traj = runner.adapt_conditions(
            lang_tokens, img_tokens, state_raw
        )
        local_hidden = self._local_condition_hidden(
            local_condition, device=device, dtype=state_traj.dtype
        )

        noisy_action = torch.randn(
            (B, runner.pred_horizon, runner.action_dim),
            dtype=dtype, device=device
        )
        base_noise = torch.randn_like(noisy_action)
        expanded_valid = action_valid_mask.expand(-1, runner.pred_horizon, -1)
        runner.noise_scheduler_sample.set_timesteps(runner.num_inference_timesteps)
        timesteps = runner.noise_scheduler_sample.timesteps

        def noised_base_at(timestep):
            value = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
            t_batch = torch.full((B,), value, device=device, dtype=torch.long)
            return runner.noise_scheduler.add_noise(
                base_action, base_noise, t_batch
            ).to(dtype)

        for step_idx, t in enumerate(timesteps):
            noisy_action = noisy_action * inside + noised_base_at(t) * outside
            action_raw = torch.cat([noisy_action, expanded_valid], dim=2)
            action_hidden = runner.state_adaptor(action_raw) + local_hidden
            state_action_traj = torch.cat([state_traj, action_hidden], dim=1)
            model_output = runner.model(
                state_action_traj, ctrl_freqs, t.unsqueeze(-1).to(device),
                lang_cond, img_cond, lang_mask=lang_attn_mask
            )
            noisy_action = runner.noise_scheduler_sample.step(
                model_output, t, noisy_action
            ).prev_sample.to(dtype)
            next_base = (
                noised_base_at(timesteps[step_idx + 1])
                if step_idx + 1 < len(timesteps) else base_action
            )
            noisy_action = noisy_action * inside + next_base * outside

        return noisy_action * expanded_valid

    @torch.no_grad()
    def predict_inpaint(
        self,
        batch: Dict[str, torch.Tensor],
        known_action: torch.Tensor,
        clean_known_mask: torch.Tensor,
        noised_known_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Diffuse only the user-mask latent region.

        ``clean_known_mask`` selects the spatial mask-conditioning dimensions,
        which remain clean at every reverse step. ``noised_known_mask`` selects
        latent dimensions outside the dilated generation region; those values
        follow the forward-noised base-image latent at each timestep and become
        the exact base latent at the final step. The remaining dimensions are
        generated by RDT.
        """
        tensors = (known_action, clean_known_mask, noised_known_mask)
        if any(x.ndim != 3 for x in tensors):
            raise ValueError(
                "known_action, clean_known_mask and noised_known_mask must "
                "all have shape (B,T,D)."
            )
        if not (
            known_action.shape == clean_known_mask.shape == noised_known_mask.shape
        ):
            raise ValueError(
                f"Inpainting shapes differ: known={tuple(known_action.shape)}, "
                f"clean={tuple(clean_known_mask.shape)}, "
                f"noised={tuple(noised_known_mask.shape)}"
            )
        expected = (self.spec.chunk_len, self.spec.chunk_dim)
        if tuple(known_action.shape[1:]) != expected:
            raise ValueError(
                f"Known chunk has shape {tuple(known_action.shape[1:])}; "
                f"expected {expected}."
            )

        (lang_tokens, lang_attn_mask, img_tokens, state_tokens,
         action_valid_mask, ctrl_freqs) = self._prepare_prediction_inputs(batch)
        runner = self.runner
        B = known_action.shape[0]
        device = state_tokens.device
        dtype = state_tokens.dtype
        known_action = known_action.to(device=device, dtype=dtype)
        clean_known = clean_known_mask.to(
            device=device, dtype=dtype).clamp(0, 1)
        noised_known = noised_known_mask.to(
            device=device, dtype=dtype).clamp(0, 1)
        if float((clean_known * noised_known).amax().detach()) > 0:
            raise ValueError(
                "clean_known_mask and noised_known_mask must be disjoint."
            )
        unknown = (1.0 - clean_known - noised_known).clamp(0, 1)

        action_valid_mask = action_valid_mask.to(dtype=dtype)
        state_tokens = torch.cat([state_tokens, action_valid_mask], dim=2)
        lang_cond, img_cond, state_traj = runner.adapt_conditions(
            lang_tokens, img_tokens, state_tokens)

        noisy_action = torch.randn(
            (B, runner.pred_horizon, runner.action_dim),
            dtype=dtype, device=device,
        )
        known_noise = torch.randn_like(noisy_action)
        expanded_valid = action_valid_mask.expand(
            -1, runner.pred_horizon, -1)

        runner.noise_scheduler_sample.set_timesteps(
            runner.num_inference_timesteps)
        timesteps = runner.noise_scheduler_sample.timesteps

        def noised_known_at(timestep) -> torch.Tensor:
            t_value = (
                int(timestep.item()) if torch.is_tensor(timestep)
                else int(timestep)
            )
            t_batch = torch.full(
                (B,), t_value, device=device, dtype=torch.long)
            return runner.noise_scheduler.add_noise(
                known_action, known_noise, t_batch).to(dtype)

        def clamp_at(
            sample: torch.Tensor,
            noised_base: torch.Tensor,
        ) -> torch.Tensor:
            return (
                sample * unknown
                + known_action * clean_known
                + noised_base * noised_known
            )

        for step_idx, t in enumerate(timesteps):
            noisy_action = clamp_at(noisy_action, noised_known_at(t))

            action_traj = torch.cat(
                [noisy_action, expanded_valid], dim=2)
            action_traj = runner.state_adaptor(action_traj)
            state_action_traj = torch.cat(
                [state_traj, action_traj], dim=1)
            model_output = runner.model(
                state_action_traj,
                ctrl_freqs,
                t.unsqueeze(-1).to(device),
                lang_cond,
                img_cond,
                lang_mask=lang_attn_mask,
            )
            noisy_action = runner.noise_scheduler_sample.step(
                model_output, t, noisy_action).prev_sample.to(dtype)

            if step_idx + 1 < len(timesteps):
                next_known = noised_known_at(timesteps[step_idx + 1])
            else:
                next_known = known_action
            noisy_action = clamp_at(noisy_action, next_known)

        return noisy_action * expanded_valid

    @torch.no_grad()
    def predict_with_known_action(
        self,
        batch: Dict[str, torch.Tensor],
        known_action: torch.Tensor,
        known_action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sample the unknown action dimensions while clamping known ones.

        This implements diffusion inpainting in edit-action space.  At each
        reverse step, dimensions selected by ``known_action_mask`` are replaced
        with the forward-noised value of the clean ``known_action`` at that
        timestep.  The remaining dimensions are sampled normally.  For the
        oracle-mask evaluation, ``known_action`` is the GT normalized chunk and
        ``known_action_mask`` selects only the mask-SDF channel values.

        Args:
            batch: Encoded ReSCENE conditions.
            known_action: Clean normalized chunk, shape ``(B,T,D)``.
            known_action_mask: Boolean/0-1 tensor of the same shape; one means
                that the corresponding action scalar is observed.

        Returns:
            Conditionally sampled chunk of shape ``(B,T,D)``.
        """
        if known_action.ndim != 3 or known_action_mask.ndim != 3:
            raise ValueError(
                "known_action and known_action_mask must both have shape (B,T,D)."
            )
        if known_action.shape != known_action_mask.shape:
            raise ValueError(
                f"known_action shape {tuple(known_action.shape)} does not match "
                f"known_action_mask {tuple(known_action_mask.shape)}."
            )
        expected = (self.spec.chunk_len, self.spec.chunk_dim)
        if tuple(known_action.shape[1:]) != expected:
            raise ValueError(
                f"Known chunk has shape {tuple(known_action.shape[1:])}; "
                f"expected {expected}."
            )

        (lang_tokens, lang_attn_mask, img_tokens, state_tokens,
         action_valid_mask, ctrl_freqs) = self._prepare_prediction_inputs(batch)

        runner = self.runner
        B = known_action.shape[0]
        device = state_tokens.device
        dtype = state_tokens.dtype
        known_action = known_action.to(device=device, dtype=dtype)
        known_mask = known_action_mask.to(device=device, dtype=dtype).clamp(0, 1)
        action_valid_mask = action_valid_mask.to(dtype=dtype)

        # Prepare the static multimodal conditions exactly as RDTRunner does.
        state_tokens = torch.cat([state_tokens, action_valid_mask], dim=2)
        lang_cond, img_cond, state_traj = runner.adapt_conditions(
            lang_tokens, img_tokens, state_tokens)

        # Draw the ordinary initial sample first.  This preserves the same
        # unknown-channel initial noise as an unconditional run under the same
        # RNG seed; the second draw is used only to forward-noise known values.
        noisy_action = torch.randn(
            (B, runner.pred_horizon, runner.action_dim),
            dtype=dtype, device=device)
        known_noise = torch.randn_like(noisy_action)
        expanded_valid_mask = action_valid_mask.expand(-1, runner.pred_horizon, -1)

        runner.noise_scheduler_sample.set_timesteps(runner.num_inference_timesteps)
        timesteps = runner.noise_scheduler_sample.timesteps

        def noised_known_at(timestep) -> torch.Tensor:
            t_value = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
            t_batch = torch.full(
                (B,), t_value, device=device, dtype=torch.long)
            return runner.noise_scheduler.add_noise(
                known_action, known_noise, t_batch).to(dtype)

        for step_idx, t in enumerate(timesteps):
            # Clamp observed channels at the current diffusion noise level so
            # the transformer can condition all latent tokens on the GT mask.
            known_xt = noised_known_at(t)
            noisy_action = noisy_action * (1.0 - known_mask) + known_xt * known_mask

            action_traj = torch.cat([noisy_action, expanded_valid_mask], dim=2)
            action_traj = runner.state_adaptor(action_traj)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)

            model_output = runner.model(
                state_action_traj,
                ctrl_freqs,
                t.unsqueeze(-1).to(device),
                lang_cond,
                img_cond,
                lang_mask=lang_attn_mask,
            )
            noisy_action = runner.noise_scheduler_sample.step(
                model_output, t, noisy_action).prev_sample.to(dtype)

            # Clamp to the next noise level as well.  The final sample receives
            # the exact clean observed values, avoiding numerical drift.
            if step_idx + 1 < len(timesteps):
                known_next = noised_known_at(timesteps[step_idx + 1])
            else:
                known_next = known_action
            noisy_action = (
                noisy_action * (1.0 - known_mask) + known_next * known_mask
            )

        return noisy_action * expanded_valid_mask

    def forward(self, batch):
        return self.compute_loss(batch)
