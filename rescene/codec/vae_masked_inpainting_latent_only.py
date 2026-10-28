"""Mask-conditioned local VAE-latent inpainting with latent-only RDT actions.

Clean formulation used by ReSCENE V3:

    inputs  = masked/base image + user/GT mask + text + multi-view context
    output  = anomaly VAE latent only

At 256 px with SD VAE and patch_size=4:

    latent grid  : (B,4,32,32)
    RDT actions  : (B,64,64)  # 8x8 spatial tokens, 4x4x4 values/token

The mask is *not* an output channel.  It is converted to spatially aligned
binary/SDF patch tokens and concatenated with the clean base-latent patch to
form a local condition for each RDT action token:

    local condition/token = [base latent 64 | binary mask 16 | mask SDF 16]
                          = 96 dims -> learned adapter -> RDT hidden size

Only latent values inside a dilated generation region are diffused and
supervised.  Outside values are clamped to the forward-noised base latent at
every reverse step.  After VAE decoding, full-resolution pixel compositing
preserves the original image exactly outside the supplied mask.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from rescene.codec.base import EditActionCodec, EditActionSpec
from rescene.codec.sdf import mask_to_sdf


def _as_modes(value) -> Tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    modes = tuple(str(x).lower() for x in value)
    allowed = {"noise", "mean", "zero", "blur"}
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown inpainting fill mode(s) {unknown}; expected {sorted(allowed)}."
        )
    if not modes:
        raise ValueError("At least one training fill mode is required.")
    return modes


class VAEMaskedInpaintingLatentOnlyCodec(EditActionCodec):
    """Latent-only patch diffusion with the mask supplied as an input condition."""

    content_label = "latent"
    is_mask_conditioned_inpainting = True
    is_latent_only_mask_inpainting = True

    def __init__(
        self,
        vae,
        image_size: int = 256,
        patch_size: int = 4,
        latent_scale: float = 1.0,
        sdf_clip_frac: float = 0.25,
        mask_dilation_latent: int = 1,
        condition_dilation_pixels: int | None = None,
        pixel_feather_radius: int = 0,
        boundary_weight: float = 2.0,
        training_fill_modes: Sequence[str] = ("noise", "mean", "zero"),
        eval_fill_mode: str = "mean",
        blur_kernel: int = 21,
        mask_downsample: str = "max",
    ):
        super().__init__()
        if latent_scale <= 0:
            raise ValueError("latent_scale must be positive.")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if mask_dilation_latent < 0:
            raise ValueError("mask_dilation_latent must be non-negative.")
        if pixel_feather_radius < 0:
            raise ValueError("pixel_feather_radius must be non-negative.")
        if boundary_weight < 0:
            raise ValueError("boundary_weight must be non-negative.")
        if mask_downsample not in {"max", "area"}:
            raise ValueError("mask_downsample must be 'max' or 'area'.")

        self.vae = vae
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.latent_scale = float(latent_scale)
        self.sdf_clip_frac = float(sdf_clip_frac)
        self.mask_dilation_latent = int(mask_dilation_latent)
        self.pixel_feather_radius = int(pixel_feather_radius)
        self.boundary_weight = float(boundary_weight)
        self.training_fill_modes = _as_modes(training_fill_modes)
        self.eval_fill_mode = _as_modes([eval_fill_mode])[0]
        self.blur_kernel = int(blur_kernel)
        if self.blur_kernel <= 0:
            raise ValueError("blur_kernel must be positive.")
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1
        self.mask_downsample = str(mask_downsample)

        factor = int(vae.downsample_factor)
        if self.image_size % factor != 0:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by VAE factor={factor}."
            )
        self.downsample_factor = factor
        if condition_dilation_pixels is None:
            condition_dilation_pixels = self.mask_dilation_latent * factor
        self.condition_dilation_pixels = int(condition_dilation_pixels)
        if self.condition_dilation_pixels < 0:
            raise ValueError("condition_dilation_pixels must be non-negative.")

        latent_h = self.image_size // factor
        latent_w = latent_h
        latent_channels = int(vae.latent_channels)
        self.spec = EditActionSpec(
            grid_channels=latent_channels,
            grid_h=latent_h,
            grid_w=latent_w,
            mask_channels=0,
            tokenization="patch2d",
            patch_size=self.patch_size,
        )
        # Per RDT spatial token: clean base latent + binary mask + mask SDF.
        self.mask_token_dim = self.patch_size * self.patch_size
        self.local_condition_dim = self.spec.chunk_dim + 2 * self.mask_token_dim

    # ------------------------------------------------------------------
    @staticmethod
    def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask
        k = 2 * radius + 1
        return F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)

    @staticmethod
    def _blur_image(image: torch.Tensor, kernel: int) -> torch.Tensor:
        if kernel <= 1:
            return image
        pad = kernel // 2
        padded = F.pad(image, (pad, pad, pad, pad), mode="reflect")
        return F.avg_pool2d(padded, kernel_size=kernel, stride=1)

    def _binary_mask_grid(self, mask: torch.Tensor) -> torch.Tensor:
        binary = (mask.float() > 0.5).float()
        size = (self.spec.grid_h, self.spec.grid_w)
        if self.mask_downsample == "max":
            return F.adaptive_max_pool2d(binary, size)
        low = F.interpolate(binary, size=size, mode="area")
        return (low > 0.2).float()

    def _mask_sdf_grid(self, binary: torch.Tensor) -> torch.Tensor:
        return mask_to_sdf(binary, clip_frac=self.sdf_clip_frac)

    def _mask_regions(self, mask: torch.Tensor):
        core = self._binary_mask_grid(mask)
        gen = self._dilate(core, self.mask_dilation_latent).clamp(0, 1)
        ring = (gen - core).clamp(0, 1)
        return core, gen, ring

    def _patchify_single_channel(self, grid: torch.Tensor) -> torch.Tensor:
        if grid.ndim != 4 or grid.shape[1] != 1:
            raise ValueError(
                f"Expected single-channel grid (B,1,H,W), received {tuple(grid.shape)}."
            )
        b, _, h, w = grid.shape
        p = self.patch_size
        if (h, w) != (self.spec.grid_h, self.spec.grid_w):
            raise ValueError(
                f"Mask grid {(h,w)} does not match latent grid "
                f"{(self.spec.grid_h,self.spec.grid_w)}."
            )
        patches = grid.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches.view(b, self.spec.chunk_len, p * p)

    def _expanded_mask_chunk(self, mask_grid: torch.Tensor) -> torch.Tensor:
        expanded = mask_grid.expand(-1, self.spec.grid_channels, -1, -1)
        return self.grid_to_chunk(expanded)

    def _pixel_alpha(self, given_mask: torch.Tensor) -> torch.Tensor:
        core = (given_mask.float() > 0.5).float()
        r = self.pixel_feather_radius
        if r <= 0:
            return core
        dilated = self._dilate(core, r)
        k = 2 * r + 1
        soft = F.avg_pool2d(dilated, kernel_size=k, stride=1, padding=r)
        return torch.maximum(core, soft).clamp(0, 1)

    # ------------------------------------------------------------------
    def make_condition_image(
        self,
        base_img: torch.Tensor,
        mask: torch.Tensor,
        training: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Hole-fill the selected region before frozen image encoding."""
        if base_img.ndim != 4 or mask.ndim != 4:
            raise ValueError("base_img/mask must be (B,3,H,W)/(B,1,H,W).")
        core = (mask.float() > 0.5).float()
        hole = self._dilate(core, self.condition_dilation_pixels).clamp(0, 1)

        modes = self.training_fill_modes if training else (self.eval_fill_mode,)
        B = base_img.shape[0]
        if training and len(modes) > 1:
            mode_ids = torch.randint(0, len(modes), (B,), device=base_img.device)
        else:
            mode_ids = torch.zeros(B, dtype=torch.long, device=base_img.device)

        fills = {
            "mean": base_img.mean(dim=(2, 3), keepdim=True).expand_as(base_img),
            "zero": torch.zeros_like(base_img),
            "noise": torch.rand_like(base_img),
            "blur": self._blur_image(base_img, self.blur_kernel),
        }
        fill = torch.empty_like(base_img)
        for i, mode in enumerate(modes):
            select = mode_ids == i
            if select.any():
                fill[select] = fills[mode][select]
        condition = base_img * (1.0 - hole) + fill * hole
        return {
            "condition_image": condition.clamp(0, 1),
            "condition_hole_mask": hole,
            "given_mask": core,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_target(
        self,
        normal_img: torch.Tensor,
        anomaly_img: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        del normal_img, mask
        z_target = self.vae.encode(anomaly_img) * self.latent_scale
        return self.grid_to_chunk(self._normalize(z_target))

    @torch.no_grad()
    def build_training_tensors(
        self,
        base_img: torch.Tensor,
        anomaly_img: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Build latent target, spatial mask condition and masked-loss tensors."""
        z_target = self.vae.encode(anomaly_img) * self.latent_scale
        z_base = self.vae.encode(base_img) * self.latent_scale

        target_grid = self._normalize(z_target)
        base_grid = self._normalize(z_base)
        action_gt = self.grid_to_chunk(target_grid)
        base_action = self.grid_to_chunk(base_grid)

        core, gen, ring = self._mask_regions(mask)
        core = core.to(device=z_target.device, dtype=z_target.dtype)
        gen = gen.to(device=z_target.device, dtype=z_target.dtype)
        ring = ring.to(device=z_target.device, dtype=z_target.dtype)
        sdf = self._mask_sdf_grid(core).to(device=z_target.device, dtype=z_target.dtype)

        spatial_weight = core + self.boundary_weight * ring
        empty = spatial_weight.flatten(1).sum(dim=1) <= 0
        if empty.any():
            spatial_weight = spatial_weight.clone()
            spatial_weight[empty, :, self.spec.grid_h // 2, self.spec.grid_w // 2] = 1.0

        loss_weight = self._expanded_mask_chunk(spatial_weight)
        outside_mask = self._expanded_mask_chunk((1.0 - gen).clamp(0, 1))
        mask_tokens = self._patchify_single_channel(core)
        sdf_tokens = self._patchify_single_channel(sdf)
        local_condition = torch.cat([base_action, mask_tokens, sdf_tokens], dim=-1)

        return {
            "action_gt": action_gt,
            "inpaint_base_action": base_action,
            "inpaint_outside_mask": outside_mask,
            "inpaint_loss_weight": loss_weight,
            "inpaint_local_condition": local_condition,
            "inpaint_mask_tokens": mask_tokens,
            "inpaint_mask_sdf_tokens": sdf_tokens,
            "inpaint_core_mask_grid": core,
            "inpaint_generation_mask_grid": gen,
        }

    @torch.no_grad()
    def build_sampling_tensors(
        self,
        base_img: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_base = self.vae.encode(base_img) * self.latent_scale
        base_grid = self._normalize(z_base)
        base_action = self.grid_to_chunk(base_grid)

        core, gen, _ = self._mask_regions(mask)
        core = core.to(device=z_base.device, dtype=z_base.dtype)
        gen = gen.to(device=z_base.device, dtype=z_base.dtype)
        sdf = self._mask_sdf_grid(core).to(device=z_base.device, dtype=z_base.dtype)
        mask_tokens = self._patchify_single_channel(core)
        sdf_tokens = self._patchify_single_channel(sdf)
        local_condition = torch.cat([base_action, mask_tokens, sdf_tokens], dim=-1)
        outside_mask = self._expanded_mask_chunk((1.0 - gen).clamp(0, 1))

        return {
            "inpaint_base_action": base_action,
            "inpaint_outside_mask": outside_mask,
            "inpaint_local_condition": local_condition,
            "inpaint_mask_tokens": mask_tokens,
            "inpaint_mask_sdf_tokens": sdf_tokens,
            "inpaint_generation_mask_grid": gen,
            "inpaint_core_mask_grid": core,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode(
        self,
        chunk: torch.Tensor,
        normal_img: torch.Tensor,
        local_residual: torch.Tensor = None,
        *,
        given_mask: torch.Tensor | None = None,
        anomaly_strength: float = 1.0,
    ) -> dict:
        if given_mask is None:
            raise ValueError(
                "Latent-only mask-conditioned decoding requires given_mask."
            )
        anomaly_strength = float(anomaly_strength)
        if anomaly_strength < 0:
            raise ValueError(
                f"anomaly_strength must be non-negative, got {anomaly_strength}."
            )

        # Native RDT prediction in the VAE latent space.  ``self.vae.encode``
        # already uses the same Diffusers VAE scaling factor as ``decode``.
        z_candidate_native = self._denormalize(
            self.chunk_to_grid(chunk.float())
        ) / self.latent_scale

        # Inference-only severity control.  Preserve the normal/base latent as
        # the origin and extrapolate only the predicted anomaly residual:
        #
        #   z_strong = z_base + s * (z_pred - z_base)
        #
        # s=1.0 is exactly the old behavior; s>1 strengthens the anomaly;
        # s=0 collapses the candidate to the VAE reconstruction of the base.
        # The final pixel compositing below still restores the original normal
        # pixels outside the user-provided mask.
        if anomaly_strength == 1.0:
            z_candidate = z_candidate_native
            z_base = None
        else:
            z_base = self.vae.encode(normal_img.float())
            z_base = z_base.to(
                device=z_candidate_native.device, dtype=z_candidate_native.dtype
            )
            z_candidate = z_base + anomaly_strength * (
                z_candidate_native - z_base
            )

        candidate = self.vae.decode(z_candidate)
        candidate_native = (
            candidate
            if anomaly_strength == 1.0
            else self.vae.decode(z_candidate_native)
        )

        mask_pix = (given_mask.float().to(candidate.device) > 0.5).float()
        alpha = self._pixel_alpha(mask_pix).to(candidate.device)
        base = normal_img.float().to(candidate.device)
        image = base * (1.0 - alpha) + candidate * alpha
        if local_residual is not None:
            image = image + alpha * local_residual.to(image.device)

        mask_grid = self._mask_sdf_grid(
            self._binary_mask_grid(given_mask).to(candidate.device)
        ).to(candidate.dtype)
        return {
            "image": image.clamp(0, 1),
            # Strength-adjusted candidate used for the final composition.
            "candidate_image": candidate.clamp(0, 1),
            # Native s=1 prediction is returned as well for diagnostics.
            "candidate_image_native": candidate_native.clamp(0, 1),
            "mask": mask_pix,
            "composition_alpha": alpha,
            "mask_grid": mask_grid,
            "content_grid": z_candidate,
            "latent_grid": z_candidate,
            "latent_grid_native": z_candidate_native,
            "z_anomaly_pred": z_candidate,
            "z_anomaly_pred_native": z_candidate_native,
            "z_base": z_base,
            "anomaly_strength": anomaly_strength,
            "z_syn": z_candidate,
        }
