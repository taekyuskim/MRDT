"""Mask-conditioned local VAE-latent inpainting with 2-D patch tokens.

This codec keeps the same RDT action shape as ``vae_full_latent_patch`` so a
full-latent ReSCENE checkpoint can be reused as initialization:

    grid  = concat([known_mask_SDF(1,h,w), anomaly_latent(4,h,w)])
    chunk = patchify_2d(grid, patch_size=4)             # 256 px -> (B,64,80)

The mask channel is no longer a prediction target.  It is a clean, spatially
aligned input condition.  Only latent values inside a dilated generation
region are diffused and supervised.  Latent values outside that region are
clamped to the base-image latent at every reverse-diffusion step.

Training uses self-inpainting pairs built from each real anomaly image:
the GT mask region is removed from the image used in target slot 0, while the
original image supplies the target latent.  This avoids requiring an exact
normal/anomaly pair.  At inference, the base image is the user-provided normal
image and the user/GT mask specifies where an anomaly may be generated.

After VAE decoding, the candidate is composited with the original base image
using the full-resolution given mask.  Thus pixels outside the mask (apart
from an optional narrow feather ring) are preserved exactly.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F

from rescene.codec.base import EditActionCodec, EditActionSpec
from rescene.codec.sdf import mask_to_sdf, sdf_to_mask


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


class VAEMaskedInpaintingPatchCodec(EditActionCodec):
    """Known-mask latent inpainting while retaining RDT ``T=64,D=80``."""

    content_label = "latent"
    is_mask_conditioned_inpainting = True

    def __init__(
        self,
        vae,
        image_size: int = 256,
        use_sdf: bool = True,
        patch_size: int = 4,
        latent_scale: float = 1.0,
        sdf_clip_frac: float = 0.25,
        mask_dilation_latent: int = 1,
        condition_dilation_pixels: int | None = None,
        pixel_feather_radius: int = 2,
        boundary_weight: float = 2.0,
        training_fill_modes: Sequence[str] = ("noise", "mean", "zero"),
        eval_fill_mode: str = "mean",
        blur_kernel: int = 21,
        mask_downsample: str = "max",
    ):
        super().__init__()
        if latent_scale <= 0:
            raise ValueError(f"latent_scale must be positive, received {latent_scale}.")
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
        self.use_sdf = bool(use_sdf)
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
                f"image_size={self.image_size} must be divisible by the VAE "
                f"downsample factor={factor}."
            )
        self.downsample_factor = factor
        if condition_dilation_pixels is None:
            condition_dilation_pixels = self.mask_dilation_latent * factor
        self.condition_dilation_pixels = int(condition_dilation_pixels)
        if self.condition_dilation_pixels < 0:
            raise ValueError("condition_dilation_pixels must be non-negative.")

        latent_h = self.image_size // factor
        latent_w = latent_h
        self.spec = EditActionSpec(
            grid_channels=1 + int(vae.latent_channels),
            grid_h=latent_h,
            grid_w=latent_w,
            mask_channels=1,
            tokenization="patch2d",
            patch_size=self.patch_size,
        )

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
        # Reflect padding avoids the dark border introduced by zero-padded
        # average pooling.
        padded = F.pad(image, (pad, pad, pad, pad), mode="reflect")
        return F.avg_pool2d(padded, kernel_size=kernel, stride=1)

    def _binary_mask_grid(self, mask: torch.Tensor) -> torch.Tensor:
        """Preserve tiny defects when reducing the user/GT mask to VAE size."""
        binary = (mask.float() > 0.5).float()
        size = (self.spec.grid_h, self.spec.grid_w)
        if self.mask_downsample == "max":
            return F.adaptive_max_pool2d(binary, size)
        low = F.interpolate(binary, size=size, mode="area")
        return (low > 0.2).float()

    def _mask_grid_from_binary(self, binary: torch.Tensor) -> torch.Tensor:
        if self.use_sdf:
            return mask_to_sdf(binary, clip_frac=self.sdf_clip_frac)
        return binary * 2.0 - 1.0

    def _mask_grid(self, mask: torch.Tensor) -> torch.Tensor:
        return self._mask_grid_from_binary(self._binary_mask_grid(mask))

    def _mask_regions(self, mask: torch.Tensor):
        core = self._binary_mask_grid(mask)
        gen = self._dilate(core, self.mask_dilation_latent).clamp(0, 1)
        ring = (gen - core).clamp(0, 1)
        return core, gen, ring

    def mask_grid_to_pixel_mask(self, mask_grid: torch.Tensor) -> torch.Tensor:
        binary = sdf_to_mask(mask_grid) if self.use_sdf else (mask_grid > 0.0).float()
        return F.interpolate(
            binary, size=(self.image_size, self.image_size), mode="nearest"
        )

    def _pixel_alpha(self, given_mask: torch.Tensor) -> torch.Tensor:
        """Full-resolution composition alpha with an optional narrow soft ring."""
        core = (given_mask.float() > 0.5).float()
        r = self.pixel_feather_radius
        if r <= 0:
            return core
        dilated = self._dilate(core, r)
        k = 2 * r + 1
        soft = F.avg_pool2d(dilated, kernel_size=k, stride=1, padding=r)
        # Preserve the requested core exactly; feather only the immediate ring.
        return torch.maximum(core, soft).clamp(0, 1)

    # ------------------------------------------------------------------
    def make_condition_image(
        self,
        base_img: torch.Tensor,
        mask: torch.Tensor,
        training: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Remove the requested region before frozen image encoding.

        ``base_img`` stays untouched and is later used for latent/pixel
        preservation.  The returned ``condition_image`` is used only in the
        target-view image slot.
        """
        if base_img.ndim != 4 or mask.ndim != 4:
            raise ValueError("base_img and mask must have shapes (B,3,H,W)/(B,1,H,W).")
        core = (mask.float() > 0.5).float()
        hole = self._dilate(core, self.condition_dilation_pixels).clamp(0, 1)

        modes = self.training_fill_modes if training else (self.eval_fill_mode,)
        B = base_img.shape[0]
        if training and len(modes) > 1:
            mode_ids = torch.randint(
                0, len(modes), (B,), device=base_img.device
            )
        else:
            mode_ids = torch.zeros(B, dtype=torch.long, device=base_img.device)

        mean_fill = base_img.mean(dim=(2, 3), keepdim=True).expand_as(base_img)
        zero_fill = torch.zeros_like(base_img)
        noise_fill = torch.rand_like(base_img)
        blur_fill = self._blur_image(base_img, self.blur_kernel)
        fills = {
            "mean": mean_fill,
            "zero": zero_fill,
            "noise": noise_fill,
            "blur": blur_fill,
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
        # ``normal_img`` is intentionally not subtracted.
        del normal_img
        z_target = self.vae.encode(anomaly_img) * self.latent_scale
        mask_grid = self._mask_grid(mask).to(z_target.device)
        grid = torch.cat([mask_grid, z_target], dim=1)
        return self.grid_to_chunk(self._normalize(grid))

    @torch.no_grad()
    def build_training_tensors(
        self,
        base_img: torch.Tensor,
        anomaly_img: torch.Tensor,
        mask: torch.Tensor,
        *,
        base_is_target: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Build target, known values, clamp masks and masked-loss weights."""
        z_target = self.vae.encode(anomaly_img) * self.latent_scale
        z_base = z_target if base_is_target else (
            self.vae.encode(base_img) * self.latent_scale
        )
        mask_grid = self._mask_grid(mask).to(z_target.device)
        target_grid = torch.cat([mask_grid, z_target], dim=1)
        known_grid = torch.cat([mask_grid, z_base], dim=1)

        action_gt = self.grid_to_chunk(self._normalize(target_grid))
        known_action = self.grid_to_chunk(self._normalize(known_grid))

        core, gen, ring = self._mask_regions(mask)
        core = core.to(z_target.device)
        gen = gen.to(z_target.device)
        ring = ring.to(z_target.device)

        B = mask.shape[0]
        C = self.spec.grid_channels
        clean_grid = torch.zeros(
            B, C, self.spec.grid_h, self.spec.grid_w,
            device=z_target.device, dtype=z_target.dtype,
        )
        # The mask channel is a clean known condition at every diffusion step.
        clean_grid[:, :1] = 1.0

        noised_known_grid = torch.zeros_like(clean_grid)
        outside = (1.0 - gen).clamp(0, 1)
        noised_known_grid[:, 1:] = outside.expand(
            -1, self.spec.content_channels, -1, -1
        )

        loss_grid = torch.zeros_like(clean_grid)
        spatial_weight = core + self.boundary_weight * ring
        # Every anomaly has a non-empty mask, but keep training numerically safe
        # for malformed records by assigning the strongest cell if needed.
        empty = spatial_weight.flatten(1).sum(dim=1) <= 0
        if empty.any():
            spatial_weight = spatial_weight.clone()
            spatial_weight[empty, :, self.spec.grid_h // 2, self.spec.grid_w // 2] = 1.0
        loss_grid[:, 1:] = spatial_weight.expand(
            -1, self.spec.content_channels, -1, -1
        )

        return {
            "action_gt": action_gt,
            "inpaint_known_action": known_action,
            "inpaint_clean_known_mask": self.grid_to_chunk(clean_grid),
            "inpaint_noised_known_mask": self.grid_to_chunk(noised_known_grid),
            "inpaint_loss_weight": self.grid_to_chunk(loss_grid),
            "inpaint_core_mask_grid": core,
            "inpaint_generation_mask_grid": gen,
        }

    @torch.no_grad()
    def build_sampling_tensors(
        self,
        base_img: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Build action-space conditions for normal-image inpainting."""
        z_base = self.vae.encode(base_img) * self.latent_scale
        mask_grid = self._mask_grid(mask).to(z_base.device)
        known_grid = torch.cat([mask_grid, z_base], dim=1)
        known_action = self.grid_to_chunk(self._normalize(known_grid))

        _, gen, _ = self._mask_regions(mask)
        gen = gen.to(z_base.device)
        B = mask.shape[0]
        C = self.spec.grid_channels
        clean_grid = torch.zeros(
            B, C, self.spec.grid_h, self.spec.grid_w,
            device=z_base.device, dtype=z_base.dtype,
        )
        clean_grid[:, :1] = 1.0
        noised_known_grid = torch.zeros_like(clean_grid)
        noised_known_grid[:, 1:] = (1.0 - gen).expand(
            -1, self.spec.content_channels, -1, -1
        )
        return {
            "inpaint_known_action": known_action,
            "inpaint_clean_known_mask": self.grid_to_chunk(clean_grid),
            "inpaint_noised_known_mask": self.grid_to_chunk(noised_known_grid),
            "inpaint_generation_mask_grid": gen,
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
    ) -> dict:
        grid = self._denormalize(self.chunk_to_grid(chunk.float()))
        mask_grid = grid[:, :1]
        z_candidate = grid[:, 1:] / self.latent_scale
        candidate = self.vae.decode(z_candidate)

        if given_mask is None:
            mask_pix = self.mask_grid_to_pixel_mask(mask_grid)
        else:
            mask_pix = (given_mask.float().to(candidate.device) > 0.5).float()
        alpha = self._pixel_alpha(mask_pix).to(candidate.device)
        base = normal_img.float().to(candidate.device)
        image = base * (1.0 - alpha) + candidate * alpha

        if local_residual is not None:
            image = image + alpha * local_residual.to(image.device)

        return {
            "image": image.clamp(0, 1),
            "candidate_image": candidate.clamp(0, 1),
            "mask": mask_pix,
            "composition_alpha": alpha,
            "mask_grid": mask_grid,
            "content_grid": z_candidate,
            "latent_grid": z_candidate,
            "z_anomaly_pred": z_candidate,
            "z_syn": z_candidate,
        }
