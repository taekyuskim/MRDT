"""Full-anomaly VAE latent codec with two-dimensional patch tokens.

This codec is designed for unregistered industrial anomaly datasets where the
normal reference and anomaly target may share a category/view but not exact
pixel alignment.  It therefore avoids the subtraction target
``z_anomaly - z_normal`` and predicts the complete anomaly latent instead::

    z_anomaly = VAE.encode(x_anomaly)
    grid = concat([mask_SDF, z_anomaly])             # (B, 5, h, w)
    chunk = patchify_2d(grid, patch_size=P)          # (B, T, D)

For the default 256 px image, factor-8 VAE and P=4::

    grid  = (B, 5, 32, 32)
    chunk = (B, 64, 80)       # 8 x 8 tokens, 5 x 4 x 4 values/token

Decoding directly reconstructs the predicted full anomaly latent::

    x_syn = VAE.decode(z_anomaly_pred)

The target normal image remains part of the RDT conditioning stream, but it is
not subtracted from the target and is not required by the decoder.  This makes
the supervision well-defined under ``same_category_same_view_reference``
pairing, while identity preservation remains a learning/generalization issue
rather than a contaminated subtraction target.
"""

import torch
import torch.nn.functional as F

from rescene.codec.base import EditActionCodec, EditActionSpec
from rescene.codec.sdf import mask_to_sdf, sdf_to_mask


class VAEFullLatentPatchCodec(EditActionCodec):
    """Predict ``[mask SDF, full anomaly latent]`` as 2-D patch tokens."""

    content_label = "latent"

    def __init__(
        self,
        vae,
        image_size: int = 256,
        use_sdf: bool = True,
        patch_size: int = 4,
        latent_scale: float = 1.0,
        sdf_clip_frac: float = 0.25,
        composite_with_normal: bool = False,
    ):
        super().__init__()
        if latent_scale <= 0:
            raise ValueError(f"latent_scale must be positive, received {latent_scale}.")

        self.vae = vae
        self.image_size = int(image_size)
        self.use_sdf = bool(use_sdf)
        self.patch_size = int(patch_size)
        self.latent_scale = float(latent_scale)
        self.sdf_clip_frac = float(sdf_clip_frac)
        self.composite_with_normal = bool(composite_with_normal)

        factor = int(vae.downsample_factor)
        if self.image_size % factor != 0:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by the VAE "
                f"downsample factor={factor}."
            )
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
    def _mask_grid(self, mask: torch.Tensor) -> torch.Tensor:
        """Resize a pixel mask and convert it to SDF (or binary [-1,1])."""
        low = F.interpolate(
            mask.float(),
            size=(self.spec.grid_h, self.spec.grid_w),
            mode="area",
        )
        # Area pooling otherwise erases thin scratches at factor-8 resolution.
        binary = (low > 0.2).float()
        if self.use_sdf:
            return mask_to_sdf(binary, clip_frac=self.sdf_clip_frac)
        return binary * 2.0 - 1.0

    def mask_grid_to_pixel_mask(self, mask_grid: torch.Tensor) -> torch.Tensor:
        binary = sdf_to_mask(mask_grid) if self.use_sdf else (mask_grid > 0.0).float()
        return F.interpolate(
            binary,
            size=(self.image_size, self.image_size),
            mode="nearest",
        )

    def encode_images(self, img01: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(img01)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_target(self, normal_img, anomaly_img, mask) -> torch.Tensor:
        # ``normal_img`` is intentionally unused.  It remains a condition for
        # ReSCENERDT, but no unregistered latent subtraction is performed.
        del normal_img
        z_anomaly = self.vae.encode(anomaly_img) * self.latent_scale
        mask_grid = self._mask_grid(mask).to(z_anomaly.device)
        grid = torch.cat([mask_grid, z_anomaly], dim=1)
        return self.grid_to_chunk(self._normalize(grid))

    @torch.no_grad()
    def decode(self, chunk, normal_img, local_residual=None) -> dict:
        grid = self._denormalize(self.chunk_to_grid(chunk.float()))
        mask_grid = grid[:, :1]
        z_anomaly = grid[:, 1:] / self.latent_scale
        mask_pix = self.mask_grid_to_pixel_mask(mask_grid)
        image = self.vae.decode(z_anomaly)

        # Optional editing-style compositing is disabled by default.  It can be
        # enabled as an ablation without changing the training target.
        if self.composite_with_normal:
            normal = normal_img.float().to(image.device)
            gate = mask_pix.to(image.device)
            image = normal * (1.0 - gate) + image * gate

        if local_residual is not None:
            image = (
                image
                + mask_pix.to(image.device) * local_residual.to(image.device)
            ).clamp(0, 1)

        return {
            "image": image.clamp(0, 1),
            "mask": mask_pix,
            "mask_grid": mask_grid,
            "content_grid": z_anomaly,  # backward-compatible generic key
            "latent_grid": z_anomaly,
            "z_anomaly_pred": z_anomaly,
            "z_syn": z_anomaly,         # compatibility with latent-delta tooling
        }
