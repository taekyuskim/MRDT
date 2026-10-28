"""Low-resolution mask + RGB residual codec (alternative debug format).

No pretrained VAE needed. The edit action is:

    mask_grid = SDF(resize(mask, R x R))
    residual  = resize(mask * (anomaly - normal), R x R)      # in [-1, 1]
    edit_action = chunk(concat([mask_grid, residual]))

Decoding upsamples the residual and adds it inside the predicted mask.
Useful for fully offline smoke tests and as a sanity baseline.
"""

import torch
import torch.nn.functional as F

from rescene.codec.base import EditActionCodec, EditActionSpec
from rescene.codec.sdf import mask_to_sdf, sdf_to_mask


class LowResRGBResidualCodec(EditActionCodec):
    content_label = "residual"

    def __init__(self, image_size: int = 128, grid_size: int = 16,
                 use_sdf: bool = True, sdf_clip_frac: float = 0.25):
        super().__init__()
        self.image_size = image_size
        self.use_sdf = use_sdf
        self.sdf_clip_frac = sdf_clip_frac
        self.spec = EditActionSpec(grid_channels=4, grid_h=grid_size,
                                   grid_w=grid_size, mask_channels=1)

    def _mask_grid(self, mask: torch.Tensor) -> torch.Tensor:
        low = F.interpolate(mask.float(), size=(self.spec.grid_h, self.spec.grid_w),
                            mode="area")
        binary = (low > 0.2).float()
        if self.use_sdf:
            return mask_to_sdf(binary, clip_frac=self.sdf_clip_frac)
        return binary * 2.0 - 1.0

    @torch.no_grad()
    def encode_target(self, normal_img, anomaly_img, mask) -> torch.Tensor:
        size = (self.spec.grid_h, self.spec.grid_w)
        residual = mask.float() * (anomaly_img.float() - normal_img.float())
        residual_low = F.interpolate(residual, size=size, mode="area")
        mask_grid = self._mask_grid(mask).to(residual_low.device)
        grid = torch.cat([mask_grid, residual_low], dim=1)
        return self.grid_to_chunk(self._normalize(grid))

    @torch.no_grad()
    def decode(self, chunk, normal_img, local_residual=None) -> dict:
        grid = self._denormalize(self.chunk_to_grid(chunk.float()))
        mask_grid, residual_low = grid[:, :1], grid[:, 1:]
        binary_low = sdf_to_mask(mask_grid) if self.use_sdf else (mask_grid > 0).float()
        size = (self.image_size, self.image_size)
        mask_pix = F.interpolate(binary_low, size=size, mode="nearest")
        residual = F.interpolate(residual_low, size=size, mode="bilinear",
                                 align_corners=False)
        image = (normal_img.float().to(residual.device) + mask_pix * residual)
        if local_residual is not None:
            image = image + mask_pix * local_residual.to(image.device)
        return {
            "image": image.clamp(0, 1),
            "mask": mask_pix,
            "mask_grid": mask_grid,
            "content_grid": residual_low,
        }
