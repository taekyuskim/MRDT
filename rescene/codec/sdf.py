"""Mask <-> signed distance field conversion (SDF normalized to [-1, 1],
positive inside the mask)."""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def mask_to_sdf(mask: torch.Tensor, clip_frac: float = 0.25) -> torch.Tensor:
    """mask: (B, 1, H, W) float in {0,1} -> SDF (B, 1, H, W) in [-1, 1].

    ``clip_frac``: distances are normalized by clip_frac * max(H, W) and clamped.
    An empty mask maps to constant -1; a full mask to +1.
    """
    B, _, H, W = mask.shape
    clip = max(clip_frac * max(H, W), 1.0)
    out = torch.empty_like(mask, dtype=torch.float32, device="cpu")
    mask_np = (mask.detach().cpu().numpy() > 0.5)
    for b in range(B):
        fg = mask_np[b, 0]
        if not fg.any():
            out[b, 0] = -1.0
            continue
        if fg.all():
            out[b, 0] = 1.0
            continue
        d_in = distance_transform_edt(fg)
        d_out = distance_transform_edt(~fg)
        sdf = (d_in - d_out) / clip
        out[b, 0] = torch.from_numpy(np.clip(sdf, -1.0, 1.0)).float()
    return out.to(mask.device)


def sdf_to_mask(sdf: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """SDF (B, 1, H, W) -> binary mask (float {0,1})."""
    return (sdf > threshold).float()
