"""Edit-action codec interface.

An edit-action chunk is the ReSCENE replacement of RDT's robot action chunk:
a continuous tensor of shape ``(chunk_len, chunk_dim)`` representing an
industrial anomaly target.  Codecs work internally on a channel grid
``(C, H, W)`` where channel 0 is the mask representation and the remaining
channels are the content representation.

Two reversible grid/token layouts are supported:

``row`` (legacy)
    One complete grid row is one RDT token::

        (B, C, H, W) <-> (B, T=H, D=C*W)

``patch2d``
    Each non-overlapping ``P x P`` spatial patch is one RDT token in raster
    order::

        (B, C, H, W) <-> (B, T=(H/P)*(W/P), D=C*P*P)

The patch layout retains an explicit two-dimensional token grid, allowing the
RDT action-token positional embedding to be replaced with a 2-D spatial
embedding while keeping the official sequence-based transformer backbone.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class EditActionSpec:
    """Shape and tokenization specification for one edit-action grid."""

    grid_channels: int   # C (mask channels + content channels)
    grid_h: int          # H
    grid_w: int          # W
    mask_channels: int = 1
    tokenization: str = "row"      # "row" | "patch2d"
    patch_size: int = 1             # used only by patch2d

    def __post_init__(self) -> None:
        if self.grid_channels <= 0 or self.grid_h <= 0 or self.grid_w <= 0:
            raise ValueError(
                "grid_channels/grid_h/grid_w must be positive, received "
                f"{self.grid_channels}/{self.grid_h}/{self.grid_w}."
            )
        if not 0 <= self.mask_channels <= self.grid_channels:
            raise ValueError(
                f"mask_channels={self.mask_channels} must be in "
                f"[0, {self.grid_channels}]."
            )
        if self.tokenization not in {"row", "patch2d"}:
            raise ValueError(
                f"Unknown tokenization {self.tokenization!r}; expected 'row' or 'patch2d'."
            )
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, received {self.patch_size}.")
        if self.tokenization == "patch2d":
            if self.grid_h % self.patch_size != 0 or self.grid_w % self.patch_size != 0:
                raise ValueError(
                    f"Grid {(self.grid_h, self.grid_w)} must be divisible by "
                    f"patch_size={self.patch_size}."
                )

    @property
    def chunk_len(self) -> int:   # RDT pred_horizon
        if self.tokenization == "row":
            return self.grid_h
        th, tw = self.token_grid_shape
        return th * tw

    @property
    def chunk_dim(self) -> int:   # RDT action_dim
        if self.tokenization == "row":
            return self.grid_w * self.grid_channels
        return self.grid_channels * self.patch_size * self.patch_size

    @property
    def content_channels(self) -> int:
        return self.grid_channels - self.mask_channels

    @property
    def token_grid_shape(self) -> Tuple[int, int]:
        """Spatial shape of action tokens before raster flattening."""
        if self.tokenization == "row":
            return self.grid_h, 1
        return self.grid_h // self.patch_size, self.grid_w // self.patch_size

    @property
    def is_spatial_patch_sequence(self) -> bool:
        return self.tokenization == "patch2d"


class EditActionCodec(ABC):
    """Batched codec interface.

    Images are ``(B, 3, H, W)`` in ``[0, 1]`` and masks are
    ``(B, 1, H, W)`` in ``{0, 1}``.  Codec implementations expose a
    ``content_label`` used only for diagnostics and visualization.
    """

    spec: EditActionSpec
    content_label: str = "content"

    def __init__(self):
        # Per-channel normalization applied in grid space (identity by default).
        self._norm_mean = None
        self._norm_std = None

    # ---- normalization -----------------------------------------------------
    def set_normalization(self, mean, std) -> None:
        """Install per-grid-channel ``mean`` and ``std`` sequences."""
        self._norm_mean = torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        self._norm_std = (
            torch.as_tensor(std, dtype=torch.float32)
            .clamp_min(1e-4)
            .view(1, -1, 1, 1)
        )

    def normalization_state(self):
        if self._norm_mean is None:
            return None
        return {
            "mean": self._norm_mean.flatten().tolist(),
            "std": self._norm_std.flatten().tolist(),
        }

    def load_normalization_state(self, state) -> None:
        if state:
            self.set_normalization(state["mean"], state["std"])

    def _normalize(self, grid: torch.Tensor) -> torch.Tensor:
        if self._norm_mean is None:
            return grid
        return (grid - self._norm_mean.to(grid.device)) / self._norm_std.to(grid.device)

    def _denormalize(self, grid: torch.Tensor) -> torch.Tensor:
        if self._norm_mean is None:
            return grid
        return grid * self._norm_std.to(grid.device) + self._norm_mean.to(grid.device)

    # ---- grid <-> chunk ----------------------------------------------------
    def _validate_grid(self, grid: torch.Tensor) -> None:
        if grid.ndim != 4:
            raise ValueError(f"Expected grid (B,C,H,W), received {tuple(grid.shape)}.")
        _, c, h, w = grid.shape
        s = self.spec
        if (c, h, w) != (s.grid_channels, s.grid_h, s.grid_w):
            raise ValueError(
                f"Grid shape {tuple(grid.shape)} does not match spec "
                f"(C={s.grid_channels}, H={s.grid_h}, W={s.grid_w})."
            )

    def grid_to_chunk(self, grid: torch.Tensor) -> torch.Tensor:
        """Convert ``(B,C,H,W)`` into the configured RDT action sequence."""
        self._validate_grid(grid)
        b, c, h, w = grid.shape
        s = self.spec

        if s.tokenization == "row":
            return grid.permute(0, 2, 3, 1).reshape(b, h, w * c)

        p = s.patch_size
        token_h, token_w = s.token_grid_shape
        # (B,C,H,W) -> (B,C,token_h,token_w,P,P)
        patches = grid.unfold(2, p, p).unfold(3, p, p)
        # Raster-order tokens; each token contains channel-major P x P values.
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches.view(b, token_h * token_w, c * p * p)

    def chunk_to_grid(self, chunk: torch.Tensor) -> torch.Tensor:
        """Invert the configured RDT sequence back to ``(B,C,H,W)``."""
        if chunk.ndim != 3:
            raise ValueError(f"Expected chunk (B,T,D), received {tuple(chunk.shape)}.")
        s = self.spec
        b, t, d = chunk.shape
        if t != s.chunk_len or d != s.chunk_dim:
            raise ValueError(
                f"Chunk shape {tuple(chunk.shape)} does not match spec "
                f"(T={s.chunk_len}, D={s.chunk_dim})."
            )

        if s.tokenization == "row":
            return (
                chunk.reshape(b, s.grid_h, s.grid_w, s.grid_channels)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        p = s.patch_size
        token_h, token_w = s.token_grid_shape
        grid = chunk.view(b, token_h, token_w, s.grid_channels, p, p)
        # (B,token_h,token_w,C,P,P) -> (B,C,token_h,P,token_w,P)
        grid = grid.permute(0, 3, 1, 4, 2, 5).contiguous()
        return grid.view(b, s.grid_channels, s.grid_h, s.grid_w)

    # ---- codec API ---------------------------------------------------------
    @abstractmethod
    def encode_target(
        self,
        normal_img: torch.Tensor,
        anomaly_img: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build the normalized ground-truth edit-action chunk ``(B,T,D)``."""

    @abstractmethod
    def decode(
        self,
        chunk: torch.Tensor,
        normal_img: torch.Tensor,
        local_residual: torch.Tensor = None,
    ) -> dict:
        """Decode a normalized chunk.

        Returns a dict containing at least:

        ``image``
            ``(B,3,H,W)`` synthesized anomaly image in ``[0,1]``.
        ``mask``
            ``(B,1,H,W)`` predicted binary anomaly mask.
        ``mask_grid``
            ``(B,1,h,w)`` raw mask channel at codec grid resolution.
        ``content_grid``
            ``(B,C-1,h,w)`` raw content channels at codec grid resolution.
        """
