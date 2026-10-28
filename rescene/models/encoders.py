"""Frozen text / image encoders for ReSCENE conditioning.

Two families per modality:

* the official RDT path (T5-XXL text encoder, SigLIP vision tower) — used when
  the checkpoints are available;
* deterministic ``debug`` encoders that need no downloads — used for offline
  smoke tests and tiny overfitting runs. They are frozen random projections,
  so different texts/images still get distinguishable embeddings.

Both families expose the same API:

    text:  encode(list[str])            -> (B, L, dim) float32, (B, L) bool mask
    image: encode((B, 3, H, W) in [0,1]) -> (B, P, dim) float32
"""

import contextlib
import hashlib
import sys
import types
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# make the official RDT modules importable when running from the repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ===========================================================================
# Text encoders
# ===========================================================================
class DebugTextEncoder(nn.Module):
    """Hash-based bag-of-token embedding. Deterministic, frozen, offline."""

    def __init__(self, dim: int = 64, vocab_size: int = 4096, max_length: int = 32,
                 device="cpu", seed: int = 1234):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        g = torch.Generator().manual_seed(seed)
        table = torch.randn(vocab_size, dim, generator=g) / (dim ** 0.5)
        self.register_buffer("table", table)
        self.vocab_size = vocab_size
        self.to(device)

    def _token_ids(self, text: str) -> List[int]:
        tokens = "".join(c if c.isalnum() else " " for c in text.lower()).split()
        tokens = tokens[: self.max_length] or ["<empty>"]
        return [int(hashlib.sha1(t.encode()).hexdigest()[:8], 16) % self.vocab_size
                for t in tokens]

    @torch.no_grad()
    def encode(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = [self._token_ids(t) for t in texts]
        L = max(len(i) for i in ids)
        device = self.table.device
        emb = torch.zeros(len(texts), L, self.dim, device=device)
        mask = torch.zeros(len(texts), L, dtype=torch.bool, device=device)
        for b, seq in enumerate(ids):
            # add a positional rotation so word order matters slightly
            for j, tid in enumerate(seq):
                emb[b, j] = torch.roll(self.table[tid], shifts=j % self.dim, dims=0)
            mask[b, : len(seq)] = True
        return emb, mask


class T5TextEncoder(nn.Module):
    """Wraps the official RDT T5-XXL embedder (frozen)."""

    def __init__(self, model_id: str = "google/t5-v1_1-xxl", device="cpu",
                 max_length: int = 120, dtype=torch.bfloat16):
        super().__init__()
        from models.multimodal_encoder.t5_encoder import T5Embedder
        # the official embedder asserts on a fixed model list; allow smaller
        # T5 variants (e.g. t5-base) for cheap validation runs
        if model_id not in T5Embedder.available_models:
            T5Embedder.available_models = T5Embedder.available_models + [model_id]
        self.embedder = T5Embedder(device=device, from_pretrained=model_id,
                                   model_max_length=max_length, torch_dtype=dtype)
        self.dim = self.embedder.model.config.d_model
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        emb, attn = self.embedder.get_text_embeddings(list(texts))
        return emb.float(), attn.bool()


def build_text_encoder(name: str, device="cpu", t5_model: str = "google/t5-v1_1-xxl",
                       max_length: int = 120, dim: int = 64):
    if name == "debug":
        return DebugTextEncoder(dim=dim, max_length=max_length, device=device)
    if name == "t5":
        return T5TextEncoder(model_id=t5_model, device=device, max_length=max_length)
    raise ValueError(f"Unknown text encoder {name!r} (expected 't5' or 'debug')")


# ===========================================================================
# Image encoders
# ===========================================================================
class DebugImageEncoder(nn.Module):
    """Frozen random patch projection: resize to a small grid of patches and
    project raw pixels with a fixed random matrix. Deterministic, offline."""

    def __init__(self, dim: int = 64, grid: int = 4, patch_px: int = 16,
                 device="cpu", seed: int = 4321):
        super().__init__()
        self.dim = dim
        self.grid = grid
        self.patch_px = patch_px
        self.tokens_per_image = grid * grid
        g = torch.Generator().manual_seed(seed)
        in_dim = 3 * patch_px * patch_px
        proj = torch.randn(in_dim, dim, generator=g) / (in_dim ** 0.5)
        self.register_buffer("proj", proj)
        self.to(device)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x = images.to(self.proj.device).float()
        side = self.grid * self.patch_px
        x = F.interpolate(x, size=(side, side), mode="bilinear", align_corners=False)
        x = x * 2.0 - 1.0
        # (B, 3, G*p, G*p) -> (B, G*G, 3*p*p)
        patches = F.unfold(x, kernel_size=self.patch_px, stride=self.patch_px)
        patches = patches.transpose(1, 2)  # (B, G*G, 3*p*p)
        return patches @ self.proj          # (B, P, dim)


def _siglip_sdpa_attention(self, hidden_states, attention_mask=None, output_attentions=False):
    """SiglipAttention.forward (transformers 4.41, eager) via F.scaled_dot_product_attention,
    which uses flash / memory-efficient kernels for bf16 inputs. Same math, no attention probs."""
    b, n, _ = hidden_states.size()
    q, k, v = (proj(hidden_states).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
               for proj in (self.q_proj, self.k_proj, self.v_proj))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, scale=self.scale)
    return self.out_proj(out.transpose(1, 2).reshape(b, n, self.embed_dim)), None


class SiglipImageEncoder(nn.Module):
    """Wraps the official RDT SigLIP vision tower (frozen); pools the 27x27
    patch grid down to ``pool_grid`` x ``pool_grid`` tokens."""

    # "fp32": unchanged. "bf16": tower under bf16 autocast (weights, LayerNorm, softmax and
    # residual stream stay fp32) with SDPA attention.
    precision = "fp32"

    def set_precision(self, precision: str) -> None:
        if precision not in ("fp32", "bf16"):
            raise ValueError(f"Unsupported precision={precision!r}")
        self.precision = precision
        for module in self.tower.vision_tower.modules():
            if type(module).__name__ == "SiglipAttention":
                if precision == "bf16":
                    module.forward = types.MethodType(_siglip_sdpa_attention, module)
                elif "forward" in module.__dict__:
                    del module.forward  # back to the class (eager) implementation

    def __init__(self, model_id: str = "google/siglip-so400m-patch14-384",
                 device="cpu", pool_grid: int = 8, dtype=torch.float32):
        super().__init__()
        from types import SimpleNamespace
        from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
        self.tower = SiglipVisionTower(
            vision_tower=model_id,
            args=SimpleNamespace(mm_vision_select_feature="patch"))
        self.tower.vision_tower.to(device=device, dtype=dtype)
        self.tower.vision_tower.requires_grad_(False)
        self.pool_grid = pool_grid
        self.tokens_per_image = pool_grid * pool_grid
        self.dim = self.tower.hidden_size
        self.input_size = self.tower.config.image_size
        proc = self.tower.image_processor
        self.register_buffer(
            "mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1).to(device))
        self.register_buffer(
            "std", torch.tensor(proc.image_std).view(1, 3, 1, 1).to(device))

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x = images.to(self.mean.device).float()
        x = F.interpolate(x, size=(self.input_size, self.input_size),
                          mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        with (torch.autocast("cuda", dtype=torch.bfloat16) if self.precision == "bf16" else contextlib.nullcontext()):
            feats = self.tower(x)                 # (B, N, dim), N = g*g; returned in x's dtype (fp32)
        B, N, D = feats.shape
        g = int(round(N ** 0.5))
        assert g * g == N, f"non-square SigLIP token count: {N}"
        if g != self.pool_grid:
            grid = feats.transpose(1, 2).reshape(B, D, g, g)
            grid = F.adaptive_avg_pool2d(grid.float(), self.pool_grid)
            feats = grid.reshape(B, D, -1).transpose(1, 2)
        return feats.float()


def build_image_encoder(name: str, device="cpu",
                        siglip_model: str = "google/siglip-so400m-patch14-384",
                        tokens_per_image: int = 64, dim: int = 64):
    if name == "debug":
        grid = int(round(tokens_per_image ** 0.5))
        assert grid * grid == tokens_per_image, \
            f"tokens_per_image must be a square number, got {tokens_per_image}"
        return DebugImageEncoder(dim=dim, grid=grid, device=device)
    if name == "siglip":
        grid = int(round(tokens_per_image ** 0.5))
        assert grid * grid == tokens_per_image, \
            f"tokens_per_image must be a square number, got {tokens_per_image}"
        return SiglipImageEncoder(model_id=siglip_model, device=device, pool_grid=grid)
    raise ValueError(f"Unknown image encoder {name!r} (expected 'siglip' or 'debug')")
