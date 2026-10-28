"""VAE latent-delta edit-action codec (the default first-milestone codec).

    z_normal  = VAE.encode(x_normal)
    z_anomaly = VAE.encode(x_anomaly)
    latent_delta = z_anomaly - z_normal
    mask_grid = SDF(resize(mask, latent_res))       # or plain binary in [-1,1]
    edit_action = chunk(concat([mask_grid, latent_delta]))

Decoding:

    z_syn = VAE.encode(x_target_normal) + latent_delta
    x_syn = VAE.decode(z_syn)
    x_syn = x_syn + mask_pixel * local_residual     # optional correction branch

The VAE is frozen and pretrained (default: stabilityai/sd-vae-ft-mse). A
network-free ``debug`` VAE (blurry 8x avg-pool / bilinear round trip) is
provided so the full pipeline can run offline; it is for pipeline debugging
only, not for quality evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rescene.codec.base import EditActionCodec, EditActionSpec
from rescene.codec.sdf import mask_to_sdf, sdf_to_mask


# ---------------------------------------------------------------------------
# VAE wrappers: encode (B,3,H,W) [0,1] -> (B,C,h,w); decode inverse.
# ---------------------------------------------------------------------------
def _load_autoencoder_kl(vae_id: str):
    """Load an AutoencoderKL without memory-mapping the safetensors file.

    ``safetensors.torch.load_file`` mmaps the checkpoint; on machines under
    heavy memory pressure (and on some Windows setups) reading the mapped
    pages can hard-crash the process (access violation) instead of raising.
    We therefore read the bytes explicitly by default and fall back to the
    standard ``from_pretrained`` for anything the manual path cannot handle
    (sharded checkpoints, subfolders, etc.).
    """
    from diffusers import AutoencoderKL
    try:
        from huggingface_hub import hf_hub_download
        from pathlib import Path as _P

        if _P(vae_id).is_dir():
            cfg_path = _P(vae_id) / "config.json"
            st_path = _P(vae_id) / "diffusion_pytorch_model.safetensors"
            bin_path = _P(vae_id) / "diffusion_pytorch_model.bin"
        else:
            cfg_path = hf_hub_download(vae_id, "config.json")
            try:
                st_path = _P(hf_hub_download(vae_id,
                                             "diffusion_pytorch_model.safetensors"))
                bin_path = None
            except Exception:
                st_path = None
                bin_path = _P(hf_hub_download(vae_id, "diffusion_pytorch_model.bin"))
        config = AutoencoderKL.load_config(str(_P(cfg_path).parent))
        model = AutoencoderKL.from_config(config)
        if st_path is not None and _P(st_path).is_file():
            import safetensors.torch
            with open(st_path, "rb") as f:
                state = safetensors.torch.load(f.read())
        else:
            state = torch.load(bin_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model
    except Exception as e:
        print(f"[vae] no-mmap load failed ({type(e).__name__}: {e}); "
              f"falling back to AutoencoderKL.from_pretrained")
        return AutoencoderKL.from_pretrained(vae_id)


class DiffusersVAE(nn.Module):
    def __init__(self, vae_id: str, device="cpu"):
        super().__init__()
        self.vae = _load_autoencoder_kl(vae_id).to(device).eval()
        self.vae.requires_grad_(False)
        self.vae_id = vae_id
        self.latent_channels = self.vae.config.latent_channels
        self.downsample_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)

    @property
    def device(self):
        return next(self.vae.parameters()).device

    @torch.no_grad()
    def encode(self, img01: torch.Tensor) -> torch.Tensor:
        x = img01.to(self.device) * 2.0 - 1.0
        posterior = self.vae.encode(x).latent_dist
        return posterior.mode() * self.scaling_factor

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.vae.decode(z.to(self.device) / self.scaling_factor).sample
        return ((x + 1.0) / 2.0).clamp(0, 1)


class DebugVAE(nn.Module):
    """Network-free stand-in with the same API (4 latent channels, factor 8).

    encode: channels 0-2 = 8x average-pooled RGB (mapped to [-1,1]),
            channel 3   = local high-frequency energy.
    decode: bilinear upsampling of channels 0-2. Blurry by construction.
    """
    latent_channels = 4
    downsample_factor = 8
    scaling_factor = 1.0
    vae_id = "debug"

    def __init__(self, device="cpu"):
        super().__init__()
        self._device = torch.device(device)

    @property
    def device(self):
        return self._device

    def to(self, device):
        self._device = torch.device(device)
        return self

    @torch.no_grad()
    def encode(self, img01: torch.Tensor) -> torch.Tensor:
        x = img01.to(self._device).float()
        f = self.downsample_factor
        low = F.avg_pool2d(x, f)                      # (B,3,h,w) in [0,1]
        up = F.interpolate(low, scale_factor=f, mode="bilinear", align_corners=False)
        hf = (x - up).abs().mean(dim=1, keepdim=True)
        hf = F.avg_pool2d(hf, f) * 8.0                # rough high-freq energy
        return torch.cat([low * 2.0 - 1.0, hf], dim=1)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        low = (z[:, :3].to(self._device) + 1.0) / 2.0
        f = self.downsample_factor
        x = F.interpolate(low, scale_factor=f, mode="bilinear", align_corners=False)
        return x.clamp(0, 1)


def load_vae(vae_id: str, device="cpu"):
    """vae_id: HF model id / local path of an AutoencoderKL, or 'debug'."""
    if vae_id == "debug":
        return DebugVAE(device=device)
    return DiffusersVAE(vae_id, device=device)


# ---------------------------------------------------------------------------
class VAELatentDeltaCodec(EditActionCodec):
    content_label = "delta"

    def __init__(self, vae, image_size: int = 256, use_sdf: bool = True,
                 delta_scale: float = 1.0, mask_gate_delta: bool = True,
                 sdf_clip_frac: float = 0.25):
        super().__init__()
        self.vae = vae
        self.image_size = image_size
        self.use_sdf = use_sdf
        self.delta_scale = delta_scale
        self.mask_gate_delta = mask_gate_delta
        self.sdf_clip_frac = sdf_clip_frac
        f = vae.downsample_factor
        assert image_size % f == 0, (
            f"image_size {image_size} must be divisible by the VAE downsample "
            f"factor {f}")
        h = image_size // f
        self.spec = EditActionSpec(grid_channels=1 + vae.latent_channels,
                                   grid_h=h, grid_w=h, mask_channels=1)

    # -- helpers ---------------------------------------------------------------
    def _mask_grid(self, mask: torch.Tensor) -> torch.Tensor:
        """(B,1,H,W) {0,1} -> (B,1,h,w) SDF in [-1,1] (or binary in [-1,1])."""
        low = F.interpolate(mask.float(), size=(self.spec.grid_h, self.spec.grid_w),
                            mode="area")
        binary = (low > 0.2).float()  # keep thin structures alive after area pooling
        if self.use_sdf:
            return mask_to_sdf(binary, clip_frac=self.sdf_clip_frac)
        return binary * 2.0 - 1.0

    def mask_grid_to_pixel_mask(self, mask_grid: torch.Tensor) -> torch.Tensor:
        binary = sdf_to_mask(mask_grid) if self.use_sdf else (mask_grid > 0.0).float()
        return F.interpolate(binary, size=(self.image_size, self.image_size),
                             mode="nearest")

    def encode_images(self, img01: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(img01)

    # -- codec API ----------------------------------------------------------
    @torch.no_grad()
    def encode_target(self, normal_img, anomaly_img, mask) -> torch.Tensor:
        z_n = self.vae.encode(normal_img)
        z_a = self.vae.encode(anomaly_img)
        delta = (z_a - z_n) * self.delta_scale
        mask_grid = self._mask_grid(mask).to(delta.device)
        grid = torch.cat([mask_grid, delta], dim=1)
        return self.grid_to_chunk(self._normalize(grid))

    @torch.no_grad()
    def decode(self, chunk, normal_img, local_residual=None) -> dict:
        grid = self._denormalize(self.chunk_to_grid(chunk.float()))
        mask_grid = grid[:, :1]
        delta = grid[:, 1:] / self.delta_scale
        mask_pix = self.mask_grid_to_pixel_mask(mask_grid)

        if self.mask_gate_delta:
            gate = sdf_to_mask(mask_grid, threshold=-0.1) if self.use_sdf \
                else (mask_grid > -0.5).float()
            delta = delta * gate

        z_n = self.vae.encode(normal_img)
        z_syn = z_n + delta.to(z_n.device)
        image = self.vae.decode(z_syn)

        if local_residual is not None:
            image = (image + mask_pix.to(image.device) *
                     local_residual.to(image.device)).clamp(0, 1)
        return {
            "image": image,
            "mask": mask_pix,
            "mask_grid": mask_grid,
            "content_grid": delta,
            "z_normal": z_n,
            "z_syn": z_syn,
        }
