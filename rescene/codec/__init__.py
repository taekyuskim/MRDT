from rescene.codec.base import EditActionCodec, EditActionSpec
from rescene.codec.lowres_residual import LowResRGBResidualCodec
from rescene.codec.vae_full_latent_patch import VAEFullLatentPatchCodec
from rescene.codec.vae_masked_inpainting_patch import VAEMaskedInpaintingPatchCodec
from rescene.codec.vae_masked_inpainting_latent_only import VAEMaskedInpaintingLatentOnlyCodec
from rescene.codec.vae_latent_delta import VAELatentDeltaCodec, load_vae

__all__ = [
    "EditActionCodec",
    "EditActionSpec",
    "LowResRGBResidualCodec",
    "VAELatentDeltaCodec",
    "VAEFullLatentPatchCodec",
    "VAEMaskedInpaintingPatchCodec",
    "VAEMaskedInpaintingLatentOnlyCodec",
    "load_vae",
    "build_codec",
]


def build_codec(config: dict, device="cpu"):
    """Build a codec from ``configs/rescene/*.yaml`` key ``codec``."""
    kind = config.get("kind", "vae_latent_delta")

    if kind == "vae_latent_delta":
        vae = load_vae(
            config.get("vae_id", "stabilityai/sd-vae-ft-mse"),
            device=device,
        )
        return VAELatentDeltaCodec(
            vae,
            image_size=config.get("image_size", 256),
            use_sdf=config.get("use_sdf", True),
            delta_scale=config.get("delta_scale", 1.0),
            mask_gate_delta=config.get("mask_gate_delta", True),
            sdf_clip_frac=config.get("sdf_clip_frac", 0.25),
        )

    if kind == "vae_full_latent_patch":
        vae = load_vae(
            config.get("vae_id", "stabilityai/sd-vae-ft-mse"),
            device=device,
        )
        return VAEFullLatentPatchCodec(
            vae,
            image_size=config.get("image_size", 256),
            use_sdf=config.get("use_sdf", True),
            patch_size=config.get("patch_size", 4),
            latent_scale=config.get("latent_scale", 1.0),
            sdf_clip_frac=config.get("sdf_clip_frac", 0.25),
            composite_with_normal=config.get("composite_with_normal", False),
        )


    if kind == "vae_masked_inpainting_patch":
        vae = load_vae(
            config.get("vae_id", "stabilityai/sd-vae-ft-mse"),
            device=device,
        )
        return VAEMaskedInpaintingPatchCodec(
            vae,
            image_size=config.get("image_size", 256),
            use_sdf=config.get("use_sdf", True),
            patch_size=config.get("patch_size", 4),
            latent_scale=config.get("latent_scale", 1.0),
            sdf_clip_frac=config.get("sdf_clip_frac", 0.25),
            mask_dilation_latent=config.get("mask_dilation_latent", 1),
            condition_dilation_pixels=config.get("condition_dilation_pixels"),
            pixel_feather_radius=config.get("pixel_feather_radius", 2),
            boundary_weight=config.get("boundary_weight", 2.0),
            training_fill_modes=config.get(
                "training_fill_modes", ["noise", "mean", "zero"]
            ),
            eval_fill_mode=config.get("eval_fill_mode", "mean"),
            blur_kernel=config.get("blur_kernel", 21),
            mask_downsample=config.get("mask_downsample", "max"),
        )


    if kind == "vae_masked_inpainting_latent_only":
        vae = load_vae(
            config.get("vae_id", "stabilityai/sd-vae-ft-mse"),
            device=device,
        )
        return VAEMaskedInpaintingLatentOnlyCodec(
            vae,
            image_size=config.get("image_size", 256),
            patch_size=config.get("patch_size", 4),
            latent_scale=config.get("latent_scale", 1.0),
            sdf_clip_frac=config.get("sdf_clip_frac", 0.25),
            mask_dilation_latent=config.get("mask_dilation_latent", 1),
            condition_dilation_pixels=config.get("condition_dilation_pixels"),
            pixel_feather_radius=config.get("pixel_feather_radius", 0),
            boundary_weight=config.get("boundary_weight", 2.0),
            training_fill_modes=config.get(
                "training_fill_modes", ["noise", "mean", "zero"]
            ),
            eval_fill_mode=config.get("eval_fill_mode", "mean"),
            blur_kernel=config.get("blur_kernel", 21),
            mask_downsample=config.get("mask_downsample", "max"),
        )

    if kind == "lowres_residual":
        return LowResRGBResidualCodec(
            image_size=config.get("image_size", 128),
            grid_size=config.get("grid_size", 16),
            use_sdf=config.get("use_sdf", True),
            sdf_clip_frac=config.get("sdf_clip_frac", 0.25),
        )

    raise ValueError(
        f"Unknown codec kind: {kind!r}. Expected one of: "
        "'vae_latent_delta', 'vae_full_latent_patch', 'vae_masked_inpainting_patch', 'vae_masked_inpainting_latent_only', 'lowres_residual'."
    )
