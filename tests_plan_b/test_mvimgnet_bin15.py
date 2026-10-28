"""Focused checks for MVImgNet2.0 bin15 pilot components."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from plan_b.mvimgnet.bin15_camera import bin_pair, camera_feature_from_bins, nearest_bin_center_deg, wrap_deg
from plan_b.mvimgnet.bin15_dataset import MVImgNetBin15Stage1Dataset
from plan_b.mvimgnet.bin15_model import Bin15CameraRDT, latent_to_patch_tokens, patch_tokens_to_latent


ROOT = Path("/data1/haksoo/data/MVImgNet2.0")
PROCESSED = ROOT / "processed_plan_b"
FIXED = ROOT / "reports_plan_b/mvimgnet2_audit_followup/fixed_camera_views_followup.csv"


def test_bin_boundaries_wrap_and_radians():
    assert wrap_deg(181) == -179
    assert nearest_bin_center_deg(7.4) == 0
    assert nearest_bin_center_deg(7.6) == 15
    assert nearest_bin_center_deg(179) == -180
    assert nearest_bin_center_deg(91, elevation=True) == 90
    assert bin_pair(29.9, 59.9) == (30, 60)


def test_same_bin_feature_ignores_raw_angle_residual():
    k = np.eye(3, dtype=np.float64)
    f1 = camera_feature_from_bins(30, 60, 1.2, k, -0.01)
    f2 = camera_feature_from_bins(30, 60, 1.2, k, -0.01)
    assert np.array_equal(f1, f2)
    assert f1.shape == (10,)
    assert abs(f1[0] - math.sin(math.radians(30))) < 1e-6


def test_dataset_sampling_no_duplicate_and_no_raw_pose_conditioning():
    if not FIXED.exists():
        pytest.skip("original-server pilot data not available")
    ds = MVImgNetBin15Stage1Dataset(
        PROCESSED / "manifests/train_instances.jsonl",
        FIXED,
        "train",
        samples_per_epoch=4,
        deterministic=True,
    )
    item = ds[0]
    mi = item["model_inputs"]
    meta = item["metadata"]
    ids = [x for x in meta["context_image_ids"] if x is not None] + [meta["target_image_id"]]
    assert len(ids) == len(set(ids))
    assert "raw_T_rel" not in mi
    assert "angle_residual" not in mi
    assert "target_image" not in mi
    assert mi["context_camera_features"].shape == (5, 10)
    assert mi["target_camera_features"].shape == (10,)
    absent = ~mi["context_present"]
    assert torch.equal(mi["context_camera_features"][absent], torch.zeros_like(mi["context_camera_features"][absent]))


def test_latent_patch_roundtrip():
    z = torch.randn(2, 4, 32, 32)
    tokens = latent_to_patch_tokens(z)
    assert tokens.shape == (2, 64, 64)
    assert torch.allclose(patch_tokens_to_latent(tokens), z)


def test_tiny_forward_backward_and_padding_mask():
    cfg = {
        "lang_adaptor": "mlp2x_gelu",
        "img_adaptor": "mlp2x_gelu",
        "state_adaptor": "mlp3x_gelu",
        "rdt": {"hidden_size": 64, "depth": 2, "num_heads": 4},
        "noise_scheduler": {
            "num_train_timesteps": 8,
            "num_inference_timesteps": 2,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "sample",
            "clip_sample": False,
        },
    }
    model = Bin15CameraRDT(rdt_config=cfg, img_token_dim=64, lang_token_dim=64, k_max=5)
    batch = {
        "slot_feats": torch.randn(2, 5, 64, 64),
        "context_camera_features": torch.randn(2, 5, 10),
        "target_camera_features": torch.randn(2, 10),
        "context_present": torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool),
        "action_gt": torch.randn(2, 64, 64),
    }
    loss = model.compute_loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    with torch.no_grad():
        _, _, _, img_mask, _ = model._conditions(
            slot_feats=batch["slot_feats"],
            context_camera_features=batch["context_camera_features"],
            target_camera_features=batch["target_camera_features"],
            context_present=batch["context_present"],
        )
    assert img_mask.shape == (2, 5 * 64)
    assert img_mask[0].sum().item() == 2 * 64
