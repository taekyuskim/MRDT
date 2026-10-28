import pytest
from pathlib import Path

import numpy as np

from plan_b.mvimgnet.cameras import (
    latent_grid_to_patch_tokens,
    letterbox_transform,
    patch_tokens_to_latent_grid,
    relative_world_to_camera,
)
from plan_b.mvimgnet.colmap_io import Image as ColmapImage
from plan_b.mvimgnet.colmap_io import qvec_to_rotmat


def test_quaternion_identity_and_noncontiguous_relative_ids():
    r = qvec_to_rotmat(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(r, np.eye(3))
    images = {
        101: ColmapImage(101, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 3, "001.jpg", np.zeros((0, 2)), np.zeros((0,), dtype=np.int64)),
        999: ColmapImage(999, np.array([1.0, 0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), 7, "009.jpg", np.zeros((0, 2)), np.zeros((0,), dtype=np.int64)),
    }
    rel = relative_world_to_camera(images, [101, 999], 101)
    assert np.allclose(rel[101], np.eye(4))
    assert np.allclose(rel[999][:3, 3], [2.0, 0.0, 0.0])


def test_letterbox_center_projection():
    a, meta = letterbox_transform(640, 480, 256)
    assert meta["resized_width"] == 256
    assert meta["resized_height"] == 192
    center = a @ np.array([320.0, 240.0, 1.0])
    assert np.allclose(center[:2], [128.0, 128.0])


def test_latent_patch_round_trip():
    latent = np.arange(4 * 32 * 32, dtype=np.float32).reshape(4, 32, 32)
    tokens = latent_grid_to_patch_tokens(latent)
    assert tokens.shape == (64, 64)
    recovered = patch_tokens_to_latent_grid(tokens)
    assert np.array_equal(recovered, latent)


def test_real_artifacts_exist_after_prepare_if_run():
    processed = Path("/data1/haksoo/data/MVImgNet2.0/processed_plan_b")
    if not (processed / "manifests" / "preview_episodes.jsonl").exists():
        pytest.skip("original-server pilot data not available")
    assert (processed / "instances.jsonl").exists()
    assert (processed / "split_assignment.jsonl").exists()
