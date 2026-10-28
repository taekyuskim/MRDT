"""Camera, resize and latent-token helpers for MVImgNet2.0 episodes."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .colmap_io import Camera, Image, camera_center_world, world_to_camera_matrix


def camera_intrinsics_matrix(camera: Camera) -> np.ndarray:
    p = np.asarray(camera.params, dtype=np.float64)
    if camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    if camera.model in {"SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"}:
        if camera.model == "SIMPLE_RADIAL":
            f, cx, cy = p[:3]
            fx, fy = f, f
        else:
            fx, fy, cx, cy = p[:4]
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"Camera model {camera.model} is preserved but not training-ready for pinhole K")


def distortion_parameters(camera: Camera) -> np.ndarray:
    p = np.asarray(camera.params, dtype=np.float64)
    if camera.model in {"SIMPLE_PINHOLE", "PINHOLE"}:
        return np.zeros((0,), dtype=np.float64)
    if camera.model == "SIMPLE_RADIAL":
        return p[3:]
    if camera.model == "RADIAL":
        return p[4:]
    if camera.model in {"OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"}:
        return p[4:]
    return p[4:] if len(p) > 4 else np.zeros((0,), dtype=np.float64)


def letterbox_transform(width: int, height: int, out_size: int = 256) -> Tuple[np.ndarray, Dict[str, float]]:
    scale = min(out_size / float(width), out_size / float(height))
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    pad_x = (out_size - new_w) / 2.0
    pad_y = (out_size - new_h) / 2.0
    a = np.array([[scale, 0.0, pad_x], [0.0, scale, pad_y], [0.0, 0.0, 1.0]], dtype=np.float64)
    meta = {
        "source_width": int(width),
        "source_height": int(height),
        "output_size": int(out_size),
        "scale": float(scale),
        "resized_width": int(new_w),
        "resized_height": int(new_h),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
    }
    return a, meta


def processed_intrinsics(camera: Camera, actual_width: int, actual_height: int, out_size: int = 256):
    k_source = camera_intrinsics_matrix(camera)
    a, resize_meta = letterbox_transform(actual_width, actual_height, out_size)
    return a @ k_source, k_source, resize_meta


def relative_world_to_camera(images: Dict[int, Image], frame_ids: Iterable[int], reference_id: int) -> Dict[int, np.ndarray]:
    t_ref = world_to_camera_matrix(images[reference_id].qvec, images[reference_id].tvec)
    inv_ref = np.linalg.inv(t_ref)
    out = {}
    for image_id in frame_ids:
        out[image_id] = world_to_camera_matrix(images[image_id].qvec, images[image_id].tvec) @ inv_ref
    return out


def median_context_baseline(images: Dict[int, Image], context_ids: List[int], eps: float = 1e-8) -> float:
    centers = [camera_center_world(images[i].qvec, images[i].tvec) for i in context_ids]
    dists = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dists.append(float(np.linalg.norm(centers[i] - centers[j])))
    if not dists:
        return 0.0
    scale = float(np.median(dists))
    return scale if math.isfinite(scale) and scale > eps else 0.0


def normalize_relative_translation(t_rel: np.ndarray, scale: float) -> np.ndarray:
    out = np.array(t_rel, dtype=np.float64, copy=True)
    out[:3, 3] /= scale
    return out


def rotation_angle_degrees(t_rel: np.ndarray) -> float:
    r = np.asarray(t_rel[:3, :3], dtype=np.float64)
    trace = float(np.trace(r))
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return float(math.degrees(math.acos(cos_theta)))


def latent_grid_to_patch_tokens(latent: np.ndarray, patch_size: int = 4) -> np.ndarray:
    arr = np.asarray(latent)
    if arr.shape[-3:] != (4, 32, 32):
        raise ValueError(f"Expected latent [...,4,32,32], got {arr.shape}")
    leading = arr.shape[:-3]
    arr = arr.reshape(*leading, 4, 8, patch_size, 8, patch_size)
    arr = arr.transpose(*range(len(leading)), -4, -2, -5, -3, -1)
    return arr.reshape(*leading, 64, 64)


def patch_tokens_to_latent_grid(tokens: np.ndarray, patch_size: int = 4) -> np.ndarray:
    arr = np.asarray(tokens)
    if arr.shape[-2:] != (64, 64):
        raise ValueError(f"Expected tokens [...,64,64], got {arr.shape}")
    leading = arr.shape[:-2]
    arr = arr.reshape(*leading, 8, 8, 4, patch_size, patch_size)
    arr = arr.transpose(*range(len(leading)), -3, -5, -2, -4, -1)
    return arr.reshape(*leading, 4, 32, 32)
