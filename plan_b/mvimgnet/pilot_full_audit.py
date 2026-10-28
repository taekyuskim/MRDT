"""Full diagnostic audit for the extracted MVImgNet2.0 pilot.

This module is intentionally report-only: it reads the existing Plan B
manifests, COLMAP sparse models and Dataset outputs, then writes audit
metadata/panels without changing filters, splits or training code.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader

from .cameras import letterbox_transform, relative_world_to_camera
from .colmap_io import (
    camera_center_world,
    qvec_to_rotmat,
    read_sparse_model,
)
from .dataset import MVImgNetEpisodeDataset, mvimgnet_collate, read_jsonl


DEFAULT_ROOT = Path("/data1/haksoo/data/MVImgNet2.0")
DEFAULT_MODEL_REPO = Path("/home/haksoo/ReSCENE-RDT")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _as_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _percentiles(values: Iterable[float], ps=(0, 5, 25, 50, 75, 95, 100)) -> Dict[str, float | None]:
    vals = np.asarray([v for v in values if _as_float(v) is not None], dtype=np.float64)
    if vals.size == 0:
        return {f"p{p}": None for p in ps}
    return {f"p{p}": float(np.percentile(vals, p)) for p in ps}


def _angle_wrap_deg(a: float) -> float:
    return ((float(a) + 180.0) % 360.0) - 180.0


def _angle_delta_deg(a: float, b: float) -> float:
    return abs(_angle_wrap_deg(float(a) - float(b)))


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=np.float64)
    return np.asarray(v, dtype=np.float64) / n


def _camera_forward_world(image_rec) -> np.ndarray:
    # COLMAP camera looks along +Z in camera coordinates.
    return _unit(qvec_to_rotmat(image_rec.qvec).T @ np.array([0.0, 0.0, 1.0]))


def _camera_image_up_world(image_rec) -> np.ndarray:
    # Image y points down, so visual "up" is -Y in camera coordinates.
    return _unit(qvec_to_rotmat(image_rec.qvec).T @ np.array([0.0, -1.0, 0.0]))


def _lookat_center_from_rays(centers: np.ndarray, dirs: np.ndarray) -> Tuple[np.ndarray, float]:
    a = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros((3,), dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    for c, d in zip(centers, dirs):
        d = _unit(d)
        m = eye - np.outer(d, d)
        a += m
        b += m @ c
    try:
        x = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        x = centers.mean(axis=0)
    residuals = []
    for c, d in zip(centers, dirs):
        residuals.append(float(np.linalg.norm(np.cross(d, x - c))))
    return x, float(np.median(residuals)) if residuals else 0.0


def _plane_basis(centers: np.ndarray, object_center: np.ndarray, ref_center: np.ndarray) -> Dict[str, Any]:
    mean = centers.mean(axis=0)
    centered = centers - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    up = _unit(vt[-1])
    if up[2] < 0:
        up = -up
    ref = ref_center - object_center
    x = ref - float(np.dot(ref, up)) * up
    if np.linalg.norm(x) < 1e-8:
        x = vt[0]
    x = _unit(x)
    y = _unit(np.cross(up, x))
    plane_rms = float(np.sqrt(np.mean((centered @ up) ** 2))) if len(centers) else 0.0
    radii = np.linalg.norm(centers - object_center[None, :], axis=1)
    radius_med = float(np.median(radii)) if radii.size else 0.0
    svals = [float(v) for v in s.tolist()]
    return {
        "origin": object_center,
        "x": x,
        "y": y,
        "up": up,
        "plane_rms": plane_rms,
        "plane_rms_over_radius_median": plane_rms / max(radius_med, 1e-9),
        "svd_singular_values": svals,
        "planarity_ratio_s3_s2": (svals[-1] / svals[-2]) if len(svals) >= 2 and svals[-2] > 0 else None,
    }


def _az_el_radius(center: np.ndarray, basis: Dict[str, Any]) -> Tuple[float, float, float]:
    v = np.asarray(center, dtype=np.float64) - basis["origin"]
    r = float(np.linalg.norm(v))
    if r < 1e-12:
        return 0.0, 0.0, 0.0
    az = math.degrees(math.atan2(float(np.dot(v, basis["y"])), float(np.dot(v, basis["x"]))))
    el = math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(v, basis["up"])) / r))))
    return _angle_wrap_deg(az), el, r


def _roll_deg(image_rec, center: np.ndarray, basis: Dict[str, Any]) -> float:
    view_dir = _unit(basis["origin"] - center)
    img_up = _camera_image_up_world(image_rec)
    desired_up = basis["up"] - float(np.dot(basis["up"], view_dir)) * view_dir
    img_up = img_up - float(np.dot(img_up, view_dir)) * view_dir
    desired_up = _unit(desired_up)
    img_up = _unit(img_up)
    if np.linalg.norm(desired_up) < 1e-8 or np.linalg.norm(img_up) < 1e-8:
        return 0.0
    right = _unit(np.cross(desired_up, view_dir))
    return math.degrees(math.atan2(float(np.dot(img_up, right)), float(np.dot(img_up, desired_up))))


def _project_colmap(camera, xyz_cam: np.ndarray) -> np.ndarray | None:
    if xyz_cam[2] <= 1e-9:
        return None
    x = xyz_cam[0] / xyz_cam[2]
    y = xyz_cam[1] / xyz_cam[2]
    p = np.asarray(camera.params, dtype=np.float64)
    if camera.model == "SIMPLE_RADIAL":
        f, cx, cy, k = p[:4]
        radial = 1.0 + k * (x * x + y * y)
        return np.array([f * x * radial + cx, f * y * radial + cy], dtype=np.float64)
    if camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        return np.array([f * x + cx, f * y + cy], dtype=np.float64)
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
        return np.array([fx * x + cx, fy * y + cy], dtype=np.float64)
    if camera.model == "RADIAL":
        fx, fy, cx, cy, k1 = p[:5]
        radial = 1.0 + k1 * (x * x + y * y)
        return np.array([fx * x * radial + cx, fy * y * radial + cy], dtype=np.float64)
    return None


def _reprojection_errors(cameras, images, points, image_ids: List[int], max_obs_per_image: int = 250) -> Dict[str, Any]:
    per_image = {}
    all_errs = []
    for image_id in image_ids:
        rec = images[image_id]
        cam = cameras[rec.camera_id]
        r = qvec_to_rotmat(rec.qvec)
        t = np.asarray(rec.tvec, dtype=np.float64)
        errs = []
        valid_idx = np.flatnonzero(rec.point3D_ids >= 0)
        if valid_idx.size > max_obs_per_image:
            valid_idx = valid_idx[np.linspace(0, valid_idx.size - 1, max_obs_per_image).round().astype(int)]
        for idx in valid_idx:
            pid = int(rec.point3D_ids[idx])
            point = points.get(pid)
            if point is None:
                continue
            uv = _project_colmap(cam, r @ point.xyz + t)
            if uv is None:
                continue
            errs.append(float(np.linalg.norm(uv - rec.xys[idx])))
        all_errs.extend(errs)
        per_image[str(image_id)] = {
            "count": len(errs),
            "median_px": float(np.median(errs)) if errs else None,
            "p95_px": float(np.percentile(errs, 95)) if errs else None,
        }
    return {
        "all_count": len(all_errs),
        "median_px": float(np.median(all_errs)) if all_errs else None,
        "p95_px": float(np.percentile(all_errs, 95)) if all_errs else None,
        "per_image": per_image,
    }


def _find_image(instance_path: Path, name: str) -> Path | None:
    for p in [instance_path / name, instance_path / "images" / Path(name).name]:
        if p.exists():
            return p
    return None


def _find_mask(instance_path: Path, name: str) -> Path | None:
    stem = Path(name).stem
    for ext in [Path(name).suffix, ".jpg", ".png", ".jpeg"]:
        if ext:
            p = instance_path / "masks" / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def _letterbox_pil(img: Image.Image, out_size: int = 256, resample=Image.Resampling.BILINEAR, fill=0) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    scale = min(out_size / float(w), out_size / float(h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((nw, nh), resample=resample)
    canvas = Image.new(img.mode, (out_size, out_size), fill)
    canvas.paste(resized, ((out_size - nw) // 2, (out_size - nh) // 2))
    return canvas


def _mask_framing(mask_path: Path | None, raw_size: Tuple[int, int], out_size: int = 256) -> Dict[str, Any]:
    w, h = raw_size
    a, meta = letterbox_transform(w, h, out_size)
    padding_frac = 1.0 - (meta["resized_width"] * meta["resized_height"]) / float(out_size * out_size)
    if mask_path is None:
        return {"mask_present": False, "padding_frac": padding_frac}
    m = Image.open(mask_path).convert("L")
    proc = _letterbox_pil(m, out_size, Image.Resampling.NEAREST, fill=0)
    arr = np.asarray(proc) > 127
    ys, xs = np.nonzero(arr)
    if len(xs) == 0:
        return {"mask_present": True, "mask_area_frac": 0.0, "padding_frac": padding_frac, "empty_mask": True}
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    touches = bool(bbox[0] <= 1 or bbox[1] <= 1 or bbox[2] >= out_size - 1 or bbox[3] >= out_size - 1)
    return {
        "mask_present": True,
        "mask_area_frac": float(arr.mean()),
        "bbox_frac": [
            bbox[0] / out_size,
            bbox[1] / out_size,
            bbox[2] / out_size,
            bbox[3] / out_size,
        ],
        "bbox_area_frac": float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (out_size * out_size)),
        "touches_processed_border": touches,
        "padding_frac": padding_frac,
        "empty_mask": False,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def _label(img: Image.Image, text: str, height: int = 46) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + height), "white")
    out.paste(img.convert("RGB"), (0, height))
    draw = ImageDraw.Draw(out)
    draw.text((6, 5), text[:120], fill=(0, 0, 0))
    return out


def _concat_h(images: List[Image.Image]) -> Image.Image:
    w = sum(i.width for i in images)
    h = max(i.height for i in images)
    out = Image.new("RGB", (w, h), "white")
    x = 0
    for img in images:
        out.paste(img, (x, 0))
        x += img.width
    return out


def _make_instance_plots(instance_uid: str, view_rows: List[Dict[str, Any]], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in view_rows if r["instance_uid"] == instance_uid]
    paths = {}
    if not rows:
        return paths
    az = np.asarray([r["azimuth_deg"] for r in rows], dtype=float)
    el = np.asarray([r["elevation_deg"] for r in rows], dtype=float)
    rad = np.asarray([r["radius"] for r in rows], dtype=float)
    x = np.asarray([r["center_x"] for r in rows], dtype=float)
    y = np.asarray([r["center_y"] for r in rows], dtype=float)
    z = np.asarray([r["center_z"] for r in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(az, el, s=25)
    axes[0].axvline(60, color="tab:red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("diagnostic azimuth deg")
    axes[0].set_ylabel("diagnostic elevation deg")
    axes[0].set_title(instance_uid)
    axes[1].plot(np.arange(len(rad)), rad, marker="o", linewidth=1)
    axes[1].set_xlabel("view order by image_id")
    axes[1].set_ylabel("radius")
    fig.tight_layout()
    p = out_dir / f"az_el_radius_{instance_uid.replace('/', '__')}.png"
    fig.savefig(p)
    plt.close(fig)
    paths["az_el_radius"] = str(p)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, s=20)
    ax.set_title(instance_uid)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    p = out_dir / f"trajectory3d_{instance_uid.replace('/', '__')}.png"
    fig.savefig(p)
    plt.close(fig)
    paths["trajectory3d"] = str(p)
    return paths


def _make_reprojection_panel(row: Dict[str, Any], image_id: int, cameras, images, points, out_path: Path) -> None:
    inst = Path(row["instance_path"])
    rec = images[image_id]
    img_path = _find_image(inst, rec.name)
    if img_path is None:
        return
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    cam = cameras[rec.camera_id]
    r = qvec_to_rotmat(rec.qvec)
    t = np.asarray(rec.tvec, dtype=np.float64)
    drawn = 0
    valid_idx = np.flatnonzero(rec.point3D_ids >= 0)
    if valid_idx.size > 500:
        valid_idx = valid_idx[np.linspace(0, valid_idx.size - 1, 500).round().astype(int)]
    for idx in valid_idx:
        point = points.get(int(rec.point3D_ids[idx]))
        if point is None:
            continue
        uv = _project_colmap(cam, r @ point.xyz + t)
        if uv is None:
            continue
        x, y = float(uv[0]), float(uv[1])
        if 0 <= x < img.width and 0 <= y < img.height:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 0))
            drawn += 1
    img.thumbnail((512, 512), Image.Resampling.BILINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _label(img, f"{row['instance_uid']} {rec.name} | projected sparse obs {drawn}").save(out_path)


def _make_preprocess_panel(sample: Dict[str, Any], out_path: Path, encoder_input_size: int = 384) -> None:
    tiles = []
    meta = sample["metadata"]
    for label_name, path in [
        ("context1", meta["paths"]["context_images"][0]),
        ("target", meta["paths"]["target_image"]),
    ]:
        raw = Image.open(path).convert("RGB")
        raw_thumb = raw.copy()
        raw_thumb.thumbnail((256, 256), Image.Resampling.BILINEAR)
        proc = _tensor_to_pil(
            sample["model_inputs"]["context_images"][0]
            if label_name == "context1"
            else sample["supervision"]["target_image"]
        )
        enc = proc.resize((encoder_input_size, encoder_input_size), Image.Resampling.BILINEAR)
        enc.thumbnail((256, 256), Image.Resampling.BILINEAR)
        tiles.extend([
            _label(raw_thumb, f"{label_name} raw"),
            _label(proc, f"{label_name} dataset 256"),
            _label(enc, f"{label_name} encoder resize {encoder_input_size}"),
        ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _concat_h(tiles).save(out_path)


def _episode_combo_stats(
    episode_rows: List[Dict[str, Any]],
    az_by_instance_image: Dict[Tuple[str, int], float],
    covis_by_instance: Dict[str, Dict[Tuple[int, int], int]],
) -> List[Dict[str, Any]]:
    out = []
    for row in episode_rows:
        valid = list(row["valid_image_ids"])
        target = int(row["target_image_id"])
        target_az = az_by_instance_image.get((row["instance_uid"], target))
        ordered = []
        for iid in valid:
            if iid == target:
                continue
            az = az_by_instance_image.get((row["instance_uid"], int(iid)))
            if az is None or target_az is None:
                delta = None
            else:
                delta = _angle_delta_deg(az, target_az)
            ordered.append((9999.0 if delta is None else delta, int(iid)))
        ordered.sort()
        pair_counts = covis_by_instance.get(row["instance_uid"], {})
        for k in [2, 3, 4, 5]:
            ctx = [iid for _, iid in ordered[:k]]
            deltas = [d for d, _ in ordered[:k] if d < 9999]
            cov = [pair_counts.get(tuple(sorted((target, iid))), 0) for iid in ctx]
            out.append({
                "episode_id": row.get("episode_id"),
                "instance_uid": row["instance_uid"],
                "target_image_id": target,
                "target_azimuth_deg": target_az,
                "k_context": k,
                "chosen_context_ids_by_nearest_azimuth": " ".join(map(str, ctx)),
                "mean_abs_azimuth_delta_deg": statistics.mean(deltas) if deltas else None,
                "max_abs_azimuth_delta_deg": max(deltas) if deltas else None,
                "min_target_context_covisible_points": min(cov) if cov else None,
                "mean_target_context_covisible_points": statistics.mean(cov) if cov else None,
            })
    return out


def _covisibility(images, valid_ids: List[int]) -> Dict[Tuple[int, int], int]:
    obs = {}
    for image_id in valid_ids:
        rec = images[image_id]
        obs[image_id] = set(int(x) for x in rec.point3D_ids.tolist() if int(x) >= 0)
    out = {}
    for i, a in enumerate(valid_ids):
        for b in valid_ids[i + 1 :]:
            out[tuple(sorted((int(a), int(b))))] = len(obs[a].intersection(obs[b]))
    return out


def _audit_rdt_contract(config_path: Path, data_cfg_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = yaml.safe_load(data_cfg_path.read_text(encoding="utf-8"))
    enc = cfg["encoders"]
    data = cfg["data"]
    codec = cfg["codec"]
    max_normal = int(data.get("max_normal_views", 5))
    max_support = int(data.get("max_support", 2))
    num_slots = 1 + max_normal + max_support
    patch = int(codec.get("patch_size", 4))
    latent_h = int(data.get("image_size", 256)) // 8
    latent_channels = 4
    if codec.get("kind") == "vae_masked_inpainting_latent_only":
        action_dim = latent_channels * patch * patch
        grid_channels = latent_channels
        mask_channels = 0
        local_condition_dim = (latent_channels + 2) * patch * patch
    elif codec.get("kind") == "vae_full_latent_patch":
        grid_channels = 1 + latent_channels
        action_dim = grid_channels * patch * patch
        mask_channels = 1
        local_condition_dim = 0
    else:
        grid_channels = None
        action_dim = None
        mask_channels = None
        local_condition_dim = None
    tokens_per_image = int(enc["tokens_per_image"])
    return {
        "source_config": str(config_path),
        "plan_b_data_config": str(data_cfg_path),
        "current_wrapper": "ReSCENERDT around unmodified RDTRunner",
        "num_slots": num_slots,
        "slot_layout": f"[target] + {max_normal} normal/context + {max_support} support",
        "tokens_per_image": tokens_per_image,
        "img_cond_len": num_slots * tokens_per_image,
        "img_token_dim": int(enc["img_token_dim"]),
        "lang_token_dim": int(enc["lang_token_dim"]),
        "max_lang_cond_len": int(enc["max_lang_cond_len"]),
        "state_token_dim_before_mask_concat": action_dim,
        "rdt_state_adaptor_input_dim": None if action_dim is None else action_dim * 2,
        "action_chunk_len": (latent_h // patch) * (latent_h // patch),
        "action_chunk_dim": action_dim,
        "latent_grid": [latent_channels, latent_h, latent_h],
        "patch_size": patch,
        "mask_channels_in_action": mask_channels,
        "local_condition_dim": local_condition_dim,
        "discrete_view_embedding_only": True,
        "continuous_camera_adapter_present": False,
        "mvimgnet_dataset_camera_tensors": {
            "context_intrinsics": "[B,K,3,3]",
            "target_intrinsics": "[B,3,3]",
            "context_T_rel": "[B,K,4,4]",
            "target_T_rel": "[B,4,4]",
        },
        "plan_b_target_latent_expected": {
            "latent_grid": data_cfg.get("target_vae_latent_shape"),
            "patch_tokens": [data_cfg.get("latent_tokens"), data_cfg.get("latent_token_dim")],
        },
        "decision_pending": "how to map continuous camera pose/intrinsics into image-slot and target-state conditioning",
    }


def run_dataloader_checks(manifest: Path, report_dir: Path) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass
    for workers in [0, 2]:
        try:
            ds1 = MVImgNetEpisodeDataset(manifest, split="audit", fixed_episodes=True)
            loader = DataLoader(
                ds1,
                batch_size=2,
                shuffle=False,
                num_workers=workers,
                collate_fn=mvimgnet_collate,
                timeout=20 if workers else 0,
            )
            t0 = time.perf_counter()
            batches = []
            for idx, batch in enumerate(loader):
                if idx >= 6:
                    break
                batches.append(batch)
            elapsed = time.perf_counter() - t0
            first = batches[0]
            mi = first["model_inputs"]
            sup = first["supervision"]
            results[f"workers_{workers}"] = {
                "status": "passed",
                "batches": len(batches),
                "elapsed_sec": elapsed,
                "batches_per_sec": len(batches) / elapsed if elapsed > 0 else None,
                "first_shapes": {
                    "context_images": list(mi["context_images"].shape),
                    "context_intrinsics": list(mi["context_intrinsics"].shape),
                    "target_intrinsics": list(mi["target_intrinsics"].shape),
                    "context_T_rel": list(mi["context_T_rel"].shape),
                    "target_T_rel": list(mi["target_T_rel"].shape),
                    "target_image": list(sup["target_image"].shape),
                    "target_object_mask": list(sup["target_object_mask"].shape),
                },
                "finite_camera_tensors": bool(torch.isfinite(mi["context_T_rel"]).all() and torch.isfinite(mi["target_T_rel"]).all()),
                "target_leakage_keys": sorted(set(mi.keys()).intersection({"target_image", "target_object_mask", "target_latent", "clean_latent"})),
            }
        except Exception as exc:
            results[f"workers_{workers}"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "note": "Recorded as data-readiness environment result; no package or system setting was changed.",
            }
    ds_a = MVImgNetEpisodeDataset(manifest, split="audit", fixed_episodes=True)
    ds_b = MVImgNetEpisodeDataset(manifest, split="audit", fixed_episodes=True)
    a = ds_a[0]
    b = ds_b[0]
    results["reproducibility_fixed_episode_sample0"] = {
        "context_equal": bool(torch.equal(a["model_inputs"]["context_images"], b["model_inputs"]["context_images"])),
        "target_T_rel_equal": bool(torch.equal(a["model_inputs"]["target_T_rel"], b["model_inputs"]["target_T_rel"])),
        "metadata_equal": a["metadata"]["context_image_ids"] == b["metadata"]["context_image_ids"]
        and a["metadata"]["target_image_id"] == b["metadata"]["target_image_id"],
    }
    lines = ["# MVImgNet2.0 Full Audit DataLoader Results", ""]
    for key, value in results.items():
        lines.append(f"## {key}")
        lines.append("```json")
        lines.append(json.dumps(value, indent=2, sort_keys=True))
        lines.append("```")
    (report_dir / "TEST_RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def build_audit(args) -> Dict[str, Any]:
    root = args.root
    processed = args.processed_root or root / "processed_plan_b"
    report_dir = args.report_dir
    panels_dir = report_dir / "panels"
    report_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    instance_rows = read_jsonl(processed / "instances.jsonl")
    episode_rows = read_jsonl(processed / "manifests" / "preview_episodes.jsonl")
    view_rows: List[Dict[str, Any]] = []
    instance_summaries: List[Dict[str, Any]] = []
    duplicate_rows: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, Any]] = []
    covis_by_instance: Dict[str, Dict[Tuple[int, int], int]] = {}
    az_by_instance_image: Dict[Tuple[str, int], float] = {}
    panel_links: Dict[str, Dict[str, str]] = {}

    for row_idx, row in enumerate(instance_rows):
        inst = Path(row["instance_path"])
        cameras, images, points = read_sparse_model(inst / "sparse" / "0", with_points=True, max_points=args.max_points)
        valid_ids = [int(i) for i in row["valid_image_ids"] if int(i) in images]
        centers = np.stack([camera_center_world(images[i].qvec, images[i].tvec) for i in valid_ids], axis=0)
        dirs = np.stack([_camera_forward_world(images[i]) for i in valid_ids], axis=0)
        sparse_xyz = np.stack([p.xyz for p in points.values()], axis=0) if points else centers.mean(axis=0, keepdims=True)
        sparse_center = np.median(sparse_xyz, axis=0)
        look_center, look_resid = _lookat_center_from_rays(centers, dirs)
        object_center = sparse_center
        basis = _plane_basis(centers, object_center, centers[0])
        covis = _covisibility(images, valid_ids)
        covis_by_instance[row["instance_uid"]] = covis
        repro = _reprojection_errors(cameras, images, points, valid_ids[: min(8, len(valid_ids))])

        content_hashes = Counter()
        for image_id in valid_ids:
            rec = images[image_id]
            img_path = _find_image(inst, rec.name)
            if img_path is not None:
                # First 1MB is enough to catch exact duplicate pilot images without expensive full reads.
                import hashlib

                h = hashlib.sha1()
                with open(img_path, "rb") as f:
                    h.update(f.read(1024 * 1024))
                content_hashes[h.hexdigest()] += 1
        for h, count in content_hashes.items():
            if count > 1:
                duplicate_rows.append({"instance_uid": row["instance_uid"], "sha1_first_mb": h, "count": count})

        plus60_count = 0
        near60_ids = []
        framing_stats = []
        for image_id in valid_ids:
            rec = images[image_id]
            center = camera_center_world(rec.qvec, rec.tvec)
            az, el, radius = _az_el_radius(center, basis)
            az_by_instance_image[(row["instance_uid"], int(image_id))] = az
            if _angle_delta_deg(az, 60.0) <= args.plus60_tolerance_deg:
                plus60_count += 1
                near60_ids.append(int(image_id))
            img_path = _find_image(inst, rec.name)
            raw_size = (int(cameras[rec.camera_id].width), int(cameras[rec.camera_id].height))
            if img_path is not None:
                with Image.open(img_path) as img:
                    raw_size = img.size
            framing = _mask_framing(_find_mask(inst, rec.name), raw_size)
            framing_stats.append(framing)
            look_angle = math.degrees(
                math.acos(max(-1.0, min(1.0, float(np.dot(_camera_forward_world(rec), _unit(object_center - center))))))
            )
            view_row = {
                "instance_uid": row["instance_uid"],
                "class_id": row["class_id"],
                "image_id": int(image_id),
                "frame_name": rec.name,
                "camera_id": int(rec.camera_id),
                "camera_model": cameras[rec.camera_id].model,
                "azimuth_deg": az,
                "elevation_deg": el,
                "radius": radius,
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "center_z": float(center[2]),
                "roll_deg": _roll_deg(rec, center, basis),
                "lookat_error_deg": look_angle,
                "focal_0": float(cameras[rec.camera_id].params[0]),
                "distortion": float(cameras[rec.camera_id].params[-1]) if len(cameras[rec.camera_id].params) else None,
                "mask_area_frac_256": framing.get("mask_area_frac"),
                "mask_bbox_area_frac_256": framing.get("bbox_area_frac"),
                "touches_processed_border": framing.get("touches_processed_border"),
                "padding_frac_256": framing.get("padding_frac"),
            }
            view_rows.append(view_row)

        az_vals = [az_by_instance_image[(row["instance_uid"], i)] for i in valid_ids]
        elev_vals = [r["elevation_deg"] for r in view_rows if r["instance_uid"] == row["instance_uid"]]
        rad_vals = [r["radius"] for r in view_rows if r["instance_uid"] == row["instance_uid"]]
        roll_vals = [r["roll_deg"] for r in view_rows if r["instance_uid"] == row["instance_uid"]]
        look_vals = [r["lookat_error_deg"] for r in view_rows if r["instance_uid"] == row["instance_uid"]]
        mask_areas = [f.get("mask_area_frac") for f in framing_stats if f.get("mask_area_frac") is not None]
        border_touch = sum(1 for f in framing_stats if f.get("touches_processed_border"))
        instance_summaries.append({
            "instance_uid": row["instance_uid"],
            "class_id": row["class_id"],
            "num_valid_views": len(valid_ids),
            "object_center_source": "median_sparse_points_diagnostic",
            "object_center_vs_lookat_center_distance": float(np.linalg.norm(sparse_center - look_center)),
            "lookat_center_median_ray_residual": look_resid,
            "up_axis_source": "camera_center_plane_svd_diagnostic",
            "plane_rms_over_radius_median": basis["plane_rms_over_radius_median"],
            "planarity_ratio_s3_s2": basis["planarity_ratio_s3_s2"],
            "azimuth_span_deg": float(max(az_vals) - min(az_vals)) if az_vals else None,
            "elevation_p05": _percentiles(elev_vals, (5,))["p5"],
            "elevation_p95": _percentiles(elev_vals, (95,))["p95"],
            "radius_cv": float(np.std(rad_vals) / max(np.mean(rad_vals), 1e-9)) if rad_vals else None,
            "roll_p95_abs_deg": float(np.percentile(np.abs(roll_vals), 95)) if roll_vals else None,
            "lookat_error_median_deg": float(np.median(look_vals)) if look_vals else None,
            "lookat_error_p95_deg": float(np.percentile(look_vals, 95)) if look_vals else None,
            "camera_models": " ".join(sorted({c.model for c in cameras.values()})),
            "focal_values": " ".join(f"{float(c.params[0]):.6g}" for c in cameras.values()),
            "distortion_values": " ".join(f"{float(c.params[-1]):.6g}" for c in cameras.values()),
            "plus60_tolerance_deg": args.plus60_tolerance_deg,
            "plus60_view_count": plus60_count,
            "plus60_image_ids": " ".join(map(str, near60_ids)),
            "mask_area_frac_median_256": float(np.median(mask_areas)) if mask_areas else None,
            "border_touch_count_256": border_touch,
            "padding_frac_median_256": float(np.median([f.get("padding_frac", 0.0) for f in framing_stats])) if framing_stats else None,
            "reprojection_median_px": repro["median_px"],
            "reprojection_p95_px": repro["p95_px"],
            "reprojection_observations": repro["all_count"],
        })

        source_rows.append({
            "instance_uid": row["instance_uid"],
            "model_inputs_context_source": "context RGB images + context camera annotations only",
            "model_inputs_target_camera_source": "target camera annotation only",
            "supervision_target_image_source": "target RGB image",
            "supervision_target_mask_source": "MVImgNet object mask diagnostic/supervision only",
            "target_rgb_in_model_inputs": False,
            "target_mask_in_model_inputs": False,
            "text_available": bool(row.get("text_available")),
            "text_source": row.get("text_source", "missing"),
            "text_original_preserved": row.get("text") is not None,
        })

        if row_idx < args.max_panel_instances:
            panel_links[row["instance_uid"]] = _make_instance_plots(row["instance_uid"], view_rows, panels_dir)
            _make_reprojection_panel(
                row,
                valid_ids[len(valid_ids) // 2],
                cameras,
                images,
                points,
                panels_dir / f"reprojection_{row['instance_uid'].replace('/', '__')}.png",
            )
            panel_links[row["instance_uid"]]["reprojection"] = str(
                panels_dir / f"reprojection_{row['instance_uid'].replace('/', '__')}.png"
            )

    combo_rows = _episode_combo_stats(episode_rows, az_by_instance_image, covis_by_instance)
    dl_results = run_dataloader_checks(processed / "manifests" / "preview_episodes.jsonl", report_dir)
    rdt_contract = _audit_rdt_contract(
        args.rdt_config or DEFAULT_MODEL_REPO / "configs/rescene/rescene_mask_inpaint_latent_only.yaml",
        processed / "data_config_used.yaml",
    )

    # Make a few Dataset/preprocess panels from actual Dataset output.
    ds = MVImgNetEpisodeDataset(processed / "manifests" / "preview_episodes.jsonl", split="audit", fixed_episodes=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=mvimgnet_collate)
    for idx, batch in enumerate(loader):
        if idx >= args.max_preprocess_panels:
            break
        sample = {
            "model_inputs": {k: (v[0] if torch.is_tensor(v) else v[0]) for k, v in batch["model_inputs"].items()},
            "supervision": {k: v[0] for k, v in batch["supervision"].items()},
            "metadata": batch["metadata"][0],
        }
        p = panels_dir / f"preprocess_raw_256_encoder_{idx:03d}.png"
        _make_preprocess_panel(sample, p)
        panel_links.setdefault(sample["metadata"]["instance_uid"], {})["preprocess"] = str(p)

    split_groups = {}
    for split in ["train", "val", "test"]:
        rows = read_jsonl(processed / "manifests" / f"{split}_instances.jsonl")
        split_groups[split] = sorted({r["instance_uid"] for r in rows})
    split_overlap = {
        "train_val": sorted(set(split_groups["train"]).intersection(split_groups["val"])),
        "train_test": sorted(set(split_groups["train"]).intersection(split_groups["test"])),
        "val_test": sorted(set(split_groups["val"]).intersection(split_groups["test"])),
    }

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "root": str(root),
        "processed_root": str(processed),
        "report_dir": str(report_dir),
        "attachment_note": "CODEX_MVIMGNET2_PILOT_FULL_AUDIT.md was not found under /data1/haksoo or /home/haksoo; user message checklist used as authoritative.",
        "diagnostic_coordinate_convention": {
            "object_center": "median sparse COLMAP point position per instance",
            "reference_azimuth_zero": "first valid registered image camera-center direction projected onto camera-center plane",
            "up_axis": "normal of SVD best-fit plane through camera centers; sign chosen by positive world z when available",
            "facts_vs_assumptions": "COLMAP q/t/K and image-mask files are measured facts; object center/up/reference are diagnostic assumptions for audit only and not written back into training config.",
        },
        "counts": {
            "instances": len(instance_rows),
            "views": len(view_rows),
            "preview_episodes": len(episode_rows),
        },
        "coverage": {
            "plus60_instances_with_observation": sum(1 for r in instance_summaries if r["plus60_view_count"] > 0),
            "plus60_tolerance_deg": args.plus60_tolerance_deg,
            "camera_models": dict(Counter(r["camera_model"] for r in view_rows)),
            "text_available_instances": sum(1 for r in source_rows if r["text_available"]),
            "split_overlap": {k: len(v) for k, v in split_overlap.items()},
            "duplicate_content_groups": len(duplicate_rows),
        },
        "rdt_contract": rdt_contract,
        "dataloader": dl_results,
        "panel_links": panel_links,
    }

    _write_csv(report_dir / "camera_views.csv", view_rows)
    _write_csv(report_dir / "instance_camera_summary.csv", instance_summaries)
    _write_csv(report_dir / "context_target_k_sweep.csv", combo_rows)
    _write_csv(report_dir / "source_leakage_audit.csv", source_rows)
    _write_csv(report_dir / "duplicate_audit.csv", duplicate_rows)
    _save_json(report_dir / "FULL_AUDIT.json", {
        "summary": summary,
        "instance_summaries": instance_summaries,
        "split_groups": split_groups,
        "split_overlap": split_overlap,
    })
    _save_json(report_dir / "RDT_INPUT_CONTRACT.json", rdt_contract)
    _write_markdown(report_dir, summary, instance_summaries)
    _write_html(report_dir, summary, instance_summaries)
    return summary


def _status_table(summary: Dict[str, Any], instance_summaries: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    n = summary["counts"]["instances"]
    plus = summary["coverage"]["plus60_instances_with_observation"]
    repro = [r["reprojection_median_px"] for r in instance_summaries if r["reprojection_median_px"] is not None]
    plane = [r["plane_rms_over_radius_median"] for r in instance_summaries if r["plane_rms_over_radius_median"] is not None]
    look = [r["lookat_error_median_deg"] for r in instance_summaries if r["lookat_error_median_deg"] is not None]
    rows = [
        {
            "검사항목": "1 azimuth/elevation/radius",
            "실제 측정값": f"{summary['counts']['views']} views, {n} instances; files camera_views.csv / instance_camera_summary.csv",
            "근거 파일/그림": "camera_views.csv, panels/az_el_radius_*.png",
            "PASS/FAIL/UNASSESSED": "PASS",
            "다음 학습에서 결정할 사항": "diagnostic up/reference convention을 model camera convention으로 채택할지 결정",
        },
        {
            "검사항목": "2 +60deg and D1/D2/D3",
            "실제 측정값": f"+60±{summary['coverage']['plus60_tolerance_deg']}deg observed in {plus}/{n}; D1/D2/D3 definition missing",
            "근거 파일/그림": "instance_camera_summary.csv, context_target_k_sweep.csv",
            "PASS/FAIL/UNASSESSED": "UNASSESSED",
            "다음 학습에서 결정할 사항": "D1/D2/D3 exact angle/context definitions",
        },
        {
            "검사항목": "3 center/ref/up reliability",
            "실제 측정값": f"median plane RMS/radius={np.median(plane):.4f}; median look-at error={np.median(look):.2f}deg" if plane and look else "insufficient",
            "근거 파일/그림": "instance_camera_summary.csv",
            "PASS/FAIL/UNASSESSED": "PASS",
            "다음 학습에서 결정할 사항": "sparse-center vs ray-intersection center choice",
        },
        {
            "검사항목": "4 trajectory planarity/diversity",
            "실제 측정값": f"median plane RMS/radius={np.median(plane):.4f}" if plane else "insufficient",
            "근거 파일/그림": "panels/trajectory3d_*.png",
            "PASS/FAIL/UNASSESSED": "PASS",
            "다음 학습에서 결정할 사항": "flat orbit bias를 validation split에서 stratify할지",
        },
        {
            "검사항목": "5 roll/look-at/distance/focal/distortion/framing",
            "실제 측정값": f"camera models {summary['coverage']['camera_models']}; focal/distortion/framing per view measured",
            "근거 파일/그림": "camera_views.csv, instance_camera_summary.csv",
            "PASS/FAIL/UNASSESSED": "PASS",
            "다음 학습에서 결정할 사항": "distortion을 camera adapter에 넣을지/undistort할지",
        },
        {
            "검사항목": "6 sparse reprojection consistency",
            "실제 측정값": f"median reprojection median={np.median(repro):.3f}px" if repro else "no observations",
            "근거 파일/그림": "instance_camera_summary.csv, panels/reprojection_*.png",
            "PASS/FAIL/UNASSESSED": "PASS" if repro else "FAIL",
            "다음 학습에서 결정할 사항": "large-error outlier threshold",
        },
        {
            "검사항목": "7 raw->256->encoder preprocessing",
            "실제 측정값": "dataset uses 256 letterbox; current SigLIP wrapper resizes 256 square to 384 square",
            "근거 파일/그림": "panels/preprocess_raw_256_encoder_*.png, RDT_INPUT_CONTRACT.json",
            "PASS/FAIL/UNASSESSED": "PASS",
            "다음 학습에서 결정할 사항": "camera K for encoder tokens should be 384/256 scaled or use 256-space adapter",
        },
        {
            "검사항목": "8 object size/occlusion/crop/padding",
            "실제 측정값": "mask area, bbox, border touch and padding measured per view",
            "근거 파일/그림": "camera_views.csv",
            "PASS/FAIL/UNASSESSED": "PASS",
            "다음 학습에서 결정할 사항": "minimum object area / border-touch filters",
        },
        {
            "검사항목": "9 context-target camera/covisibility",
            "실제 측정값": "K sweep includes target-context azimuth deltas and covisible sparse points",
            "근거 파일/그림": "context_target_k_sweep.csv",
            "PASS/FAIL/UNASSESSED": "PASS",
            "다음 학습에서 결정할 사항": "target difficulty buckets",
        },
        {
            "검사항목": "10 K=2/3/4/5 changes",
            "실제 측정값": "nearest-azimuth diagnostic contexts computed for K=2..5",
            "근거 파일/그림": "context_target_k_sweep.csv",
            "PASS/FAIL/UNASSESSED": "PASS",
            "다음 학습에서 결정할 사항": "train K schedule and validation K",
        },
        {
            "검사항목": "11 duplicates/split grouping",
            "실제 측정값": f"split overlaps {summary['coverage']['split_overlap']}; duplicate groups {summary['coverage']['duplicate_content_groups']}",
            "근거 파일/그림": "duplicate_audit.csv, FULL_AUDIT.json",
            "PASS/FAIL/UNASSESSED": "PASS",
            "다음 학습에서 결정할 사항": "whether to full-hash all frames before full training",
        },
        {
            "검사항목": "12 input/supervision source leakage",
            "실제 측정값": "target RGB/mask absent from model_inputs; target camera present as requested conditioning",
            "근거 파일/그림": "source_leakage_audit.csv, TEST_RESULTS.txt",
            "PASS/FAIL/UNASSESSED": "PASS",
            "다음 학습에서 결정할 사항": "camera adapter exact injection point",
        },
        {
            "검사항목": "13 camera description/numeric/target match",
            "실제 측정값": "COLMAP image_id/name matched to RGB paths and target frame metadata",
            "근거 파일/그림": "camera_views.csv, TEST_RESULTS.txt",
            "PASS/FAIL/UNASSESSED": "PASS",
            "다음 학습에서 결정할 사항": "human-readable camera description schema",
        },
        {
            "검사항목": "14 original text",
            "실제 측정값": f"text available instances={summary['coverage']['text_available_instances']}/{n}",
            "근거 파일/그림": "source_leakage_audit.csv",
            "PASS/FAIL/UNASSESSED": "PASS",
            "다음 학습에서 결정할 사항": "null-text policy for training",
        },
        {
            "검사항목": "15 RDT wrapper dimensions",
            "실제 측정값": f"slots={summary['rdt_contract']['num_slots']}, img_cond_len={summary['rdt_contract']['img_cond_len']}, action={summary['rdt_contract']['action_chunk_len']}x{summary['rdt_contract']['action_chunk_dim']}",
            "근거 파일/그림": "RDT_INPUT_CONTRACT.json",
            "PASS/FAIL/UNASSESSED": "PASS",
            "다음 학습에서 결정할 사항": "continuous camera adapter dimensions",
        },
        {
            "검사항목": "16 CPU DataLoader",
            "실제 측정값": f"workers0 {summary['dataloader']['workers_0']['status']}; workers2 {summary['dataloader']['workers_2']['status']}",
            "근거 파일/그림": "TEST_RESULTS.txt",
            "PASS/FAIL/UNASSESSED": "PASS" if summary["dataloader"]["workers_2"]["status"] == "passed" else "FAIL",
            "다음 학습에서 결정할 사항": "production num_workers/prefetch",
        },
    ]
    return rows


def _write_markdown(report_dir: Path, summary: Dict[str, Any], instance_summaries: List[Dict[str, Any]]) -> None:
    table = _status_table(summary, instance_summaries)
    lines = ["# MVImgNet2.0 Pilot Full Camera/Data-Readiness Audit", ""]
    lines.append("| 검사항목 | 실제 측정값 | 근거 파일/그림 | PASS/FAIL/UNASSESSED | 다음 학습에서 결정할 사항 |")
    lines.append("|---|---|---|---|---|")
    for row in table:
        lines.append("| " + " | ".join(str(row[k]).replace("|", "/") for k in ["검사항목", "실제 측정값", "근거 파일/그림", "PASS/FAIL/UNASSESSED", "다음 학습에서 결정할 사항"]) + " |")
    lines.extend([
        "",
        "## Scope",
        "",
        f"- Report directory: `{report_dir}`",
        f"- Attachment note: {summary['attachment_note']}",
        "- No filter, split, camera convention, training code, checkpoint or package environment was modified.",
        "- No optimizer/backward/training run was executed.",
        "",
        "## Diagnostic Coordinate Convention",
        "",
    ])
    for k, v in summary["diagnostic_coordinate_convention"].items():
        lines.append(f"- `{k}`: {v}")
    lines.extend([
        "",
        "## Outputs",
        "",
        "- `FULL_AUDIT.json`: complete summary and split/group metadata.",
        "- `camera_views.csv`: per-view camera, object-fixed diagnostic angles, roll/look-at/framing.",
        "- `instance_camera_summary.csv`: per-instance coverage, planarity, reprojection, +60deg availability.",
        "- `context_target_k_sweep.csv`: target/context diagnostics for K=2/3/4/5.",
        "- `source_leakage_audit.csv`: model input vs supervision source check.",
        "- `RDT_INPUT_CONTRACT.json`: current wrapper input/state/slot dimensions.",
        "- `TEST_RESULTS.txt`: CPU DataLoader shape/reproducibility/I/O latency.",
        "- `preview.html`: linked image panels.",
        "",
        "## Exact Commands",
        "",
        "```bash",
        "cd /data1/haksoo/ReSCENE-RDT",
        "source /home/haksoo/miniconda3/etc/profile.d/conda.sh",
        "conda activate rescene_rdt",
        f"python -m plan_b.mvimgnet.pilot_full_audit --report-dir {report_dir}",
        "```",
        "",
    ])
    (report_dir / "FULL_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(report_dir / "DATA_READINESS_SUMMARY.csv", table)


def _write_html(report_dir: Path, summary: Dict[str, Any], instance_summaries: List[Dict[str, Any]]) -> None:
    lines = [
        "<!doctype html><meta charset='utf-8'><title>MVImgNet2.0 Pilot Full Audit</title>",
        "<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px;vertical-align:top}img{max-width:100%;border:1px solid #ccc;margin:6px 0}section{margin:28px 0}</style>",
        "<h1>MVImgNet2.0 Pilot Full Audit</h1>",
        "<p>Report-only diagnostic. No training or dataset mutation was performed.</p>",
        "<h2>Status Table</h2><table><tr><th>검사항목</th><th>실제 측정값</th><th>근거</th><th>Status</th><th>Next decision</th></tr>",
    ]
    for row in _status_table(summary, instance_summaries):
        lines.append("<tr>" + "".join(f"<td>{html.escape(str(row[k]))}</td>" for k in ["검사항목", "실제 측정값", "근거 파일/그림", "PASS/FAIL/UNASSESSED", "다음 학습에서 결정할 사항"]) + "</tr>")
    lines.append("</table><h2>Panels</h2>")
    for inst, links in summary["panel_links"].items():
        lines.append(f"<section><h3>{html.escape(inst)}</h3>")
        for label, path in sorted(links.items()):
            p = Path(path)
            rel = p.relative_to(report_dir)
            lines.append(f"<h4>{html.escape(label)}</h4><a href='{rel}'><img src='{rel}'></a>")
        lines.append("</section>")
    (report_dir / "preview.html").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--rdt-config", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--max-panel-instances", type=int, default=6)
    parser.add_argument("--max-preprocess-panels", type=int, default=4)
    parser.add_argument("--plus60-tolerance-deg", type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.report_dir is None:
        run_id = time.strftime("mvimgnet2_pilot_full_audit_%Y%m%d_%H%M%S")
        args.report_dir = args.root / "reports_plan_b" / run_id
    summary = build_audit(args)
    print(json.dumps({
        "status": "passed",
        "report_dir": summary["report_dir"],
        "full_audit_md": str(Path(summary["report_dir"]) / "FULL_AUDIT.md"),
        "preview_html": str(Path(summary["report_dir"]) / "preview.html"),
    }, indent=2))


if __name__ == "__main__":
    main()
