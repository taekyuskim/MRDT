"""Dataset and collate contract for MVImgNet2.0 target-view episodes."""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .cameras import (
    distortion_parameters,
    median_context_baseline,
    normalize_relative_translation,
    processed_intrinsics,
    relative_world_to_camera,
    rotation_angle_degrees,
)
from .colmap_io import read_sparse_model


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _candidate_names(name: str) -> List[str]:
    p = Path(name)
    return [name, p.name, str(Path("images") / p.name)]


@lru_cache(maxsize=256)
def _load_sparse_cached(instance_path: str):
    return read_sparse_model(Path(instance_path) / "sparse" / "0", with_points=False)


def _find_image_path(instance_path: Path, name: str) -> Path | None:
    for candidate in _candidate_names(name):
        p = instance_path / candidate
        if p.exists() and p.is_file():
            return p
    return None


def _find_mask_path(instance_path: Path, image_name: str) -> Path | None:
    stem_name = Path(image_name).name
    for ext in [Path(stem_name).suffix, ".png", ".jpg", ".jpeg"]:
        if not ext:
            continue
        p = instance_path / "masks" / (Path(stem_name).stem + ext)
        if p.exists() and p.is_file():
            return p
    return None


def _letterbox_pil(img: Image.Image, out_size: int, resample: int, fill=0) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    scale = min(out_size / float(w), out_size / float(h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = img.resize((new_w, new_h), resample=resample)
    if img.mode == "RGB":
        canvas = Image.new("RGB", (out_size, out_size), fill)
    else:
        canvas = Image.new(img.mode, (out_size, out_size), fill)
    canvas.paste(resized, ((out_size - new_w) // 2, (out_size - new_h) // 2))
    return canvas


def _pixel_valid_mask(width: int, height: int, out_size: int) -> torch.Tensor:
    scale = min(out_size / float(width), out_size / float(height))
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    x0 = (out_size - new_w) // 2
    y0 = (out_size - new_h) // 2
    mask = torch.zeros((1, out_size, out_size), dtype=torch.bool)
    mask[:, y0 : y0 + new_h, x0 : x0 + new_w] = True
    return mask


def _load_rgb(path: Path, out_size: int):
    img = Image.open(path).convert("RGB")
    raw_size = img.size
    proc = _letterbox_pil(img, out_size, Image.Resampling.BILINEAR, fill=(0, 0, 0))
    arr = np.asarray(proc, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor, raw_size


def _load_mask(path: Path | None, out_size: int, raw_size):
    if path is None:
        return torch.zeros((1, out_size, out_size), dtype=torch.bool)
    img = Image.open(path).convert("L")
    proc = _letterbox_pil(img, out_size, Image.Resampling.NEAREST, fill=0)
    arr = np.asarray(proc, dtype=np.uint8)
    return torch.from_numpy(arr > 127).unsqueeze(0)


class MVImgNetEpisodeDataset(Dataset):
    """Returns context images/cameras separately from target supervision."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        k_context: int = 3,
        image_size: int = 256,
        seed: int = 0,
        fixed_episodes: bool | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = read_jsonl(self.manifest_path)
        self.split = split
        self.k_context = k_context
        self.image_size = image_size
        self.seed = seed
        self.epoch = 0
        if fixed_episodes is None:
            fixed_episodes = bool(self.rows and "context_image_ids" in self.rows[0])
        self.fixed_episodes = fixed_episodes

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def _sample_episode(self, idx: int):
        row = self.rows[idx]
        if self.fixed_episodes:
            return row, list(row["context_image_ids"]), row["target_image_id"]
        image_ids = list(row["valid_image_ids"])
        rng = random.Random((self.seed + 1000003 * self.epoch + idx) & 0xFFFFFFFF)
        chosen = rng.sample(image_ids, self.k_context + 1)
        return row, chosen[: self.k_context], chosen[self.k_context]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row, context_ids, target_id = self._sample_episode(idx)
        instance_path = Path(row["instance_path"])
        cameras, images, _ = _load_sparse_cached(str(instance_path))
        all_ids = context_ids + [target_id]
        reference_id = context_ids[0]
        rel_raw = relative_world_to_camera(images, all_ids, reference_id)
        scale = median_context_baseline(images, context_ids)
        if scale <= 0:
            raise ValueError(f"Invalid context baseline for {row['instance_uid']} ids={context_ids}")

        context_images = []
        context_intrinsics = []
        context_t_rel = []
        context_raw_t_rel = []
        context_distortion = []
        context_resize = []
        context_frame_ids = []
        context_paths = []

        for image_id in context_ids:
            image_rec = images[image_id]
            image_path = _find_image_path(instance_path, image_rec.name)
            if image_path is None:
                raise FileNotFoundError(f"COLMAP image {image_rec.name} not found under {instance_path}")
            rgb, raw_size = _load_rgb(image_path, self.image_size)
            camera = cameras[image_rec.camera_id]
            k_proc, k_raw, resize_meta = processed_intrinsics(camera, raw_size[0], raw_size[1], self.image_size)
            context_images.append(rgb)
            context_intrinsics.append(torch.as_tensor(k_proc, dtype=torch.float32))
            context_t_rel.append(torch.as_tensor(normalize_relative_translation(rel_raw[image_id], scale), dtype=torch.float32))
            context_raw_t_rel.append(torch.as_tensor(rel_raw[image_id], dtype=torch.float32))
            context_distortion.append(torch.as_tensor(distortion_parameters(camera), dtype=torch.float32))
            context_resize.append(resize_meta)
            context_frame_ids.append(image_rec.name)
            context_paths.append(str(image_path))

        target_rec = images[target_id]
        target_path = _find_image_path(instance_path, target_rec.name)
        if target_path is None:
            raise FileNotFoundError(f"COLMAP target image {target_rec.name} not found under {instance_path}")
        target_image, target_raw_size = _load_rgb(target_path, self.image_size)
        target_camera = cameras[target_rec.camera_id]
        target_k_proc, target_k_raw, target_resize = processed_intrinsics(
            target_camera, target_raw_size[0], target_raw_size[1], self.image_size
        )
        target_mask = _load_mask(_find_mask_path(instance_path, target_rec.name), self.image_size, target_raw_size)
        target_valid = _pixel_valid_mask(target_raw_size[0], target_raw_size[1], self.image_size)
        target_t_rel = normalize_relative_translation(rel_raw[target_id], scale)

        model_inputs = {
            "context_images": torch.stack(context_images, dim=0),
            "context_intrinsics": torch.stack(context_intrinsics, dim=0),
            "target_intrinsics": torch.as_tensor(target_k_proc, dtype=torch.float32),
            "context_T_rel": torch.stack(context_t_rel, dim=0),
            "target_T_rel": torch.as_tensor(target_t_rel, dtype=torch.float32),
            "context_present": torch.ones((self.k_context,), dtype=torch.bool),
            "text": None,
            "text_available": False,
        }
        supervision = {
            "target_image": target_image,
            "target_object_mask": target_mask,
            "target_pixel_valid": target_valid,
        }
        metadata = {
            "dataset": "mvimgnet2",
            "split": self.split,
            "class_id": row["class_id"],
            "class_name": row.get("class_name"),
            "instance_uid": row["instance_uid"],
            "reconstruction_uid": row.get("reconstruction_uid", "sparse/0"),
            "context_image_ids": context_ids,
            "target_image_id": target_id,
            "context_frame_ids": context_frame_ids,
            "target_frame_id": target_rec.name,
            "paths": {"context_images": context_paths, "target_image": str(target_path), "instance_path": str(instance_path)},
            "raw_camera_parameters": {
                "target_camera_model": target_camera.model,
                "target_camera_params": [float(x) for x in target_camera.params],
                "context_resize": context_resize,
                "target_resize": target_resize,
                "context_T_rel_raw": [x.tolist() for x in context_raw_t_rel],
                "target_T_rel_raw": rel_raw[target_id].tolist(),
            },
            "normalization_scale": float(scale),
            "target_relative_rotation_deg": rotation_angle_degrees(target_t_rel),
            "text_source": "missing",
            "sample_seed": row.get("sample_seed", self.seed),
        }
        forbidden = {"target_image", "target_object_mask", "target_latent", "clean_latent"}
        if forbidden.intersection(model_inputs):
            raise AssertionError("Target supervision leaked into model_inputs")
        return {"model_inputs": model_inputs, "supervision": supervision, "metadata": metadata}


def mvimgnet_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    model_inputs: Dict[str, Any] = {}
    supervision: Dict[str, Any] = {}
    for key in batch[0]["model_inputs"]:
        values = [b["model_inputs"][key] for b in batch]
        model_inputs[key] = torch.stack(values, dim=0) if torch.is_tensor(values[0]) else values
    for key in batch[0]["supervision"]:
        values = [b["supervision"][key] for b in batch]
        supervision[key] = torch.stack(values, dim=0) if torch.is_tensor(values[0]) else values
    return {"model_inputs": model_inputs, "supervision": supervision, "metadata": [b["metadata"] for b in batch]}
