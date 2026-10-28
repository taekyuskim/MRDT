"""MVImgNet2.0 bin-conditioned Stage-1 pilot Dataset."""

from __future__ import annotations

import json
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .bin15_camera import (
    camera_feature_from_bins,
    load_fixed_camera_rows,
    occupied_pair_bins,
)
from .cameras import processed_intrinsics
from .colmap_io import read_sparse_model
from .dataset import read_jsonl


def _load_rgb(path: Path, out_size: int = 256) -> tuple[torch.Tensor, tuple[int, int]]:
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    raw_size = img.size
    scale = min(out_size / float(raw_size[0]), out_size / float(raw_size[1]))
    nw, nh = int(round(raw_size[0] * scale)), int(round(raw_size[1] * scale))
    resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (out_size, out_size), (0, 0, 0))
    canvas.paste(resized, ((out_size - nw) // 2, (out_size - nh) // 2))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), raw_size


def _find_image(instance_path: Path, name: str) -> Path:
    for p in [instance_path / name, instance_path / "images" / Path(name).name]:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing image {name} under {instance_path}")


@lru_cache(maxsize=256)
def _load_sparse_cached(instance_path: str):
    return read_sparse_model(Path(instance_path) / "sparse" / "0", with_points=False)


class MVImgNetBin15Stage1Dataset(Dataset):
    """Sample target bins first, then context frames from other occupied bins."""

    def __init__(
        self,
        manifest_path: str | Path,
        fixed_camera_csv: str | Path,
        split: str,
        *,
        image_size: int = 256,
        k_max: int = 5,
        k_min: int = 2,
        samples_per_epoch: int | None = None,
        seed: int = 0,
        deterministic: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = read_jsonl(self.manifest_path)
        if not self.rows:
            raise ValueError(f"empty manifest: {manifest_path}")
        self.fixed = load_fixed_camera_rows(fixed_camera_csv)
        self.split = split
        self.image_size = int(image_size)
        self.k_max = int(k_max)
        self.k_min = int(k_min)
        self.samples_per_epoch = int(samples_per_epoch or len(self.rows))
        self.seed = int(seed)
        self.epoch = 0
        self.deterministic = bool(deterministic)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _rng(self, idx: int) -> random.Random:
        if self.deterministic:
            return random.Random(self.seed + idx)
        return random.Random((self.seed + 1_000_003 * self.epoch + idx) & 0xFFFFFFFF)

    def _choose(self, idx: int):
        rng = self._rng(idx)
        row = self.rows[idx % len(self.rows)] if self.deterministic else self.rows[rng.randrange(len(self.rows))]
        fixed_rows = self.fixed[row["instance_uid"]]
        valid_ids = [int(i) for i in row["valid_image_ids"] if int(i) in fixed_rows]
        bins = occupied_pair_bins(fixed_rows[i] for i in valid_ids)
        usable_bins = {b: ids for b, ids in bins.items() if ids}
        if len(usable_bins) < self.k_min + 1:
            raise ValueError(f"too few occupied bins for {row['instance_uid']}: {len(usable_bins)}")
        target_bin = sorted(usable_bins)[idx % len(usable_bins)] if self.deterministic else rng.choice(list(usable_bins))
        target_id = sorted(usable_bins[target_bin])[0] if self.deterministic else rng.choice(usable_bins[target_bin])
        context_bins = [b for b in usable_bins if b != target_bin]
        k = self.k_max if self.deterministic else rng.randint(self.k_min, min(self.k_max, len(context_bins)))
        chosen_bins = sorted(context_bins)[:k] if self.deterministic else rng.sample(context_bins, k)
        context_ids = []
        for b in chosen_bins:
            ids = [i for i in usable_bins[b] if i != target_id]
            if not ids:
                continue
            context_ids.append(sorted(ids)[0] if self.deterministic else rng.choice(ids))
        if len(set(context_ids + [target_id])) != len(context_ids) + 1:
            raise AssertionError("context/target frame duplicate")
        return row, context_ids[: self.k_max], target_id, target_bin

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row, context_ids, target_id, target_bin = self._choose(idx)
        instance_path = Path(row["instance_path"])
        cameras, images, _ = _load_sparse_cached(str(instance_path))

        context_images = torch.zeros((self.k_max, 3, self.image_size, self.image_size), dtype=torch.float32)
        context_camera = torch.zeros((self.k_max, 10), dtype=torch.float32)
        context_present = torch.zeros((self.k_max,), dtype=torch.bool)
        context_frame_ids: List[str | None] = [None] * self.k_max
        context_descriptions: List[str | None] = [None] * self.k_max
        context_image_ids: List[int | None] = [None] * self.k_max
        raw_camera_meta: Dict[str, Any] = {"contexts": [], "target": None}

        for slot, image_id in enumerate(context_ids):
            rec = images[image_id]
            fixed = self.fixed[row["instance_uid"]][image_id]
            path = _find_image(instance_path, rec.name)
            rgb, raw_size = _load_rgb(path, self.image_size)
            cam = cameras[rec.camera_id]
            k256, kraw, resize_meta = processed_intrinsics(cam, raw_size[0], raw_size[1], self.image_size)
            feat = camera_feature_from_bins(fixed.az_bin, fixed.el_bin, fixed.radius_refnorm, k256, cam.params[-1])
            context_images[slot] = rgb
            context_camera[slot] = torch.from_numpy(feat)
            context_present[slot] = True
            context_frame_ids[slot] = rec.name
            context_descriptions[slot] = fixed.description
            context_image_ids[slot] = int(image_id)
            raw_camera_meta["contexts"].append(
                {
                    "image_id": int(image_id),
                    "frame_name": rec.name,
                    "qvec": [float(x) for x in rec.qvec],
                    "tvec": [float(x) for x in rec.tvec],
                    "camera_model": cam.model,
                    "camera_params": [float(x) for x in cam.params],
                    "resize": resize_meta,
                    "bin_description": fixed.description,
                }
            )

        target_rec = images[target_id]
        target_fixed = self.fixed[row["instance_uid"]][target_id]
        target_path = _find_image(instance_path, target_rec.name)
        target_image, target_raw_size = _load_rgb(target_path, self.image_size)
        target_cam = cameras[target_rec.camera_id]
        target_k256, _, target_resize = processed_intrinsics(target_cam, target_raw_size[0], target_raw_size[1], self.image_size)
        target_feat = camera_feature_from_bins(
            target_fixed.az_bin,
            target_fixed.el_bin,
            target_fixed.radius_refnorm,
            target_k256,
            target_cam.params[-1],
        )
        raw_camera_meta["target"] = {
            "image_id": int(target_id),
            "frame_name": target_rec.name,
            "qvec": [float(x) for x in target_rec.qvec],
            "tvec": [float(x) for x in target_rec.tvec],
            "camera_model": target_cam.model,
            "camera_params": [float(x) for x in target_cam.params],
            "resize": target_resize,
            "bin_description": target_fixed.description,
        }

        return {
            "model_inputs": {
                "context_images": context_images,
                "context_camera_features": context_camera,
                "target_camera_features": torch.from_numpy(target_feat),
                "context_present": context_present,
                "text": None,
                "text_available": False,
            },
            "supervision": {"target_image": target_image},
            "metadata": {
                "dataset": "mvimgnet2",
                "split": self.split,
                "class_id": row["class_id"],
                "class_name": row.get("class_name"),
                "instance_uid": row["instance_uid"],
                "target_image_id": int(target_id),
                "target_frame_id": target_rec.name,
                "target_bin": list(target_bin),
                "target_description": target_fixed.description,
                "context_image_ids": context_image_ids,
                "context_frame_ids": context_frame_ids,
                "context_descriptions": context_descriptions,
                "paths": {"target_image": str(target_path), "instance_path": str(instance_path)},
                "raw_camera_parameters": raw_camera_meta,
                "conditioning_policy": "bin15_centers_only_no_raw_pose_no_residual",
                "base_present": False,
                "generation_mask": 1,
            },
        }


def mvimgnet_bin15_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"model_inputs": {}, "supervision": {}, "metadata": [b["metadata"] for b in batch]}
    for key in batch[0]["model_inputs"]:
        values = [b["model_inputs"][key] for b in batch]
        out["model_inputs"][key] = torch.stack(values, 0) if torch.is_tensor(values[0]) else values
    for key in batch[0]["supervision"]:
        values = [b["supervision"][key] for b in batch]
        out["supervision"][key] = torch.stack(values, 0) if torch.is_tensor(values[0]) else values
    return out
