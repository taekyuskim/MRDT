"""Small COLMAP binary readers used by the MVImgNet2.0 data pipeline.

The binary layout follows COLMAP's documented sparse model format.  The
important convention for this project is kept explicit: qvec is Hamilton
``[qw, qx, qy, qz]`` and poses are world-to-camera transforms.
"""

from __future__ import annotations

import collections
import os
import struct
from pathlib import Path
from typing import BinaryIO, Dict, List, Tuple

import numpy as np


Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


CAMERA_MODEL_IDS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _read_bytes(fid: BinaryIO, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError(f"Unexpected EOF while reading COLMAP binary: wanted {num_bytes}, got {len(data)}")
    return struct.unpack("<" + fmt, data)


def _read_c_string(fid: BinaryIO) -> str:
    chars = []
    while True:
        ch = fid.read(1)
        if ch == b"":
            raise EOFError("Unexpected EOF while reading COLMAP image name")
        if ch == b"\x00":
            return b"".join(chars).decode("utf-8")
        chars.append(ch)


def read_cameras_binary(path: os.PathLike[str] | str) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    with open(path, "rb") as fid:
        (num_cameras,) = _read_bytes(fid, 8, "Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_bytes(fid, 24, "iiQQ")
            if model_id not in CAMERA_MODEL_IDS:
                raise ValueError(f"Unsupported COLMAP camera model id {model_id} in {path}")
            model_name, num_params = CAMERA_MODEL_IDS[model_id]
            params = np.array(_read_bytes(fid, 8 * num_params, "d" * num_params), dtype=np.float64)
            cameras[camera_id] = Camera(camera_id, model_name, width, height, params)
    return cameras


def read_images_binary(path: os.PathLike[str] | str) -> Dict[int, Image]:
    images: Dict[int, Image] = {}
    with open(path, "rb") as fid:
        (num_images,) = _read_bytes(fid, 8, "Q")
        for _ in range(num_images):
            elems = _read_bytes(fid, 64, "idddddddi")
            image_id = elems[0]
            qvec = np.array(elems[1:5], dtype=np.float64)
            tvec = np.array(elems[5:8], dtype=np.float64)
            camera_id = elems[8]
            name = _read_c_string(fid)
            (num_points2d,) = _read_bytes(fid, 8, "Q")
            xys = np.empty((num_points2d, 2), dtype=np.float64)
            point3d_ids = np.empty((num_points2d,), dtype=np.int64)
            for i in range(num_points2d):
                x, y, point3d_id = _read_bytes(fid, 24, "ddq")
                xys[i] = (x, y)
                point3d_ids[i] = point3d_id
            images[image_id] = Image(image_id, qvec, tvec, camera_id, name, xys, point3d_ids)
    return images


def read_points3d_binary(path: os.PathLike[str] | str, max_points: int | None = None) -> Dict[int, Point3D]:
    points: Dict[int, Point3D] = {}
    with open(path, "rb") as fid:
        (num_points,) = _read_bytes(fid, 8, "Q")
        keep = num_points if max_points is None else min(num_points, max_points)
        for idx in range(num_points):
            point_id = _read_bytes(fid, 8, "Q")[0]
            xyz = np.array(_read_bytes(fid, 24, "ddd"), dtype=np.float64)
            rgb = np.array(_read_bytes(fid, 3, "BBB"), dtype=np.uint8)
            error = _read_bytes(fid, 8, "d")[0]
            track_length = _read_bytes(fid, 8, "Q")[0]
            image_ids: List[int] = []
            point2d_idxs: List[int] = []
            for _ in range(track_length):
                image_id, point2d_idx = _read_bytes(fid, 8, "ii")
                if idx < keep:
                    image_ids.append(image_id)
                    point2d_idxs.append(point2d_idx)
            if idx < keep:
                points[point_id] = Point3D(point_id, xyz, rgb, error, image_ids, point2d_idxs)
    return points


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"qvec must have shape (4,), got {q.shape}")
    q = q / np.linalg.norm(q)
    qw, qx, qy, qz = q
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def world_to_camera_matrix(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = qvec_to_rotmat(qvec)
    mat[:3, 3] = np.asarray(tvec, dtype=np.float64)
    return mat


def camera_center_world(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    r = qvec_to_rotmat(qvec)
    t = np.asarray(tvec, dtype=np.float64)
    return -r.T @ t


def read_sparse_model(sparse_dir: os.PathLike[str] | str, with_points: bool = False, max_points: int = 20000):
    sparse = Path(sparse_dir)
    cameras = read_cameras_binary(sparse / "cameras.bin")
    images = read_images_binary(sparse / "images.bin")
    points = {}
    if with_points and (sparse / "points3D.bin").exists():
        points = read_points3d_binary(sparse / "points3D.bin", max_points=max_points)
    return cameras, images, points
