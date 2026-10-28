"""15-degree fixed-camera bin utilities for MVImgNet2.0 Plan B."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


BIN_DEG = 15.0
CAMERA_FEATURE_DIM = 10


def wrap_deg(x: float) -> float:
    return ((float(x) + 180.0) % 360.0) - 180.0


def nearest_bin_center_deg(x: float, *, elevation: bool = False) -> float:
    b = round(float(x) / BIN_DEG) * BIN_DEG
    if elevation:
        return float(max(-90.0, min(90.0, b)))
    return wrap_deg(b)


def bin_pair(azimuth_deg: float, elevation_deg: float) -> Tuple[int, int]:
    return (
        int(nearest_bin_center_deg(azimuth_deg)),
        int(nearest_bin_center_deg(elevation_deg, elevation=True)),
    )


def bin_description(az_bin: int, el_bin: int) -> str:
    return f"az_bin={az_bin:+04d}deg el_bin={el_bin:+04d}deg"


def camera_feature_from_bins(
    az_bin: float,
    el_bin: float,
    radius_refnorm: float,
    k256: np.ndarray,
    distortion_k: float,
) -> np.ndarray:
    """Return the exact 10D camera feature used by the pilot.

    The feature intentionally uses only the binned angular centers, not raw
    azimuth/elevation residuals or raw pose matrices.
    """
    az = math.radians(float(az_bin))
    el = math.radians(float(el_bin))
    radius = max(float(radius_refnorm), 1e-6)
    feat = np.array(
        [
            math.sin(az),
            math.cos(az),
            math.sin(el),
            math.cos(el),
            math.log(radius),
            float(k256[0, 0]) / 256.0,
            float(k256[1, 1]) / 256.0,
            float(k256[0, 2]) / 256.0,
            float(k256[1, 2]) / 256.0,
            float(distortion_k),
        ],
        dtype=np.float32,
    )
    if feat.shape != (CAMERA_FEATURE_DIM,):
        raise AssertionError(feat.shape)
    return feat


@dataclass(frozen=True)
class FixedCameraRow:
    instance_uid: str
    image_id: int
    frame_name: str
    azimuth_deg: float
    elevation_deg: float
    radius_refnorm: float
    az_bin: int
    el_bin: int
    description: str


def load_fixed_camera_rows(path: str | Path) -> Dict[str, Dict[int, FixedCameraRow]]:
    """Load follow-up reference-image-up fixed-camera metadata."""
    out: Dict[str, Dict[int, FixedCameraRow]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            az_bin, el_bin = bin_pair(float(r["refup_azimuth_deg"]), float(r["refup_elevation_deg"]))
            row = FixedCameraRow(
                instance_uid=r["instance_uid"],
                image_id=int(r["image_id"]),
                frame_name=r["frame_name"],
                azimuth_deg=float(r["refup_azimuth_deg"]),
                elevation_deg=float(r["refup_elevation_deg"]),
                radius_refnorm=float(r["refup_radius_refnorm"]),
                az_bin=az_bin,
                el_bin=el_bin,
                description=bin_description(az_bin, el_bin),
            )
            out.setdefault(row.instance_uid, {})[row.image_id] = row
    return out


def occupied_pair_bins(rows: Iterable[FixedCameraRow]) -> Dict[Tuple[int, int], list[int]]:
    bins: Dict[Tuple[int, int], list[int]] = {}
    for row in rows:
        bins.setdefault((row.az_bin, row.el_bin), []).append(row.image_id)
    return bins
