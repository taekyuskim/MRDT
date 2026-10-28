"""CPU preprocessing: SIMPLE_RADIAL correction and foreground framing.

No depth normalization is claimed. Original camera metadata is used only here.
RGB and mask share a single inverse map; contours define the undistorted crop.
"""
import math
import cv2
import numpy as np
from PIL import Image

PREPROCESS_POLICY = 'simple_radial_safe_undistort_mask128_center_longedge80_gray128_v2'
CONDITIONING_POLICY = 'bin15_direction_sincos4_only_v1'
FILL = 0.8
SIZE = 256


class NonInvertibleDistortion(ValueError):
    """The supplied radial model cannot invert the foreground coordinates."""


class MissingCanonicalForeground(ValueError):
    """A mask is empty or has no foreground after the output resampling."""


def direction_features(az_bin, el_bin):
    values = np.asarray([az_bin, el_bin], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite direction')
    if not np.allclose(values / 15, np.round(values / 15)):
        raise ValueError('Expected 15-degree bin centers')
    az, el = np.deg2rad(values)
    return np.array([np.sin(az), np.cos(az), np.sin(el), np.cos(el)], dtype=np.float32)


def undistort_points(points, params):
    """Invert COLMAP SIMPLE_RADIAL on its monotonic central branch."""
    f, cx, cy, k = np.asarray(params, dtype=np.float64)
    if not np.isfinite([f, cx, cy, k]).all() or f <= 0:
        raise ValueError('Invalid SIMPLE_RADIAL parameters')
    xy = (np.asarray(points, dtype=np.float64) - [cx, cy]) / f
    rd = np.linalg.norm(xy, axis=-1)
    if k == 0:
        return np.asarray(points, dtype=np.float64).copy()
    low = np.zeros_like(rd)
    if k < 0:
        turning = math.sqrt(-1 / (3 * k))
        maximum = turning * (1 + k * turning * turning)
        if np.any(rd >= maximum * (1 - 1e-8)):
            raise NonInvertibleDistortion('Foreground outside invertible SIMPLE_RADIAL domain')
        high = np.full_like(rd, turning)
    else:
        high = rd.copy()
    for _ in range(40):
        mid = (low + high) / 2
        below = mid * (1 + k * mid * mid) < rd
        low = np.where(below, mid, low)
        high = np.where(below, high, mid)
    ru = (low + high) / 2
    ratio = np.divide(ru, rd, out=np.ones_like(rd), where=rd > 1e-15)
    return xy * ratio[..., None] * f + [cx, cy]


def canonicalize(rgb, mask, params, *, size=SIZE, fill=FILL):
    rgb, mask = np.asarray(rgb), np.asarray(mask)
    if rgb.dtype != np.uint8 or mask.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB and mask')
    if rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError('RGB/mask size mismatch')
    if size < 16 or not 0 < fill < 1:
        raise ValueError('Invalid output framing')
    binary = np.uint8(mask >= 128)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise MissingCanonicalForeground('Empty foreground')
    points = np.concatenate(contours).reshape(-1, 2).astype(np.float64)
    # Pixel edges, not just centers, determine the foreground extent.
    edges = (points[:, None, :] + np.array([[-.5,-.5],[-.5,.5],[.5,-.5],[.5,.5]])).reshape(-1,2)
    effective_params = np.array(params, dtype=np.float64, copy=True)
    distortion_status = 'corrected' if effective_params[3] != 0 else 'zero_distortion'
    try:
        rectified = undistort_points(edges, effective_params)
    except NonInvertibleDistortion:
        # Never invent a valid inverse or drop this sample silently. Keep all
        # foreground pixels and record the explicit identity-distortion fallback.
        effective_params[3] = 0
        rectified = undistort_points(edges, effective_params)
        distortion_status = 'identity_fallback_noninvertible_calibration'
    lo, hi = rectified.min(axis=0), rectified.max(axis=0)
    center = (lo + hi) / 2
    crop_side = float(max(hi - lo) / fill)
    if not np.isfinite(crop_side) or crop_side <= 0:
        raise ValueError('Invalid undistorted foreground extent')
    f, cx, cy, k = effective_params
    # Supersampling avoids aliasing during shrinking; one source warp avoids
    # the full-resolution undistort image and an extra resize of its RGB.
    out_n = size * 2
    axis = (np.arange(out_n, dtype=np.float64) + .5) * crop_side / out_n - crop_side / 2
    ux, uy = np.meshgrid(axis + center[0], axis + center[1])
    x, y = (ux - cx) / f, (uy - cy) / f
    r2 = x*x + y*y
    factor = 1 + k*r2
    map_x, map_y = (f*x*factor + cx).astype(np.float32), (f*y*factor + cy).astype(np.float32)
    folded = (1 + 3*k*r2) <= 0
    map_x[folded], map_y[folded] = -1, -1
    source = np.where(binary[...,None] != 0, rgb, np.uint8(128))
    color = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(128,128,128))
    foreground = cv2.remap(binary.astype(np.float32), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.remap(np.ones_like(binary), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    color = cv2.resize(color, (size,size), interpolation=cv2.INTER_AREA)
    foreground = cv2.resize(foreground, (size,size), interpolation=cv2.INTER_AREA)
    valid = cv2.resize(valid.astype(np.float32), (size,size), interpolation=cv2.INTER_AREA)
    if foreground.max() == 0:
        raise MissingCanonicalForeground('Foreground vanished during canonicalization')
    info = {'policy': PREPROCESS_POLICY, 'undistorted_bbox': [*lo.tolist(),*hi.tolist()],
            'crop_center': center.tolist(), 'crop_side': crop_side, 'output_size': size,
            'fill': fill, 'source_size': [rgb.shape[1],rgb.shape[0]],
            'source_foreground_touches_edge': bool(binary[0].any() or binary[-1].any() or binary[:,0].any() or binary[:,-1].any()),
            'perspective_normalized': False, 'distortion_status': distortion_status,
            'source_distortion_k': float(params[3]), 'effective_distortion_k': float(k)}
    return color, foreground, valid, info


def load_canonical_view(view):
    if view['camera_model'] != 'SIMPLE_RADIAL':
        raise ValueError('Unsupported camera model: '+view['camera_model'])
    with Image.open(view['path']) as im, Image.open(view['mask_path']) as mask:
        if im.getexif().get(274,1) != 1 or mask.getexif().get(274,1) != 1:
            raise ValueError('Unexpected EXIF rotation')
        if mask.mode != 'L':
            raise ValueError('Expected audited grayscale mask')
        if im.size != (view['resize']['source_width'],view['resize']['source_height']):
            raise ValueError('Camera source size mismatch')
        return canonicalize(np.array(im.convert('RGB')),np.array(mask),view['camera_params'])
