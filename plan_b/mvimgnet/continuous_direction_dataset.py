"""Continuous angular conditioning over the unchanged bin-based view sampler."""
import numpy as np
import torch
from .roll_direction_dataset import DirectionFullBin15Dataset, tensor_view, INVALID_VIEW_POLICY
from .roll_canonical_object import PREPROCESS_POLICY
from . import latent_cache

CONDITIONING_POLICY = 'continuous_direction_sincos4_raw_catalog_v1'
ANGLE_SOURCE = 'raw_catalog_azimuth_elevation_deg'
SELECTION_BIN_DEG = 15


def direction_features(azimuth_deg, elevation_deg):
    """Four direction values, without rounding; angles use the object reference basis."""
    angles = np.asarray([azimuth_deg, elevation_deg], dtype=np.float64)
    if angles.shape != (2,) or not np.isfinite(angles).all():
        raise ValueError('Expected two finite scalar angles in degrees')
    if not -90.0 <= angles[1] <= 90.0:
        raise ValueError('Elevation must be in [-90, 90] degrees')
    # Wrapping is periodic, not quantization. Preserve the original values in metadata.
    az = np.deg2rad((angles[0] + 180.0) % 360.0 - 180.0)
    el = np.deg2rad(angles[1])
    return np.array([np.sin(az), np.cos(az), np.sin(el), np.cos(el)], dtype=np.float32)


def continuous_conditions(views, target_azimuth_deg, target_elevation_deg):
    if not 1 <= len(views) <= 5:
        raise ValueError('Expected one to five context views')
    directions = torch.zeros(5, 4)
    present = torch.zeros(5, dtype=torch.bool)
    for slot, view in enumerate(views):
        directions[slot] = torch.from_numpy(direction_features(view['azimuth_deg'], view['elevation_deg']))
        present[slot] = True
    return {'context_camera_features': directions,
            'target_camera_features': torch.from_numpy(direction_features(target_azimuth_deg, target_elevation_deg)),
            'context_present': present}


def direction_description(view):
    return f"az={view['azimuth_deg']:+.2f}deg el={view['elevation_deg']:+.2f}deg"


def context_inputs(views, target_azimuth_deg, target_elevation_deg, *, basis):
    """Inference contract: continuous target direction, no target RGB/mask/pose."""
    inputs = continuous_conditions(views, target_azimuth_deg, target_elevation_deg)
    images = torch.zeros(5, 3, 256, 256)
    transforms = [None] * 5
    for slot, view in enumerate(views):
        images[slot], _, _, transforms[slot] = tensor_view(view, basis)
    return {**inputs, 'context_images': images, 'text': None, 'text_available': False}, transforms


class ContinuousDirectionDataset(DirectionFullBin15Dataset):
    def __init__(self, *args, text_bank=None, **kwargs):
        """text_bank: optional class_text.TextBank; adds class-prompt T5 embeddings per sample."""
        super().__init__(*args, **kwargs)
        self.text_bank = text_bank

    def __getitem__(self, index):
        # Keep view selection, invalid-view repair, image/mask transforms and supervision exact.
        batch = super().__getitem__(index)
        meta = batch['metadata']
        row = self.rows[meta['instance_uid']]
        by_id = {v['image_id']: v for v in row['views']}
        views = [by_id[i] for i in meta['context_image_ids'] if i is not None]
        target = by_id[meta['target_image_id']]
        conditions = continuous_conditions(views, target['azimuth_deg'], target['elevation_deg'])
        assert torch.equal(conditions['context_present'], batch['model_inputs']['context_present'])
        batch['model_inputs'].update(conditions)
        pad = [None] * (5 - len(views))
        meta.update(conditioning_policy=CONDITIONING_POLICY,
                    conditioning_angle_source=ANGLE_SOURCE,
                    selection_bin_deg=SELECTION_BIN_DEG,
                    context_angles_deg=[[float(v['azimuth_deg']), float(v['elevation_deg'])] for v in views] + pad,
                    target_angles_deg=[float(target['azimuth_deg']), float(target['elevation_deg'])],
                    context_descriptions=[direction_description(v) for v in views] + pad,
                    target_description=direction_description(target))
        if latent_cache.active():
            # Latents of the final (post-repair) views; absent context slots stay zero.
            ids = [i for i in meta['context_image_ids'] if i is not None]
            z = latent_cache.lookup(row['instance_uid'], ids + [meta['target_image_id']])
            context = torch.zeros(5, *latent_cache.LATENT_SHAPE)
            context[:len(ids)] = z[:len(ids)]
            batch['model_inputs']['context_latent_z'] = context
            batch['supervision']['target_latent_z'] = z[len(ids)]
        if self.text_bank is not None:
            embeds, mask = self.text_bank.lookup(row['class_id'])
            batch['model_inputs'].update(text_embeds=embeds, text_mask=mask, text_available=True,
                                         text=self.text_bank.meta['classes'][str(row['class_id'])]['prompt'])
        return batch
