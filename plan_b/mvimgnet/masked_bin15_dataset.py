"""Object-on-gray inputs and supervision with unchanged full-frame geometry."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps

from .full_bin15_dataset import FullBin15Dataset
from .bin15_camera import bin_description

PREPROCESS_POLICY = 'foreground_jpeg_threshold128_gray128_fullframe_v1'
BACKGROUND_RGB = (128, 128, 128)


def load_masked_rgb(image_path, mask_path, size=256):
    with Image.open(image_path) as source, Image.open(mask_path) as mask_source:
        if source.size != mask_source.size:
            raise ValueError('RGB/mask size mismatch: ' + str(image_path))
        if source.getexif().get(274, 1) not in (None, 1) or mask_source.getexif().get(274, 1) not in (None, 1):
            raise ValueError('RGB/mask EXIF geometry must be identity')
        if mask_source.mode != 'L':
            raise ValueError('Expected audited grayscale JPEG mask: ' + str(mask_path))
        rgb = source.convert('RGB')
        # JPEG values near 0/255 are codec artifacts, not calibrated alpha.
        mask = mask_source.point(lambda value: 255 if value >= 128 else 0)
        if mask.getbbox() is None:
            raise ValueError('Unexpected empty mask after audited view filtering: ' + str(mask_path))
        raw_size = rgb.size
        composite = Image.composite(rgb, Image.new('RGB', raw_size, BACKGROUND_RGB), mask)
        scale = min(size / raw_size[0], size / raw_size[1])
        resized_size = (int(round(raw_size[0] * scale)), int(round(raw_size[1] * scale)))
        offset = ((size - resized_size[0]) // 2, (size - resized_size[1]) // 2)
        canvas = Image.new('RGB', (size, size), BACKGROUND_RGB)
        canvas.paste(composite.resize(resized_size, Image.Resampling.BILINEAR), offset)
        mask_canvas = Image.new('L', (size, size), 0)
        mask_canvas.paste(mask.resize(resized_size, Image.Resampling.BOX), offset)
        valid_canvas = Image.new('L', (size, size), 0)
        valid_canvas.paste(255, (*offset, offset[0] + resized_size[0], offset[1] + resized_size[1]))
        image = torch.from_numpy(np.array(canvas, dtype=np.float32) / 255).permute(2, 0, 1).contiguous()
        foreground = torch.from_numpy(np.array(mask_canvas, dtype=np.float32) / 255).unsqueeze(0)
        valid = torch.from_numpy(np.array(valid_canvas, dtype=np.float32) / 255).unsqueeze(0)
    return image, foreground, valid


class MaskedFullBin15Dataset(FullBin15Dataset):
    def __getitem__(self, index):
        row, context_ids, target_id, target_bin = self._choose(index)
        views = {v['image_id']: v for v in row['views']}
        images = torch.zeros(5, 3, 256, 256)
        cameras = torch.zeros(5, 10)
        present = torch.zeros(5, dtype=torch.bool)
        frame_ids, descriptions, padded_ids, mask_paths = [None]*5, [None]*5, [None]*5, [None]*5
        for slot, image_id in enumerate(context_ids):
            view = views[image_id]
            images[slot] = load_masked_rgb(view['path'], view['mask_path'])[0]
            cameras[slot] = torch.tensor(view['features'], dtype=torch.float32)
            present[slot] = True
            frame_ids[slot], padded_ids[slot], mask_paths[slot] = view['frame_name'], image_id, view['mask_path']
            descriptions[slot] = bin_description(view['az_bin'], view['el_bin'])
        target = views[target_id]
        target_image, target_mask, target_valid = load_masked_rgb(target['path'], target['mask_path'])
        return {'model_inputs': {'context_images': images, 'context_camera_features': cameras,
            'context_present': present, 'target_camera_features': torch.tensor(target['features'], dtype=torch.float32),
            'text': None, 'text_available': False},
            'supervision': {'target_image': target_image, 'target_foreground_mask': target_mask, 'target_valid_region': target_valid},
            'metadata': {'dataset': 'mvimgnet2', 'split': self.split, 'instance_uid': row['instance_uid'],
                'class_id': row['class_id'], 'class_name': row['class_name'], 'target_image_id': target_id,
                'target_frame_id': target['frame_name'], 'target_bin': list(target_bin),
                'target_description': bin_description(*target_bin), 'context_image_ids': padded_ids,
                'context_frame_ids': frame_ids, 'context_descriptions': descriptions,
                'context_mask_paths': mask_paths, 'target_mask_path': target['mask_path'],
                'preprocessing_policy': PREPROCESS_POLICY,
                'conditioning_policy': 'bin15_centers_only_no_raw_pose_no_residual',
                'base_present': False, 'generation_mask': 1}}
