"""All-object epoch sampler with canonical RGB and direction-only camera inputs."""
import cv2
import torch
from .full_bin15_dataset import FullBin15Dataset
from .bin15_camera import bin_description
from .canonical_object import load_canonical_view, direction_features, PREPROCESS_POLICY, CONDITIONING_POLICY, MissingCanonicalForeground

INVALID_VIEW_POLICY = 'missing_foreground_same_object_same_bin_first_v1'


def tensor_view(view):
    cv2.setNumThreads(1)
    color, mask, valid, info = load_canonical_view(view)
    return (torch.from_numpy(color.copy()).permute(2,0,1).float()/255,
            torch.from_numpy(mask.copy()).unsqueeze(0),
            torch.from_numpy(valid.copy()).unsqueeze(0), info)


def context_inputs(views, target_az_bin, target_el_bin):
    """Shared inference/training contract; no target image or target mask."""
    if not 1 <= len(views) <= 5:
        raise ValueError('Expected one to five context views')
    images = torch.zeros(5,3,256,256)
    directions = torch.zeros(5,4)
    present = torch.zeros(5,dtype=torch.bool)
    transforms = [None]*5
    for slot, view in enumerate(views):
        images[slot], _, _, transforms[slot] = tensor_view(view)
        directions[slot] = torch.from_numpy(direction_features(view['az_bin'],view['el_bin']))
        present[slot] = True
    return {'context_images':images, 'context_camera_features':directions,
            'target_camera_features':torch.from_numpy(direction_features(target_az_bin,target_el_bin)),
            'context_present':present, 'text':None, 'text_available':False}, transforms


def usable_selection(row, context_ids, target_id, loader=tensor_view):
    """Keep the original selection unless its mask cannot produce foreground.

    Alternatives come from this object only, prefer the same camera bin, and
    preserve distinct context/target bins. No model or global RNG is consumed.
    Unexpected I/O, camera, and programming errors still fail with a checkpoint.
    """
    views = {v['image_id']:v for v in row['views']}
    valid_ids = sorted(int(i) for i in row['valid_image_ids'])
    cache, rejected, replacements = {}, {}, []

    def camera_bin(image_id):
        view = views[image_id]
        return view['az_bin'], view['el_bin']

    def select(original_id, forbidden_bins, role):
        preferred = camera_bin(original_id)
        ordered = [original_id] + sorted((i for i in valid_ids if i != original_id),
                                         key=lambda i:(camera_bin(i) != preferred, i))
        for image_id in ordered:
            if camera_bin(image_id) in forbidden_bins or image_id in rejected:
                continue
            if image_id not in cache:
                try:
                    cache[image_id] = loader(views[image_id])
                except MissingCanonicalForeground as error:
                    rejected[image_id] = {'image_id':image_id, 'frame':views[image_id]['frame_name'],
                        'reason':str(error), 'foreground_fraction':views[image_id].get('foreground_fraction')}
                    continue
            if image_id != original_id:
                replacements.append({'role':role, 'from_image_id':original_id, 'to_image_id':image_id,
                    'from_bin':list(preferred), 'to_bin':list(camera_bin(image_id))})
            return image_id
        return None

    # Reserve existing context bins so a repair changes as few views as possible.
    selected_target = select(target_id, {camera_bin(i) for i in context_ids}, 'target')
    if selected_target is None:
        selected_target = select(target_id, set(), 'target')
    if selected_target is None:
        raise MissingCanonicalForeground('No usable target foreground in object '+row['instance_uid'])
    occupied = {camera_bin(selected_target)}
    selected_context = []
    for slot, original_id in enumerate(context_ids):
        remaining_bins = {camera_bin(i) for i in context_ids[slot+1:]}
        replacement = select(original_id, occupied | remaining_bins, 'context_'+str(slot))
        if replacement is None:
            replacement = select(original_id, occupied, 'context_'+str(slot))
        if replacement is None:
            replacements.append({'role':'context_'+str(slot), 'from_image_id':original_id,
                                 'to_image_id':None, 'reason':'no_unused_usable_camera_bin'})
            continue
        selected_context.append(replacement);occupied.add(camera_bin(replacement))
    if len(selected_context) < 2:
        raise MissingCanonicalForeground('Fewer than two usable context bins in object '+row['instance_uid'])
    return selected_context, selected_target, cache, list(rejected.values()), replacements


class DirectionFullBin15Dataset(FullBin15Dataset):
    def __getitem__(self,index):
        row, context_ids, target_id, target_bin = self._choose(index)
        views = {v['image_id']:v for v in row['views']}
        context_ids, target_id, loaded, rejected, replacements = usable_selection(row, context_ids, target_id)
        target = views[target_id]
        target_bin = (target['az_bin'], target['el_bin'])
        images = torch.zeros(5,3,256,256)
        directions = torch.zeros(5,4)
        present = torch.zeros(5,dtype=torch.bool)
        transforms = [None]*5
        for slot, image_id in enumerate(context_ids):
            images[slot], _, _, transforms[slot] = loaded[image_id]
            directions[slot] = torch.from_numpy(direction_features(views[image_id]['az_bin'],views[image_id]['el_bin']))
            present[slot] = True
        inputs = {'context_images':images, 'context_camera_features':directions,
            'target_camera_features':torch.from_numpy(direction_features(*target_bin)),
            'context_present':present, 'text':None, 'text_available':False}
        target_rgb, target_mask, valid, target_transform = loaded[target_id]
        pad = [None]*(5-len(context_ids))
        return {'model_inputs':inputs,
            'supervision':{'target_image':target_rgb,'target_foreground_mask':target_mask,'target_valid_region':valid},
            'metadata':{'instance_uid':row['instance_uid'],'class_id':row['class_id'],'class_name':row['class_name'],
                'split':self.split,'target_image_id':target_id,'target_frame_id':target['frame_name'],
                'target_bin':list(target_bin),'target_description':bin_description(*target_bin),
                'context_image_ids':context_ids+pad,
                'context_frame_ids':[views[i]['frame_name'] for i in context_ids]+pad,
                'context_descriptions':[bin_description(views[i]['az_bin'],views[i]['el_bin']) for i in context_ids]+pad,
                'context_transforms':transforms,'target_transform':target_transform,
                'preprocessing_policy':PREPROCESS_POLICY,'conditioning_policy':CONDITIONING_POLICY,
                'invalid_view_policy':INVALID_VIEW_POLICY,'rejected_views':rejected,'view_replacements':replacements}}
