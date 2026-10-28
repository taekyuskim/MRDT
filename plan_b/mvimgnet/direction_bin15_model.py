"""Four direction features; no distance, intrinsic, distortion or size input."""
from .bin15_model import Bin15CameraRDT
from .canonical_object import CONDITIONING_POLICY


class DirectionBin15RDT(Bin15CameraRDT):
    def __init__(self,**kwargs):
        if kwargs.pop('camera_feature_dim',4) != 4:
            raise ValueError('Direction model requires four camera features')
        super().__init__(camera_feature_dim=4,**kwargs)

    def _conditions(self,**kwargs):
        for key in ['context_camera_features','target_camera_features']:
            if kwargs[key].shape[-1] != 4:
                raise ValueError('Direction-only contract violation: '+key)
        return super()._conditions(**kwargs)

    def warm_start(self,state):
        converted = dict(state)
        keys = ['context_camera_mlp.0.weight','target_camera_mlp.0.weight']
        for key in keys:
            if converted[key].shape[1] != 10:
                raise ValueError('Expected baseline10D weight: '+key)
            converted[key] = converted[key][:,:4].clone()
        self.load_state_dict(converted,strict=True)
        return {'policy':'retain_backbone_and_direction_columns_drop_nondirection_columns',
                'converted_keys':keys,'removed_columns':[4,5,6,7,8,9],
                'optimizer':'fresh','conditioning_policy':CONDITIONING_POLICY}
