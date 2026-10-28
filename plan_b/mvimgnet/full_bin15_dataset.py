"""Read full bin15 metadata lazily from a read-only, per-worker SQLite handle."""
import json
import os
import sqlite3
from collections import OrderedDict
from pathlib import Path

import torch
from .bin15_camera import FixedCameraRow, bin_description, occupied_pair_bins
from .bin15_dataset import MVImgNetBin15Stage1Dataset, _load_rgb
from .epoch_sampling import EpochPermutation


class Catalog:
    def __init__(self, path, split):
        self.path, self.split = str(path), split
        self.connection = None
        self.pid = None
        self.cache = OrderedDict()
        label_path = Path(path).parent / 'mvimgnet2_category.json'
        self.class_names = json.loads(label_path.read_text()) if label_path.exists() else {}
        self.count = self.db().execute('SELECT count(*) FROM instances WHERE split=?', (split,)).fetchone()[0]
        if not self.count:
            raise ValueError('Empty catalog split:' + split)

    def db(self):
        if self.connection is None or self.pid != os.getpid():
            if self.connection is not None:
                self.connection.close()
            self.connection = sqlite3.connect(Path(self.path).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
            self.connection.execute('PRAGMA cache_size=-8192')
            self.pid = os.getpid()
            self.cache.clear()
        return self.connection

    def __getstate__(self):
        return {**self.__dict__, 'connection': None, 'pid': None, 'cache': OrderedDict()}

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        db = self.db()
        key = ('ordinal', index) if isinstance(index, int) else ('uid', index)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if key[0] == 'ordinal':
            found = db.execute('SELECT row_json FROM instances WHERE split=? AND ordinal=?', (self.split, index)).fetchone()
        else:
            found = db.execute('SELECT row_json FROM instances WHERE split=? AND uid=?', (self.split, index)).fetchone()
        if not found:
            raise KeyError(index)
        row = json.loads(found[0])
        row['class_name'] = self.class_names.get(row['class_id'], row.get('class_name'))
        self.cache[key] = row
        self.cache[('uid', row['instance_uid'])] = row
        while len(self.cache) > 128:
            self.cache.popitem(last=False)
        return row


class FixedRows:
    def __init__(self, catalog):
        self.catalog = catalog

    def __getitem__(self, uid):
        return {v['image_id']: FixedCameraRow(uid, v['image_id'], v['frame_name'],
            v['azimuth_deg'], v['elevation_deg'], v['radius_refnorm'], v['az_bin'], v['el_bin'],
            bin_description(v['az_bin'], v['el_bin'])) for v in self.catalog[uid]['views']}


class FullBin15Dataset(MVImgNetBin15Stage1Dataset):
    def __init__(self, catalog, split, *, deterministic=False, seed=0, samples_per_epoch=None, k_min=2):
        self.rows = Catalog(catalog, split)
        self.fixed = FixedRows(self.rows)
        self.split, self.deterministic, self.seed = split, deterministic, seed
        if not 1 <= int(k_min) <= 5:
            raise ValueError('k_min must be in [1, 5]')
        self.epoch, self.image_size, self.k_min, self.k_max = 0, 256, int(k_min), 5
        self.samples_per_epoch = samples_per_epoch or len(self.rows)
        self.object_order = EpochPermutation(len(self.rows), seed)

    def _choose(self, index):
        if self.deterministic:
            return super()._choose(index)
        # RankSamples supplies absolute sample indices, including after resume.
        # Epoch boundaries keep every object, even when N is not divisible by batch32.
        row = self.rows[self.object_order[index]]
        rng = self._rng(index)
        fixed_rows = self.fixed[row['instance_uid']]
        bins = occupied_pair_bins(fixed_rows[int(i)] for i in row['valid_image_ids'])
        usable = {b: ids for b, ids in bins.items() if ids}
        if len(usable) < self.k_min + 1:
            raise ValueError('Too few occupied bins: ' + row['instance_uid'])
        target_bin = rng.choice(list(usable))
        target_id = rng.choice(usable[target_bin])
        context_bins = [b for b in usable if b != target_bin]
        k = rng.randint(self.k_min, min(self.k_max, len(context_bins)))
        context_ids = [rng.choice(usable[b]) for b in rng.sample(context_bins, k)]
        assert len(set(context_ids + [target_id])) == len(context_ids) + 1
        return row, context_ids, target_id, target_bin

    def __getitem__(self, index):
        row, context_ids, target_id, target_bin = self._choose(index)
        views = {v['image_id']: v for v in row['views']}
        images = torch.zeros(5, 3, 256, 256)
        cameras = torch.zeros(5, 10)
        present = torch.zeros(5, dtype=torch.bool)
        frame_ids, descriptions, padded_ids = [None] * 5, [None] * 5, [None] * 5
        for slot, image_id in enumerate(context_ids):
            view = views[image_id]
            images[slot] = _load_rgb(Path(view['path']))[0]
            cameras[slot] = torch.tensor(view['features'], dtype=torch.float32)
            present[slot] = True
            frame_ids[slot], padded_ids[slot] = view['frame_name'], image_id
            descriptions[slot] = bin_description(view['az_bin'], view['el_bin'])
        target = views[target_id]
        return {'model_inputs': {'context_images': images, 'context_camera_features': cameras,
            'context_present': present, 'target_camera_features': torch.tensor(target['features'], dtype=torch.float32),
            'text': None, 'text_available': False},
            'supervision': {'target_image': _load_rgb(Path(target['path']))[0]},
            'metadata': {'dataset': 'mvimgnet2', 'split': self.split, 'instance_uid': row['instance_uid'],
                'class_id': row['class_id'], 'class_name': row['class_name'], 'target_image_id': target_id,
                'target_frame_id': target['frame_name'], 'target_bin': list(target_bin),
                'target_description': bin_description(*target_bin), 'context_image_ids': padded_ids,
                'context_frame_ids': frame_ids, 'context_descriptions': descriptions,
                'conditioning_policy': 'bin15_centers_only_no_raw_pose_no_residual',
                'base_present': False, 'generation_mask': 1}}
