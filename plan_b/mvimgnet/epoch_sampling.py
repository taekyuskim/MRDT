"""Deterministic object permutations addressed by the global sample position."""
import random
from collections import OrderedDict

SAMPLING_POLICY = 'all_objects_once_per_epoch_python_shuffle_v1'


class EpochPermutation:
    def __init__(self, count, seed=0):
        if count <= 0:
            raise ValueError('Object count must be positive')
        self.count, self.seed = count, seed
        self.cache = OrderedDict()

    def __getitem__(self, global_index):
        if global_index < 0:
            raise IndexError(global_index)
        epoch, offset = divmod(global_index, self.count)
        if epoch not in self.cache:
            order = list(range(self.count))
            random.Random(f'mvimgnet2-object-epoch-v1:{self.seed}:{epoch}').shuffle(order)
            self.cache[epoch] = order
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(epoch)
        return self.cache[epoch][offset]


class RankSamples:
    def __init__(self, start, stop, batch, rank, world):
        if not (0 <= start <= stop and batch > 0 and 0 <= rank < world):
            raise ValueError('Invalid sample range')
        self.start, self.stop, self.batch, self.rank, self.world = start, stop, batch, rank, world

    def __iter__(self):
        for step in range(self.start, self.stop):
            first = step * self.batch * self.world + self.rank * self.batch
            yield from range(first, first + self.batch)

    def __len__(self):
        return (self.stop - self.start) * self.batch
