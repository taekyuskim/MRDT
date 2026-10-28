"""Low-variance validation loss: the training objective on fixed held-out samples.

Every random choice of the training loss is fixed so that successive evaluations differ only
because the weights changed (common random numbers):
  * samples: the first N indices of the val split, drawn with the training sampler (same
    2-5 context rule and view repair) under a fixed seed;
  * timesteps: K stratified timesteps t_k = floor((k + 1/2) * T / K), every sample at every t_k;
  * noise: one Gaussian per (sample index, k) from a seeded CPU generator, independent of
    batch layout and world size.
SigLIP/VAE are frozen, so their outputs for the val set are computed once and reused.
The reported value is sum(per-sample MSE) / count, summed exactly across ranks (float64),
overall and per timestep.
"""
import torch
import torch.distributed as dist

from .bin15_dataset import mvimgnet_bin15_collate
from .continuous_direction_dataset import ContinuousDirectionDataset
from .train_bin15_pilot import make_train_batch

NOISE_SEED = 20261017


def stratified_timesteps(num_train_timesteps, k):
    return [int((i + 0.5) * num_train_timesteps / k) for i in range(k)]


def sample_noise(index, slot, shape):
    generator = torch.Generator().manual_seed(NOISE_SEED * 1_000_003 + index * 1009 + slot)
    return torch.randn(shape, generator=generator)


class ValLoss:
    def __init__(self, catalog, *, samples, timesteps, seed, encoder, vae, device, rank, world, chunk=16, k_min=2, text_bank=None,
                 context_latents=False):
        self.device, self.chunk = device, chunk
        dataset = ContinuousDirectionDataset(catalog, 'val', seed=seed, samples_per_epoch=samples, k_min=k_min,
                                             text_bank=text_bank)
        if samples > len(dataset.rows):
            raise ValueError('More val samples than val objects')
        self.indices = list(range(rank, samples, world))
        self.samples, self.k = samples, timesteps
        self.cache = []
        for start in range(0, len(self.indices), chunk):
            ids = self.indices[start:start + chunk]
            batch = mvimgnet_bin15_collate([dataset[i] for i in ids])
            with torch.no_grad():
                encoded = make_train_batch(batch, encoder, vae, device, context_latents=context_latents)
            self.cache.append((ids, {key: value.cpu() for key, value in encoded.items()}))

    @torch.no_grad()
    def partial(self, model):
        """This rank's per-timestep (sum of per-sample MSE, count), float64."""
        steps = stratified_timesteps(model.runner.num_train_timesteps, self.k)
        totals = torch.zeros(self.k, dtype=torch.float64, device=self.device)
        counts = torch.zeros(self.k, dtype=torch.float64, device=self.device)
        was_training = model.training
        model.eval()
        for ids, encoded in self.cache:
            batch = {key: value.to(self.device) for key, value in encoded.items()}
            for slot, t in enumerate(steps):
                noise = torch.stack([sample_noise(i, slot, batch['action_gt'].shape[1:]) for i in ids]).to(self.device)
                timesteps = torch.full((len(ids),), t, dtype=torch.long, device=self.device)
                losses = model.compute_loss(batch, noise=noise, timesteps=timesteps, reduction='none')
                totals[slot] += losses.double().sum()
                counts[slot] += len(ids)
        model.train(was_training)
        return steps, totals, counts

    @torch.no_grad()
    def evaluate(self, model):
        steps, totals, counts = self.partial(model)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals)
            dist.all_reduce(counts)
        assert int(counts[0].item()) == self.samples
        per_t = (totals / counts).tolist()
        return {'val_loss': float(totals.sum() / counts.sum()), 'samples': self.samples,
                'timesteps': steps, 'val_loss_by_timestep': per_t}
