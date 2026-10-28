"""CPU check for the current 4D model; no dataset or downloaded weights required."""
def main():
    import models
    import rescene
    import torch
    from pathlib import Path
    from plan_b.mvimgnet.continuous_direction_dataset import continuous_conditions, direction_features
    from plan_b.mvimgnet.direction_bin15_model import DirectionBin15RDT
    from plan_b.mvimgnet.bin15_model import latent_to_patch_tokens, patch_tokens_to_latent
    from plan_b.mvimgnet.train_bin15_pilot import _tiny_rdt_config
    from models.rdt import model as rdt_module
    root = Path(__file__).resolve().parents[1]
    assert Path(rdt_module.__file__).resolve().is_relative_to(root), rdt_module.__file__
    assert Path(rescene.__file__).resolve().is_relative_to(root), rescene.__file__
    torch.set_num_threads(2)
    torch.manual_seed(0)
    conditions = continuous_conditions([{'azimuth_deg':23.25,'elevation_deg':-12.4}], 31.1, 2.2)
    assert conditions['context_present'].tolist() == [True, False, False, False, False]
    assert torch.count_nonzero(conditions['context_camera_features'][1:]) == 0
    assert not (direction_features(23.25,-12.4) == direction_features(30,-15)).all()
    z = torch.randn(2,4,32,32)
    assert torch.equal(patch_tokens_to_latent(latent_to_patch_tokens(z)),z)
    model = DirectionBin15RDT(rdt_config=_tiny_rdt_config(),img_token_dim=64,lang_token_dim=64,dtype=torch.float32)
    batch = {key: value.unsqueeze(0) for key,value in conditions.items()}
    batch['slot_feats'] = torch.randn(1,5,64,64)
    batch['action_gt'] = torch.randn(1,64,64)
    loss = model.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.target_camera_mlp[0].weight.grad is not None
    # RDT initializes its output projection to zero; upstream gradients appear
    # after that projection receives its first update.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    torch.optim.SGD(model.parameters(),lr=.01).step()
    model.zero_grad(set_to_none=True)
    model.compute_loss(batch).backward()
    assert model.target_camera_mlp[0].weight.grad.abs().sum() > 0
    model.eval()
    # Generation must not depend on the training target field.
    generation = {k:v for k,v in batch.items() if k != 'action_gt'}
    first = model.predict(generation,num_steps=2,seed=1)
    generation['slot_feats'] = generation['slot_feats'].clone()
    generation['slot_feats'][:,1:] = 10000
    second = model.predict(generation,num_steps=2,seed=1)
    assert first.shape == (1,64,64) and torch.isfinite(first).all()
    assert torch.allclose(first,second,atol=1e-5), 'Padded images influenced output'
    print('Current 4D CPU loss/backward/sampling, padding, continuous angles, latent roundtrip, local imports: passed')

if __name__ == '__main__':main()
