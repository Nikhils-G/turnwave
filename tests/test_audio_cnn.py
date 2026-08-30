import torch

from turnwave.models.audio_cnn import AudioEOTConfig, AudioEOTModel, SpecAugment

TINY = AudioEOTConfig(n_mels=32, n_frames=64, channels=(8, 16), embed_dim=32, dropout=0.0)


def test_forward_and_embed_shapes():
    model = AudioEOTModel(TINY).eval()
    mel = torch.randn(5, TINY.n_mels, TINY.n_frames)
    assert model(mel).shape == (5,)
    assert model.embed(mel).shape == (5, TINY.embed_dim)


def test_default_config_param_count_in_target_band():
    model = AudioEOTModel(AudioEOTConfig())
    assert 3e6 < model.num_params < 8e6


def test_specaugment_masks_only_in_training():
    """A masking bug that leaked into eval would corrupt every reported number."""
    torch.manual_seed(0)
    aug = SpecAugment(AudioEOTConfig(freq_mask=8, time_mask=20, n_masks=2))
    x = torch.ones(4, 64, 201)
    aug.train()
    assert (aug(x) == 0).any(), "no masking applied during training"
    aug.eval()
    assert torch.equal(aug(x), x), "masking must be inert in eval"


def test_specaugment_does_not_mutate_its_input():
    torch.manual_seed(0)
    aug = SpecAugment(AudioEOTConfig()).train()
    x = torch.ones(2, 64, 201)
    original = x.clone()
    aug(x)
    assert torch.equal(x, original)


def test_specaugment_masks_stay_in_bounds():
    """Widths are sampled per example; a start+width past the edge would silently
    mask nothing (or wrap), so check every example keeps some signal."""
    torch.manual_seed(0)
    aug = SpecAugment(AudioEOTConfig(freq_mask=8, time_mask=20)).train()
    out = aug(torch.ones(16, 64, 201))
    assert (out != 0).any(dim=2).any(dim=1).all(), "an example was fully masked"


def test_eval_is_deterministic():
    model = AudioEOTModel(TINY).eval()
    mel = torch.randn(3, TINY.n_mels, TINY.n_frames)
    with torch.no_grad():
        assert torch.equal(model(mel), model(mel))


def test_loudness_invariance():
    """Per-example standardization: a gain change must not move the prediction,
    because prosodic shape carries the signal, not recording level."""
    model = AudioEOTModel(TINY).eval()
    mel = torch.randn(3, TINY.n_mels, TINY.n_frames)
    with torch.no_grad():
        assert torch.allclose(model(mel), model(mel * 2.0 + 5.0), atol=1e-4)


def test_gradients_reach_the_first_layer():
    model = AudioEOTModel(TINY).train()
    loss = model(torch.randn(4, TINY.n_mels, TINY.n_frames)).sum()
    loss.backward()
    first = model.blocks[0].conv1.weight
    assert first.grad is not None and first.grad.abs().sum() > 0
