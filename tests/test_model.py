import torch

from turnwave.models.text_transformer import TextEOTConfig, TextEOTModel

TINY = TextEOTConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=16, dropout=0.0)


def test_forward_shape():
    model = TextEOTModel(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (5, 12))
    lengths = torch.tensor([12, 3, 7, 1, 12])
    logits = model(idx, lengths)
    assert logits.shape == (5,)


def test_causality():
    """Changing a token must not affect hidden states at earlier positions."""
    model = TextEOTModel(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (1, 10))
    with torch.no_grad():
        h_before = model.backbone(idx)
        modified = idx.clone()
        modified[0, 7] = (modified[0, 7] + 1) % TINY.vocab_size
        h_after = model.backbone(modified)
    assert torch.allclose(h_before[:, :7], h_after[:, :7], atol=1e-5)
    assert not torch.allclose(h_before[:, 7:], h_after[:, 7:], atol=1e-5)


def test_padding_does_not_leak():
    """With right padding + causal mask, pads after lengths-1 can't change the logit."""
    model = TextEOTModel(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (1, 6))
    lengths = torch.tensor([6])
    padded = torch.cat([idx, torch.randint(0, TINY.vocab_size, (1, 4))], dim=1)
    with torch.no_grad():
        assert torch.allclose(model(idx, lengths), model(padded, lengths), atol=1e-5)


def test_default_config_param_count():
    model = TextEOTModel(TextEOTConfig())
    assert 5e6 < model.num_params < 10e6
