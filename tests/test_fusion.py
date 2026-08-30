import torch

from turnwave.models.audio_cnn import AudioEOTConfig, AudioEOTModel
from turnwave.models.fusion import FusionConfig, FusionEOTModel, build_fusion
from turnwave.models.text_transformer import TextEOTConfig, TextEOTModel

TEXT = TextEOTConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=16, dropout=0.0)
AUDIO = AudioEOTConfig(n_mels=16, n_frames=32, channels=(8, 16), embed_dim=24, dropout=0.0)


def make(freeze=True):
    return FusionEOTModel(TextEOTModel(TEXT), AudioEOTModel(AUDIO),
                          FusionConfig(hidden_dim=32, dropout=0.0, freeze_branches=freeze))


def batch(n=4):
    return (torch.randint(0, TEXT.vocab_size, (n, 12)),
            torch.tensor([12, 8, 3, 1][:n]),
            torch.randn(n, AUDIO.n_mels, AUDIO.n_frames))


def test_forward_shape():
    assert make().eval()(*batch()).shape == (4,)


def test_frozen_branches_leave_only_the_head_trainable():
    model = make(freeze=True)
    head_params = sum(p.numel() for p in model.head.parameters())
    assert model.num_trainable_params == head_params
    assert model.num_params > head_params


def test_unfreeze_makes_everything_trainable():
    model = make(freeze=False)
    assert model.num_trainable_params == model.num_params


def test_frozen_branches_stay_in_eval_during_training():
    """Otherwise BatchNorm keeps updating and the fused model drifts away from
    the branches the ablation says it started from."""
    model = make(freeze=True).train()
    assert model.training
    assert not model.text.training
    assert not model.audio.training


def test_training_does_not_change_frozen_branch_weights():
    torch.manual_seed(0)
    model = make(freeze=True).train()
    before = model.audio.blocks[0].conv1.weight.clone()
    text_before = model.text.tok_emb.weight.clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        loss = model(*batch()).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert torch.equal(model.audio.blocks[0].conv1.weight, before)
    assert torch.equal(model.text.tok_emb.weight, text_before)


def test_head_receives_gradient():
    model = make(freeze=True).train()
    model(*batch()).sum().backward()
    grads = [p.grad for p in model.head.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_output_depends_on_both_modalities():
    """A fused model that ignores one input is a bug the ablation would hide."""
    torch.manual_seed(0)
    model = make().eval()
    idx, lengths, mel = batch()
    with torch.no_grad():
        base = model(idx, lengths, mel)
        other_audio = model(idx, lengths, torch.randn_like(mel))
        other_text = model(torch.randint(0, TEXT.vocab_size, idx.shape), lengths, mel)
    assert not torch.allclose(base, other_audio, atol=1e-5), "audio input ignored"
    assert not torch.allclose(base, other_text, atol=1e-5), "text input ignored"


def test_build_fusion_from_checkpoints(tmp_path):
    from dataclasses import asdict

    text, audio = TextEOTModel(TEXT), AudioEOTModel(AUDIO)
    torch.save({"model": text.state_dict(), "config": asdict(TEXT)}, tmp_path / "text.pt")
    torch.save({"model": audio.state_dict(), "config": asdict(AUDIO)}, tmp_path / "audio.pt")

    fused = build_fusion(tmp_path / "text.pt", tmp_path / "audio.pt",
                         FusionConfig(hidden_dim=32), torch.device("cpu"))
    assert torch.equal(fused.text.tok_emb.weight, text.tok_emb.weight)
    assert torch.equal(fused.audio.blocks[0].conv1.weight, audio.blocks[0].conv1.weight)
    assert fused.eval()(*batch()).shape == (4,)


def test_fusion_checkpoint_round_trip(tmp_path):
    """A fusion checkpoint must carry its branch configs, or the weights inside it
    cannot be given a shape to load into."""
    from dataclasses import asdict

    import pytest

    from turnwave.models.fusion import load_fusion

    model = make().eval()
    payload = {
        "model": model.state_dict(),
        "config": asdict(model.cfg),
        "text_config": asdict(model.text.cfg),
        "audio_config": asdict(model.audio.cfg),
        "task": "fusion",
    }
    torch.save(payload, tmp_path / "fusion.pt")
    restored = load_fusion(tmp_path / "fusion.pt", torch.device("cpu"))

    idx, lengths, mel = batch()
    with torch.no_grad():
        assert torch.allclose(restored(idx, lengths, mel), model(idx, lengths, mel), atol=1e-6)

    torch.save({"model": model.state_dict(), "config": asdict(model.cfg)}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="not a fusion checkpoint"):
        load_fusion(tmp_path / "bad.pt", torch.device("cpu"))
