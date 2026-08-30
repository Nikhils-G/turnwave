"""Adapter tests, exercised against the harness's documented item shape."""

import warnings

import numpy as np
import pytest
import torch

from turnwave.eot_bench_adapter import (
    SCORE_POINT,
    TurnWaveAdapter,
    TurnWaveAudioOnlyAdapter,
)
from turnwave.export import export_onnx
from turnwave.models.audio_cnn import AudioEOTConfig, AudioEOTModel

AUDIO = AudioEOTConfig(n_mels=32, n_frames=64, stem_channels=8, channels=(8, 16),
                       embed_dim=16, dropout=0.0)


@pytest.fixture(autouse=True)
def _quiet():
    warnings.filterwarnings("ignore")


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    path = export_onnx(AudioEOTModel(AUDIO).eval(), "audio", tmp_path / "audio.onnx")
    monkeypatch.setenv("TURNWAVE_ONNX", str(path))
    monkeypatch.delenv("TURNWAVE_TOKENIZER", raising=False)
    return TurnWaveAudioOnlyAdapter()


def item(seconds=1.5, sample_rate=16000, text="i want to order a"):
    """The shape eot_harness passes to predict_batch."""
    return {
        "language": "en",
        "audio": {"array": np.random.randn(int(seconds * sample_rate)).astype(np.float32),
                  "sampling_rate": sample_rate},
        "messages": [{"role": "assistant", "content": "how can i help"},
                     {"role": "user", "content": text}],
    }


def test_score_point_matches_the_reference_models():
    """0.2 is what LiveKit v1 and SmartTurn declare; a different value would put
    us in a different operating regime under the same column heading."""
    assert SCORE_POINT == 0.2
    assert TurnWaveAdapter.score_point == 0.2


def test_adapter_interface_is_what_the_harness_validates():
    assert isinstance(TurnWaveAdapter.adapter_id, str) and TurnWaveAdapter.adapter_id
    assert callable(TurnWaveAdapter.predict_batch)


def test_predict_batch_returns_one_probability_per_item(adapter):
    batch = [item(), item(0.4), item(3.0)]
    probabilities = adapter.predict_batch(batch)
    assert len(probabilities) == len(batch)
    assert all(isinstance(p, float) and 0.0 <= p <= 1.0 for p in probabilities)


def test_short_prefix_is_accepted(adapter):
    """Decision points early in a turn give less audio than the window; the front
    end left-pads rather than failing."""
    assert 0.0 <= adapter.predict_batch([item(0.05)])[0] <= 1.0


def test_declares_the_audio_it_needs(adapter):
    assert adapter.max_audio_sec == pytest.approx(AUDIO.n_frames * 0.01, abs=0.02)


def test_language_gate_is_english_only(adapter):
    assert adapter.supports_language("en")
    assert not adapter.supports_language("hi")


def test_sample_rate_mismatch_is_loud(adapter):
    """Silently resampling-by-ignoring would corrupt every prosodic feature."""
    with pytest.raises(ValueError, match="Hz"):
        adapter.predict_batch([item(sample_rate=8000)])


def test_missing_model_path_fails_fast(monkeypatch):
    monkeypatch.delenv("TURNWAVE_ONNX", raising=False)
    with pytest.raises(ValueError, match="TURNWAVE_ONNX"):
        TurnWaveAdapter()


def test_current_utterance_prefers_the_trailing_user_message():
    messages = [{"role": "user", "content": "earlier turn"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "the words so far"}]
    assert TurnWaveAdapter._current_utterance(messages) == "the words so far"


def test_current_utterance_handles_an_empty_history():
    assert TurnWaveAdapter._current_utterance([]) == ""
    assert TurnWaveAdapter._current_utterance(None) == ""
