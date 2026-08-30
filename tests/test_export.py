"""Export and inference tests. Tiny models keep ONNX tracing fast."""

import warnings
from dataclasses import asdict

import numpy as np
import pytest
import torch

from turnwave.export import check_parity, example_inputs, export_onnx, io_spec, load_any
from turnwave.infer import TurnDetector
from turnwave.models.audio_cnn import AudioEOTConfig, AudioEOTModel
from turnwave.models.text_transformer import TextEOTConfig, TextEOTModel

TEXT = TextEOTConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=32, dropout=0.0)
AUDIO = AudioEOTConfig(n_mels=32, n_frames=64, stem_channels=8, channels=(8, 16),
                       embed_dim=16, dropout=0.0)


@pytest.fixture(autouse=True)
def _quiet():
    warnings.filterwarnings("ignore")


def test_io_spec_names_match_the_task():
    assert io_spec("text")[0] == ["input_ids", "lengths"]
    assert io_spec("audio")[0] == ["mel"]
    assert io_spec("fusion")[0] == ["input_ids", "lengths", "mel"]


def test_io_spec_marks_batch_dynamic():
    for task in ("text", "audio", "fusion"):
        for name, axes in io_spec(task)[1].items():
            assert axes.get(0) == "batch", f"{task}/{name} has a fixed batch axis"


def test_text_export_parity(tmp_path):
    model = TextEOTModel(TEXT).eval()
    path = export_onnx(model, "text", tmp_path / "text.onnx")
    assert path.exists()
    assert check_parity(model, "text", path) < 1e-4


def test_audio_export_parity(tmp_path):
    model = AudioEOTModel(AUDIO).eval()
    path = export_onnx(model, "audio", tmp_path / "audio.onnx")
    assert check_parity(model, "audio", path) < 1e-4


def test_export_accepts_a_different_batch_size(tmp_path):
    """dynamic_axes must actually hold, or serving one call at a time breaks."""
    import onnxruntime as ort

    model = AudioEOTModel(AUDIO).eval()
    path = export_onnx(model, "audio", tmp_path / "audio.onnx")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for batch in (1, 7):
        mel = np.random.randn(batch, AUDIO.n_mels, AUDIO.n_frames).astype(np.float32)
        assert session.run(["logits"], {"mel": mel})[0].shape == (batch,)


def test_load_any_detects_the_task(tmp_path):
    text, audio = TextEOTModel(TEXT), AudioEOTModel(AUDIO)
    torch.save({"model": text.state_dict(), "config": asdict(TEXT), "task": "text"},
               tmp_path / "t.pt")
    torch.save({"model": audio.state_dict(), "config": asdict(AUDIO), "task": "audio"},
               tmp_path / "a.pt")
    assert load_any(tmp_path / "t.pt", torch.device("cpu"))[1] == "text"
    assert load_any(tmp_path / "a.pt", torch.device("cpu"))[1] == "audio"


def test_load_any_infers_task_for_phase1_checkpoints(tmp_path):
    """Phase 1 checkpoints predate the task field and must still load."""
    text = TextEOTModel(TEXT)
    torch.save({"model": text.state_dict(), "config": asdict(TEXT)}, tmp_path / "old.pt")
    assert load_any(tmp_path / "old.pt", torch.device("cpu"))[1] == "text"


def test_detector_reads_its_modality_from_the_graph(tmp_path):
    model = AudioEOTModel(AUDIO).eval()
    path = export_onnx(model, "audio", tmp_path / "audio.onnx")
    detector = TurnDetector(path)
    assert detector.needs_audio and not detector.needs_text

    audio = np.random.randn(AUDIO.n_mels * 100).astype(np.float32)
    probability = detector.predict(audio=audio)
    assert 0.0 <= probability <= 1.0
    assert detector.is_complete(audio=audio) == (probability >= detector.threshold)


def test_detector_rejects_missing_inputs(tmp_path):
    model = AudioEOTModel(AUDIO).eval()
    detector = TurnDetector(export_onnx(model, "audio", tmp_path / "a.onnx"))
    with pytest.raises(ValueError, match="needs audio"):
        detector.predict(text="hello")


def test_text_detector_requires_a_tokenizer(tmp_path):
    model = TextEOTModel(TEXT).eval()
    path = export_onnx(model, "text", tmp_path / "text.onnx")
    with pytest.raises(ValueError, match="tokenizer"):
        TurnDetector(path)


def test_detector_derives_the_mel_front_end_from_the_graph(tmp_path):
    """The extractor must match what the exported model accepts, not the defaults."""
    model = AudioEOTModel(AUDIO).eval()
    detector = TurnDetector(export_onnx(model, "audio", tmp_path / "a.onnx"))
    assert detector.mel.cfg.n_mels == AUDIO.n_mels
    assert detector.mel.cfg.n_frames == AUDIO.n_frames


def test_detector_rejects_a_mismatched_mel_config(tmp_path):
    from turnwave.data.features import MelConfig

    model = AudioEOTModel(AUDIO).eval()
    path = export_onnx(model, "audio", tmp_path / "a.onnx")
    with pytest.raises(ValueError, match="but the model expects"):
        TurnDetector(path, mel_config=MelConfig())  # 64x201 default vs the model's 32x64
