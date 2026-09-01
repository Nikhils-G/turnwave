"""Transcription script tests. No Whisper, no network — the properties under test
are resumability and output hygiene, not ASR quality."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import transcribe_clips  # noqa: E402


def test_done_ids_resume_set(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"id": "a", "text": "x"}\n{"id": "b", "text": ""}\n')
    assert transcribe_clips.load_done_ids(path) == {"a", "b"}


def test_done_ids_missing_file_is_empty(tmp_path):
    assert transcribe_clips.load_done_ids(tmp_path / "nope.jsonl") == set()


def test_done_ids_skip_torn_tail(tmp_path):
    """The whole point of append-only output: a killed session leaves a partial
    last line, and that clip is simply redone rather than blocking the restart."""
    path = tmp_path / "t.jsonl"
    path.write_text('{"id": "a", "text": "x"}\n{"id": "b", "tex')
    assert transcribe_clips.load_done_ids(path) == {"a"}


def test_transcribe_array_downmixes_and_joins_segments():
    import numpy as np

    class Segment:
        def __init__(self, text):
            self.text = text

    class FakeModel:
        def transcribe(self, wav, **kwargs):
            assert wav.ndim == 1, "stereo must be downmixed before Whisper"
            return [Segment(" hello "), Segment("there ")], None

    stereo = np.zeros((1600, 2), dtype=np.float32)
    assert transcribe_clips.transcribe_array(FakeModel(), stereo, 16000) == "hello there"


def test_wrong_sample_rate_yields_empty():
    """The corpus is 16 kHz throughout; a different rate marks a bad row, and
    silently transcribing resampled-wrong audio would poison the text quietly."""

    class Boom:
        def transcribe(self, *a, **k):
            raise AssertionError("must not be called")

    import numpy as np

    assert transcribe_clips.transcribe_array(Boom(), np.zeros(8000), 8000) == ""


def test_language_filter_is_available():
    """Eval clips come from the multilingual v3.2 test repo; without a filter the
    GPU would transcribe ~75% non-English rows that the cache build then drops."""
    import argparse

    # smoke the CLI surface rather than the network path
    ap_source = open(transcribe_clips.__file__).read()
    assert "--languages" in ap_source and 'row.get("language")' in ap_source


def test_build_model_falls_back_when_float16_unsupported(monkeypatch, capsys):
    """A P100 rejects float16; the script must degrade to int8, not die."""
    import sys
    import types

    calls = []

    class FakeWhisperModel:
        def __init__(self, name, device=None, compute_type=None):
            calls.append(compute_type)
            if compute_type == "float16":
                raise ValueError("Requested float16 compute type, but ...")

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)

    transcribe_clips.build_model("base.en", "cuda")
    assert calls == ["float16", "int8_float16"]
