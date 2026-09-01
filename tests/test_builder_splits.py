"""Regression test for the eval-split carve.

validation and test are cut from one upstream repo by offset. When a language
filter is active, an offset counted in raw rows lands validation and test on
overlapping clips — which happened, and only the notebook's leakage gate caught
it before training. The offset must count usable examples.
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_audio_dataset  # noqa: E402
from turnwave.data.features import LogMel, MelConfig  # noqa: E402


def wav_bytes(seconds=0.3, sr=16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.random.default_rng(0).normal(scale=0.1, size=int(sr * seconds)), sr,
             format="WAV")
    return buf.getvalue()


class FakeStream:
    """Stands in for a streamed HF dataset: iterable of rows, cast is a no-op."""

    def __init__(self, rows):
        self.rows = rows

    def cast_column(self, *a, **k):
        return self

    def __iter__(self):
        return iter(self.rows)


@pytest.fixture
def multilingual_stream(monkeypatch):
    """Every other row is English; ids expose stream order."""
    audio = wav_bytes()
    rows = []
    for i in range(24):
        rows.append({"id": f"clip{i:02d}", "language": "eng" if i % 2 == 0 else "zho",
                     "endpoint_bool": i % 4 == 0, "synthetic": False,
                     "dataset": "fake", "audio": {"bytes": audio}})
    monkeypatch.setattr(build_audio_dataset, "load_dataset",
                        lambda *a, **k: FakeStream(rows))
    return rows


def build(tmp_path, name, cap, skip, mel):
    return build_audio_dataset.build_split(
        "fake/repo", None, "train", tmp_path, cap, mel,
        source="smart-turn", languages={"eng"}, quiet=True,
        split_name=name, skip=skip)


def test_offset_carves_do_not_overlap_under_a_language_filter(tmp_path, multilingual_stream):
    mel = LogMel(MelConfig())
    build(tmp_path, "validation", cap=4, skip=0, mel=mel)
    build(tmp_path, "test", cap=4, skip=4, mel=mel)

    val = {json.loads(l)["id"] for l in open(tmp_path / "validation.jsonl")}
    test = {json.loads(l)["id"] for l in open(tmp_path / "test.jsonl")}
    assert val == {"clip00", "clip02", "clip04", "clip06"}
    assert test == {"clip08", "clip10", "clip12", "clip14"}
    assert not val & test, "eval splits overlap: skip counted raw rows, not examples"
