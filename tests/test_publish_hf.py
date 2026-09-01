"""Model-card generation tests.

The card is a public artifact. Its licence claims must match the training data
each model actually used, and its verdict must follow from its numbers rather
than be written alongside them.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import publish_hf  # noqa: E402


def make_onnx_dir(tmp_path: Path, keys=("audio_eot_v2", "audio_eot", "text_eot", "fusion_eot_v2")):
    onnx = tmp_path / "onnx"
    onnx.mkdir()
    recommended = {"text_eot": "int8"}
    for key in keys:
        variant = recommended.get(key, "fp32")
        suffix = ".int8.onnx" if variant == "int8" else ".onnx"
        (onnx / f"{key}{suffix}").write_bytes(b"fake onnx")
        (onnx / f"{key}.export.json").write_text(json.dumps({
            "task": "audio",
            "fp32": {"path": f"checkpoints/onnx/{key}.onnx", "mb": 14.0, "latency_ms": 4.8},
            "int8": {"path": f"checkpoints/onnx/{key}.int8.onnx", "mb": 3.5, "latency_ms": 30.9},
            "recommended": variant,
        }))
    return onnx


def test_ships_the_variant_the_export_measured_as_faster(tmp_path):
    onnx = make_onnx_dir(tmp_path)
    assert publish_hf.shipping_file(onnx, "text_eot").name == "text_eot.int8.onnx"
    assert publish_hf.shipping_file(onnx, "audio_eot_v2").name == "audio_eot_v2.onnx"


def test_missing_export_report_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="run turnwave.export"):
        publish_hf.shipping_file(tmp_path, "audio_eot_v2")


def test_no_models_is_an_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no exported models"):
        publish_hf.model_table(empty)


def test_licences_follow_the_training_data(tmp_path):
    """DailyDialog is CC BY-NC-SA 4.0, so the models derived from it cannot be
    offered as Apache-2.0 -- that would be a false claim on a public artifact."""
    table, _ = publish_hf.model_table(make_onnx_dir(tmp_path))
    rows = {line.split("|")[1].strip(): line.split("|")[2].strip()
            for line in table.splitlines() if line.startswith("| `")}
    assert rows["`audio_eot_v2.onnx`"] == "apache-2.0"
    assert rows["`audio_eot.onnx`"] == "cc-by-4.0"
    assert rows["`text_eot.int8.onnx`"] == "cc-by-nc-sa-4.0"
    assert rows["`fusion_eot_v2.onnx`"] == "cc-by-nc-sa-4.0"


def test_non_commercial_models_are_never_labelled_permissive(tmp_path):
    table, _ = publish_hf.model_table(make_onnx_dir(tmp_path))
    for line in table.splitlines():
        if "text_eot" in line or "fusion_eot_v2" in line:
            assert "apache" not in line.lower()
            assert "non-commercial" in line.lower() or "nc" in line.lower()


def test_benchmark_table_bolds_the_best_per_column(tmp_path):
    table = publish_hf.bench_table()
    livekit = next(l for l in table.splitlines() if "LiveKit" in l)
    assert "**9.9%**" in livekit and "**543 ms**" in livekit  # best in every column
    vad = next(l for l in table.splitlines() if "VAD" in l)
    assert "**" not in vad.replace("| VAD baseline |", "")  # worst, so never bolded


def test_our_row_is_identified_and_present(tmp_path):
    ours = next(l for l in publish_hf.bench_table().splitlines() if "this model" in l)
    assert ours.startswith("| **") and "42.1%" in ours


def test_card_claim_follows_the_numbers(tmp_path):
    card, _ = publish_hf.build_card(make_onnx_dir(tmp_path), "x/y")
    beats_vad = publish_hf.BENCH["cutoff_300"] < publish_hf.BENCH["baselines"][0][1]
    assert ("beats the VAD baseline" in card) is beats_vad


def test_card_carries_the_model_index_and_metrics(tmp_path):
    """model-index is what makes the Hub render evaluation results; neither
    competitor ships one."""
    card, _ = publish_hf.build_card(make_onnx_dir(tmp_path), "x/y")
    assert "model-index:" in card
    assert "livekit/eot-bench-data" in card
    assert str(publish_hf.BENCH["auc"]) in card


def test_card_uses_the_singular_language_key(tmp_path):
    """HF indexes `language:`; `languages:` is silently ignored -- the mistake
    that left a competitor's multilingual model with zero language tags."""
    card, _ = publish_hf.build_card(make_onnx_dir(tmp_path), "x/y")
    assert "\nlanguage:\n" in card
    assert "\nlanguages:" not in card


def test_card_states_the_limitations(tmp_path):
    card, _ = publish_hf.build_card(make_onnx_dir(tmp_path), "x/y")
    for expected in ("English only", "Non-commercial",
                     "Fusion wins in-domain, not on the benchmark"):
        assert expected in card


def test_uploads_resolve_to_real_files(tmp_path):
    _, uploads = publish_hf.build_card(make_onnx_dir(tmp_path), "x/y")
    assert uploads and all(path.exists() for path in uploads.values())
