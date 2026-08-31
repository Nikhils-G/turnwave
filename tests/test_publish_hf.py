"""Model-card generation tests.

The card is a public artifact; its claims must follow from its numbers rather
than be written alongside them.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import publish_hf  # noqa: E402


def write_exports(tmp_path: Path):
    onnx = tmp_path / "onnx"
    onnx.mkdir()
    for name, recommended in (("text_eot", "int8"), ("audio_eot", "fp32"),
                              ("fusion_eot", "fp32")):
        (onnx / f"{name}.export.json").write_text(json.dumps({
            "task": name.replace("_eot", ""),
            "fp32": {"path": f"checkpoints/onnx/{name}.onnx", "mb": 14.0, "latency_ms": 4.6},
            "int8": {"path": f"checkpoints/onnx/{name}.int8.onnx", "mb": 3.5, "latency_ms": 31.2},
            "recommended": recommended,
        }))
    return onnx


def write_ablation(tmp_path: Path, fused_ap: float, audio_ap: float = 0.741):
    path = tmp_path / "ablation.json"
    metrics = lambda ap: {"accuracy": 0.6, "precision": 0.6, "recall": 0.6, "f1": 0.6, "ap": ap}
    path.write_text(json.dumps({
        "split": "test", "examples": 6000,
        "results": {"text only": metrics(0.633), "audio only": metrics(audio_ap),
                    "fused (text + audio)": metrics(fused_ap)},
    }))
    return path


def test_ships_the_variant_the_export_measured_as_faster(tmp_path):
    onnx = write_exports(tmp_path)
    assert publish_hf.shipping_file(onnx, "text_eot").name == "text_eot.int8.onnx"
    assert publish_hf.shipping_file(onnx, "audio_eot").name == "audio_eot.onnx"


def test_missing_export_report_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="run turnwave.export"):
        publish_hf.shipping_file(tmp_path, "text_eot")


def test_refuses_to_build_a_card_without_results(tmp_path):
    """A Results section reading 'not available' beside a claim that fusion wins
    would be worse than shipping no card."""
    onnx = write_exports(tmp_path)
    with pytest.raises(SystemExit, match="no ablation results"):
        publish_hf.build_tables(onnx, tmp_path / "absent.json")


def test_claim_follows_the_numbers_when_fusion_wins(tmp_path):
    onnx = write_exports(tmp_path)
    _, _, _, claim = publish_hf.build_tables(onnx, write_ablation(tmp_path, fused_ap=0.767))
    assert "Fusion beats both branches" in claim


def test_claim_reverses_when_fusion_loses(tmp_path):
    """The card must not assert a win the measurements do not support."""
    onnx = write_exports(tmp_path)
    _, _, _, claim = publish_hf.build_tables(onnx, write_ablation(tmp_path, fused_ap=0.700))
    assert "did **not** beat" in claim
    assert "Fusion beats both branches" not in claim


def test_results_table_carries_every_row(tmp_path):
    onnx = write_exports(tmp_path)
    results, latency, text_ap, _ = publish_hf.build_tables(onnx, write_ablation(tmp_path, 0.767))
    for label in ("text only", "audio only", "fused (text + audio)"):
        assert label in results
    assert "6,000 examples" in results
    assert text_ap == "0.633"
    for name in ("text", "audio", "fusion"):
        assert f"| {name} |" in latency
