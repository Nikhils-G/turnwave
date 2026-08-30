"""Export a trained branch to ONNX, quantize to INT8, and verify parity.

A turn detector runs on every pause in every call, so it has to be cheap and it
has to run where the audio pipeline already is — CPU, no GPU, alongside VAD and
ASR. ONNX plus dynamic INT8 quantization is how the model gets there.

Parity is checked rather than assumed: quantization is lossy, and a detector
whose probabilities shifted after export would silently move the threshold that
decides when the agent speaks.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .models.audio_cnn import AudioEOTConfig, AudioEOTModel
from .models.fusion import FusionConfig, FusionEOTModel, load_fusion
from .models.text_transformer import TextEOTConfig, TextEOTModel

BATCH = 4
SEQ_LEN = 24


def load_any(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, str]:
    """Rebuild whichever branch a checkpoint holds."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    task = ckpt.get("task")
    if task is None:  # Phase 1 checkpoints predate the task field
        task = "fusion" if "text_config" in ckpt else (
            "audio" if "n_mels" in ckpt["config"] else "text")
    if task == "fusion":
        return load_fusion(ckpt_path, device), task
    model = (AudioEOTModel(AudioEOTConfig(**ckpt["config"])) if task == "audio"
             else TextEOTModel(TextEOTConfig(**ckpt["config"])))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), task


def example_inputs(model: torch.nn.Module, task: str) -> tuple[torch.Tensor, ...]:
    if task == "text":
        vocab, max_len = model.cfg.vocab_size, model.cfg.max_seq_len
        seq = min(SEQ_LEN, max_len)
        return (torch.randint(0, vocab, (BATCH, seq)),
                torch.full((BATCH,), seq, dtype=torch.long))
    if task == "audio":
        return (torch.randn(BATCH, model.cfg.n_mels, model.cfg.n_frames),)
    text_cfg, audio_cfg = model.text.cfg, model.audio.cfg
    seq = min(SEQ_LEN, text_cfg.max_seq_len)
    return (torch.randint(0, text_cfg.vocab_size, (BATCH, seq)),
            torch.full((BATCH,), seq, dtype=torch.long),
            torch.randn(BATCH, audio_cfg.n_mels, audio_cfg.n_frames))


def io_spec(task: str) -> tuple[list[str], dict]:
    if task == "text":
        names = ["input_ids", "lengths"]
        axes = {"input_ids": {0: "batch", 1: "sequence"}, "lengths": {0: "batch"}}
    elif task == "audio":
        names = ["mel"]
        axes = {"mel": {0: "batch"}}
    else:
        names = ["input_ids", "lengths", "mel"]
        axes = {"input_ids": {0: "batch", 1: "sequence"},
                "lengths": {0: "batch"}, "mel": {0: "batch"}}
    axes["logits"] = {0: "batch"}
    return names, axes


def export_onnx(model: torch.nn.Module, task: str, out_path: Path, opset: int = 17) -> Path:
    """Export with the TorchScript tracer (dynamo=False).

    The newer dynamo exporter produces a graph that onnxruntime's dynamic
    quantizer cannot shape-infer -- it fails with "Inferred shape and existing
    shape differ in dimension 0: (256) vs (1)". Since INT8 is the whole point of
    exporting here, the tracer is the path that actually reaches the target.
    """
    model.eval()
    inputs = example_inputs(model, task)
    names, axes = io_spec(task)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, inputs, str(out_path),
        input_names=names, output_names=["logits"],
        dynamic_axes=axes, opset_version=opset, do_constant_folding=True,
        dynamo=False,
    )
    return out_path


def quantize_int8(onnx_path: Path, out_path: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(onnx_path), str(out_path), weight_type=QuantType.QInt8)
    return out_path


def onnx_latency_ms(onnx_path: Path, inputs: tuple[torch.Tensor, ...], names: list[str],
                    runs: int = 100, warmup: int = 10) -> float:
    """Median single-example latency on one CPU thread.

    Measured rather than assumed: dynamic INT8 quantization rewrites MatMul, so
    it speeds up the transformer but *slows down* the conv-heavy audio branch,
    where it only adds quantize/dequantize around kernels it cannot replace.
    Shipping INT8 everywhere by default would make the acoustic model slower.
    """
    import time

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    feed = {name: tensor[:1].numpy() for name, tensor in zip(names, inputs, strict=True)}

    for _ in range(warmup):
        session.run(["logits"], feed)
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(["logits"], feed)
        timings.append((time.perf_counter() - start) * 1000.0)
    timings.sort()
    return timings[len(timings) // 2]


def onnx_probs(onnx_path: Path, inputs: tuple[torch.Tensor, ...], names: list[str]) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {name: tensor.numpy() for name, tensor in zip(names, inputs, strict=True)}
    logits = session.run(["logits"], feed)[0]
    return 1.0 / (1.0 + np.exp(-logits))


def check_parity(model: torch.nn.Module, task: str, onnx_path: Path,
                 inputs: tuple[torch.Tensor, ...] | None = None) -> float:
    """Max absolute probability difference between PyTorch and ONNX."""
    inputs = inputs or example_inputs(model, task)
    names, _ = io_spec(task)
    with torch.no_grad():
        reference = torch.sigmoid(model(*inputs)).numpy()
    return float(np.abs(reference - onnx_probs(onnx_path, inputs, names)).max())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("checkpoints/onnx"))
    ap.add_argument("--name", default=None, help="basename for the exported files")
    ap.add_argument("--fp32-tolerance", type=float, default=1e-4)
    ap.add_argument("--int8-tolerance", type=float, default=5e-2,
                    help="quantization is lossy; this bounds how lossy is acceptable")
    ap.add_argument("--latency-runs", type=int, default=100)
    args = ap.parse_args(argv)

    device = torch.device("cpu")
    model, task = load_any(args.ckpt, device)
    name = args.name or f"{task}_eot"
    inputs = example_inputs(model, task)

    fp32_path = export_onnx(model, task, args.out_dir / f"{name}.onnx")
    fp32_delta = check_parity(model, task, fp32_path, inputs)

    int8_path = quantize_int8(fp32_path, args.out_dir / f"{name}.int8.onnx")
    int8_delta = check_parity(model, task, int8_path, inputs)

    names, _ = io_spec(task)
    fp32_ms = onnx_latency_ms(fp32_path, inputs, names, runs=args.latency_runs)
    int8_ms = onnx_latency_ms(int8_path, inputs, names, runs=args.latency_runs)
    recommended = "int8" if int8_ms < fp32_ms else "fp32"

    report = {
        "task": task,
        "fp32": {"path": str(fp32_path), "mb": round(fp32_path.stat().st_size / 1e6, 2),
                 "max_prob_delta": fp32_delta, "latency_ms": round(fp32_ms, 2)},
        "int8": {"path": str(int8_path), "mb": round(int8_path.stat().st_size / 1e6, 2),
                 "max_prob_delta": int8_delta, "latency_ms": round(int8_ms, 2)},
        "recommended": recommended,
    }
    (args.out_dir / f"{name}.export.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"task={task}")
    print(f"  fp32 {report['fp32']['mb']:6.2f} MB  {fp32_ms:6.2f} ms  "
          f"max prob delta {fp32_delta:.2e}")
    print(f"  int8 {report['int8']['mb']:6.2f} MB  {int8_ms:6.2f} ms  "
          f"max prob delta {int8_delta:.2e}")
    print(f"  ship: {recommended} "
          f"({max(fp32_ms, int8_ms) / max(min(fp32_ms, int8_ms), 1e-9):.1f}x faster)")
    if fp32_delta > args.fp32_tolerance:
        raise SystemExit(f"fp32 parity failed: {fp32_delta:.2e} > {args.fp32_tolerance:.0e}")
    if int8_delta > args.int8_tolerance:
        raise SystemExit(f"int8 parity failed: {int8_delta:.2e} > {args.int8_tolerance:.0e}")
    print("parity OK")


if __name__ == "__main__":
    main()
