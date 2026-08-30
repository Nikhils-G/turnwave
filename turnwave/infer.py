"""Streaming CPU inference — the shape a voice agent actually calls.

A turn detector runs on every pause in every call, on the same CPU already busy
with VAD, ASR and audio I/O. If it costs more than a few tens of milliseconds it
has spent the latency it was supposed to save. So inference here is ONNX on CPU,
INT8 by default, with the log-mel front end computed in-process.

    detector = TurnDetector("checkpoints/onnx/fusion_eot.int8.onnx",
                            tokenizer="checkpoints/tokenizer/spm.model")
    p = detector.predict(audio_tail_16k, "i want a large pepperoni and")
    if p > 0.5: respond()
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from .data.features import LogMel, MelConfig
from .data.text_pairs import normalize_asr

DEFAULT_THRESHOLD = 0.5


class TurnDetector:
    """Wraps an exported model plus its feature front end.

    The task is inferred from the ONNX graph's inputs, so the same class serves
    the text, audio, and fused models without the caller choosing.
    """

    def __init__(self, onnx_path: str | Path, tokenizer: str | Path | None = None,
                 mel_config: MelConfig | None = None, max_len: int = 128,
                 threshold: float = DEFAULT_THRESHOLD):
        import onnxruntime as ort

        options = ort.SessionOptions()
        # One thread: a voice pipeline runs many calls per box, and letting each
        # detector fan out across cores costs more in contention than it saves.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx_path), options,
                                            providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.needs_text = "input_ids" in self.input_names
        self.needs_audio = "mel" in self.input_names
        self.threshold = threshold
        self.max_len = max_len

        self.mel = LogMel(self._resolve_mel_config(mel_config)) if self.needs_audio else None
        self.tokenizer = None
        if self.needs_text:
            if tokenizer is None:
                raise ValueError("this model consumes text; pass tokenizer=<spm.model>")
            from .tokenizer import Tokenizer

            self.tokenizer = Tokenizer(tokenizer)

    def _expected_mel_shape(self) -> tuple[int, int]:
        spec = next(i for i in self.session.get_inputs() if i.name == "mel")
        _, n_mels, n_frames = spec.shape
        return int(n_mels), int(n_frames)

    def _resolve_mel_config(self, mel_config: MelConfig | None) -> MelConfig:
        """Match the front end to the graph, rather than assuming the defaults.

        The exported model fixes the mel shape it accepts. Building the extractor
        from defaults regardless means a model trained with a different window or
        mel count gets features it cannot consume — caught here as a clear error
        instead of an onnxruntime dimension complaint from three frames deeper.
        """
        import dataclasses

        n_mels, n_frames = self._expected_mel_shape()
        if mel_config is not None:
            if (mel_config.n_mels, mel_config.n_frames) != (n_mels, n_frames):
                raise ValueError(
                    f"mel_config produces {mel_config.n_mels}x{mel_config.n_frames} "
                    f"but the model expects {n_mels}x{n_frames}")
            return mel_config
        base = MelConfig()
        # n_frames = window_samples // hop + 1, so the window follows from the graph.
        window_seconds = (n_frames - 1) * base.hop_length / base.sample_rate
        return dataclasses.replace(base, n_mels=n_mels, window_seconds=window_seconds)

    def _text_inputs(self, text: str) -> dict:
        ids = self.tokenizer.encode_example("", normalize_asr(text), self.max_len)
        ids = ids or [self.tokenizer.sep_id]
        return {"input_ids": np.array([ids], dtype=np.int64),
                "lengths": np.array([len(ids)], dtype=np.int64)}

    def _audio_inputs(self, audio: np.ndarray | torch.Tensor) -> dict:
        from .data.features import fit_window

        wav = torch.as_tensor(np.asarray(audio), dtype=torch.float32)
        if wav.ndim > 1:
            wav = wav.mean(dim=0)
        mel = self.mel(fit_window(wav, self.mel.cfg.n_samples))
        return {"mel": mel.unsqueeze(0).numpy().astype(np.float32)}

    def predict(self, audio=None, text: str | None = None) -> float:
        """P(the speaker has finished). Feed whichever inputs the model needs."""
        feed = {}
        if self.needs_text:
            if text is None:
                raise ValueError("this model needs text")
            feed.update(self._text_inputs(text))
        if self.needs_audio:
            if audio is None:
                raise ValueError("this model needs audio")
            feed.update(self._audio_inputs(audio))
        logit = self.session.run(["logits"], feed)[0]
        return float(1.0 / (1.0 + np.exp(-logit.reshape(-1)[0])))

    def is_complete(self, audio=None, text: str | None = None) -> bool:
        return self.predict(audio, text) >= self.threshold


def benchmark(detector: TurnDetector, runs: int = 200, warmup: int = 20) -> dict:
    """Per-call latency, which is the number that decides deployability."""
    rng = np.random.default_rng(0)
    audio = rng.normal(size=detector.mel.cfg.n_samples).astype(np.float32) if detector.needs_audio else None
    text = "i would like to order a large pepperoni pizza please" if detector.needs_text else None

    for _ in range(warmup):
        detector.predict(audio, text)

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        detector.predict(audio, text)
        timings.append((time.perf_counter() - start) * 1000.0)
    timings.sort()
    return {
        "runs": runs,
        "mean_ms": round(statistics.fmean(timings), 2),
        "p50_ms": round(timings[len(timings) // 2], 2),
        "p95_ms": round(timings[int(len(timings) * 0.95)], 2),
        "max_ms": round(timings[-1], 2),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--text", default=None, help="predict for one utterance")
    ap.add_argument("--audio", type=Path, default=None, help="wav file for the audio tail")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--budget-ms", type=float, default=50.0)
    args = ap.parse_args(argv)

    detector = TurnDetector(args.onnx, tokenizer=args.tokenizer)

    if args.text or args.audio:
        audio = None
        if args.audio:
            import soundfile as sf

            audio, _ = sf.read(args.audio, dtype="float32")
        p = detector.predict(audio, args.text)
        verdict = "complete — respond" if p >= detector.threshold else "incomplete — keep listening"
        print(f"P(turn complete) = {p:.3f}  ->  {verdict}")

    if args.benchmark:
        stats = benchmark(detector, runs=args.runs)
        print(f"latency over {stats['runs']} runs (1 thread, CPU): "
              f"mean {stats['mean_ms']} ms | p50 {stats['p50_ms']} ms | "
              f"p95 {stats['p95_ms']} ms | max {stats['max_ms']} ms")
        verdict = "PASS" if stats["p95_ms"] < args.budget_ms else "OVER BUDGET"
        print(f"p95 vs {args.budget_ms:.0f} ms budget: {verdict}")


if __name__ == "__main__":
    main()
