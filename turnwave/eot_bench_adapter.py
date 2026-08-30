"""Adapter that plugs TurnWave into LiveKit's official eot-bench harness.

Deliberately an adapter rather than our own metric script. eot-bench's numbers
are only comparable if the protocol matches exactly, and its definitions are
subtle: a "false cutoff" is counted per mid-turn pause (not per row), "@300ms"
is a *budget on mean latency* rather than a wait threshold, and latency is
measured from the start of the final silence and includes the policy's own
action delay. Reimplementing that from the leaderboard's column headings would
produce numbers that look comparable and are not. Running their harness removes
the question.

    pip install git+https://github.com/livekit/eot-bench
    export TURNWAVE_ONNX=checkpoints/onnx/fusion_eot.onnx
    export TURNWAVE_TOKENIZER=checkpoints/tokenizer/spm.model
    eot-harness predict --path livekit/eot-bench-data --name all --split validation \
        --adapter turnwave.eot_bench_adapter:TurnWaveAdapter --output-dir output
    eot-harness compute-metrics --predictions output/.../predictions.parquet \
        --output-dir output/.../metrics

The harness instantiates the adapter with no arguments, so configuration comes
from the environment.
"""

import os

import numpy as np

from .data.features import MelConfig
from .infer import TurnDetector

# The harness scores one point per span at exactly this silence duration when an
# adapter declares it. LiveKit Turn Detector v1 and SmartTurn both use 0.2, and
# matching it is what makes our row comparable to theirs; omitting it would put
# us in the first-threshold-crossing regime instead, which is a different
# operating point wearing the same column heading.
SCORE_POINT = 0.2


class TurnWaveAdapter:
    adapter_id = "turnwave"
    display_name = "TurnWave (from scratch)"
    score_point = SCORE_POINT

    def __init__(self, onnx_path: str | None = None, tokenizer: str | None = None):
        onnx_path = onnx_path or os.environ.get("TURNWAVE_ONNX")
        if not onnx_path:
            raise ValueError("set TURNWAVE_ONNX to an exported .onnx model")
        tokenizer = tokenizer or os.environ.get("TURNWAVE_TOKENIZER")
        self.detector = TurnDetector(onnx_path, tokenizer=tokenizer)
        # Tell the harness how much causal audio we need; it trims for us.
        self.max_audio_sec = (self.detector.mel.cfg.window_seconds
                              if self.detector.needs_audio else 0.0)

    def supports_language(self, language_code: str) -> bool:
        """English only: the model is trained on English and the headline
        leaderboard is English. Claiming the other 13 would report noise."""
        return language_code == "en"

    @staticmethod
    def _current_utterance(messages: list[dict]) -> str:
        """The words heard so far in the turn being judged.

        The harness puts them in the trailing user message, already filtered to
        words that finished before the decision point.
        """
        for message in reversed(messages or []):
            if message.get("role") == "user":
                return message.get("content") or ""
        return ""

    def predict_batch(self, batch: list[dict]) -> list[float]:
        """One P(end of turn) per decision point, in input order."""
        probabilities = []
        for item in batch:
            audio = None
            if self.detector.needs_audio:
                array = np.asarray(item["audio"]["array"], dtype=np.float32)
                sample_rate = int(item["audio"]["sampling_rate"])
                expected = self.detector.mel.cfg.sample_rate
                if sample_rate != expected:
                    raise ValueError(
                        f"eot-bench audio is {sample_rate} Hz but the model was "
                        f"trained at {expected} Hz")
                audio = array
            text = self._current_utterance(item.get("messages", [])) if self.detector.needs_text else None
            probabilities.append(self.detector.predict(audio=audio, text=text))
        return probabilities


class TurnWaveAudioOnlyAdapter(TurnWaveAdapter):
    """Same model family, audio branch only — for the ablation on real data."""

    adapter_id = "turnwave-audio"
    display_name = "TurnWave audio branch"


class TurnWaveTextOnlyAdapter(TurnWaveAdapter):
    """Text branch only. Needs no audio, so max_audio_sec stays 0."""

    adapter_id = "turnwave-text"
    display_name = "TurnWave text branch"
