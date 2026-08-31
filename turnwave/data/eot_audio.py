"""Turn a silence-annotated utterance into end-of-turn training examples.

Source rows (`Scicom-intl/semantic-vad-eot`, `livekit/eot-bench-data`) carry:

    duration, silence_spans: [{start, end}], words: [{word, start, end}]

Every pause is a moment a voice agent must judge: has this person finished, or
are they thinking? The final pause is the real end of the turn; every earlier one
is a mid-turn hesitation the agent should listen through. So one utterance yields
one positive and N-1 negatives, all cut at real pauses in real speech — no
synthetic truncation, and no guessing which words had been spoken, because the
word timings say so exactly.

The cut is placed a short way *into* each pause rather than at the instant speech
stops. A live agent never decides at that instant — it waits for silence to
register first — and LiveKit's eot-bench scores models at exactly 0.2 s of
silence. Training at the same offset keeps the model's input distribution matched
to both, instead of asking it at evaluation time about a window shape it never saw.

Nothing after the cut is visible, in audio or in text, which is precisely the
information a live agent has when it must decide whether to respond.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from .text_pairs import normalize_asr


@dataclass(frozen=True)
class Cut:
    """One decision point: what was heard up to `cut_seconds`, and the answer."""

    cut_seconds: float
    text: str
    label: int
    is_final: bool


def words_before(words: list[dict], cut_seconds: float) -> str:
    """ASR-style transcript of everything finished before the cut.

    Uses `end` rather than `start` so a word still being spoken at the cut is not
    counted — the agent has not heard it yet.
    """
    spoken = [w["word"] for w in words if w.get("end") is not None and w["end"] <= cut_seconds]
    return normalize_asr(" ".join(spoken))


# eot-bench scores at 0.2 s of silence, and keeps only hold spans in [0.2, 5.0].
DEFAULT_CUT_OFFSET = 0.2


def iter_cuts(
    row: dict,
    min_words: int = 1,
    max_silence_seconds: float = 5.0,
    cut_offset_seconds: float = DEFAULT_CUT_OFFSET,
) -> Iterator[Cut]:
    """Yield one Cut per usable silence span, in time order.

    `cut_offset_seconds` places the decision point that far into the pause.

    Spans shorter than the offset are dropped, and that filter is load-bearing
    rather than tidiness: cutting 0.2 s into a 0.1 s pause would put the window
    past the start of the next word, leaking future speech into a *negative*
    example. That is the label corruption that makes a turn detector look
    excellent in training and fail on real calls, so it must be impossible by
    construction, not merely unlikely.

    Dropping short spans also matches eot-bench's own rule, which keeps hold
    spans in [0.2, 5.0] seconds.
    """
    spans = sorted(
        (s for s in row.get("silence_spans") or [] if s.get("start") is not None),
        key=lambda s: s["start"],
    )
    if not spans:
        return

    def duration(span: dict) -> float:
        return float(span.get("end", span["start"])) - float(span["start"])

    if any(duration(s) > max_silence_seconds for s in spans):
        return

    # The final span is the turn end; if it is too short to hold the decision
    # point there is no usable positive, and a row of negatives alone would teach
    # the model that this speaker never finishes.
    if duration(spans[-1]) < cut_offset_seconds:
        return

    words = row.get("words") or []
    last_index = len(spans) - 1
    for i, span in enumerate(spans):
        if duration(span) < cut_offset_seconds:
            continue  # cutting here would reach into the next word
        cut = float(span["start"]) + cut_offset_seconds
        text = words_before(words, cut)
        if len(text.split()) < min_words:
            continue
        is_final = i == last_index
        yield Cut(cut_seconds=cut, text=text, label=int(is_final), is_final=is_final)


def slice_tail(audio: "object", cut_seconds: float, sample_rate: int, n_samples: int):
    """Waveform window ending at the cut. Imports torch lazily so the pure
    example-generation logic above stays importable without it."""
    import torch

    from .features import fit_window

    wav = torch.as_tensor(audio, dtype=torch.float32)
    if wav.ndim > 1:
        wav = wav.mean(dim=0)  # downmix any stray multi-channel clip
    end = max(0, min(int(round(cut_seconds * sample_rate)), wav.numel()))
    return fit_window(wav[:end], n_samples)
