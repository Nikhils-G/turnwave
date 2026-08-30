"""Turn a silence-annotated utterance into end-of-turn training examples.

Source rows (`Scicom-intl/semantic-vad-eot`, `livekit/eot-bench-data`) carry:

    duration, silence_spans: [{start, end}], words: [{word, start, end}]

Every pause is a moment a voice agent must judge: has this person finished, or
are they thinking? The final pause is the real end of the turn; every earlier one
is a mid-turn hesitation the agent should listen through. So one utterance yields
one positive and N-1 negatives, all cut at real pauses in real speech — no
synthetic truncation, and no guessing which words had been spoken, because the
word timings say so exactly.

The cut is placed at `span.start`, the instant speech stopped. Nothing after it is
visible to the model, in audio or in text, which is precisely the information a
live agent has when it must decide whether to respond.
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


def iter_cuts(
    row: dict,
    min_words: int = 1,
    max_silence_seconds: float = 5.0,
    min_final_silence_seconds: float = 0.2,
) -> Iterator[Cut]:
    """Yield one Cut per usable silence span, in time order.

    Filters mirror the eot-bench construction rules: drop rows whose final pause
    is too short to be a real turn end, and drop rows containing an implausibly
    long pause (usually a recording artefact rather than a hesitation).
    """
    spans = sorted(
        (s for s in row.get("silence_spans") or [] if s.get("start") is not None),
        key=lambda s: s["start"],
    )
    if not spans:
        return

    if any((s.get("end", s["start"]) - s["start"]) > max_silence_seconds for s in spans):
        return

    final = spans[-1]
    if (final.get("end", final["start"]) - final["start"]) < min_final_silence_seconds:
        return

    words = row.get("words") or []
    for i, span in enumerate(spans):
        cut = float(span["start"])
        text = words_before(words, cut)
        if len(text.split()) < min_words:
            continue
        is_final = i == len(spans) - 1
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
