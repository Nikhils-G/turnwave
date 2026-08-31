"""Reader for `pipecat-ai/smart-turn-data-v3.x` — conversational end-of-turn clips.

Phase 4's model scored AP 0.945 on its own test set and AUC 0.563 on eot-bench,
because the corpus it trained on turned out to be read speech: pauses that are
reading hesitations and sentence boundaries, not conversational turn-yields. This
corpus is the correction. Its clips come from real voice-agent contexts and carry
a human-authored `endpoint_bool` — did the speaker's turn actually end here.

Two differences from the semantic-vad reader that matter:

* **One row is one example.** The clips are already cut at the decision point, so
  there are no silence spans to enumerate and no cut offset to apply. The window
  is simply the tail of the clip, which is what `features.fit_window` already does.
* **There are no transcripts.** `spoken_text` is null on every row, so this corpus
  trains the acoustic branch only. Fusion needs text and stays on Phase 4 weights.

`synthetic` is carried through to the metadata rather than dropped: 82% of the
corpus is TTS, and after Phase 4 the one thing this project should never do again
is stop asking what its training audio actually is.
"""

from collections.abc import Iterator
from dataclasses import dataclass

DATASET = "pipecat-ai/smart-turn-data-v3.2-train"
TEST_DATASET = "pipecat-ai/smart-turn-data-v3.2-test"


@dataclass(frozen=True)
class Clip:
    """One labelled decision point. `text` is always empty — see the module docstring."""

    label: int
    language: str
    synthetic: bool
    source: str
    clip_id: str
    text: str = ""


def _as_bool(value) -> bool | None:
    """The parquet columns arrive as bool, as 'True'/'False', or as None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1"):
            return True
        if lowered in ("false", "0"):
            return False
    return None


def iter_clips(row: dict, languages: set[str] | None = None,
               real_only: bool = False) -> Iterator[Clip]:
    """Yield at most one Clip per row; nothing if the row is unusable.

    An iterator rather than a scalar so the caller's loop is identical to the
    silence-span reader's, which yields several cuts per row.
    """
    label = _as_bool(row.get("endpoint_bool"))
    if label is None:
        return  # unlabelled rows are silently dropped upstream; be explicit here

    language = (row.get("language") or "").lower()
    if languages is not None and language not in languages:
        return

    synthetic = bool(_as_bool(row.get("synthetic")))
    if real_only and synthetic:
        return

    yield Clip(
        label=int(label),
        language=language,
        synthetic=synthetic,
        source=row.get("dataset") or "",
        clip_id=row.get("id") or "",
    )
