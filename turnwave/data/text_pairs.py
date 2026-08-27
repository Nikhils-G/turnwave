"""Build (context, text, label) end-of-turn examples from dialogue corpora.

Positives are complete utterances; negatives are the same utterances truncated
at a random word boundary, simulating a mid-thought pause. Some truncations are
accidentally complete phrases ("yes i agree completely" -> "yes i agree") — that
label noise is inherent to text-only EOT and is exactly the ambiguity the
acoustic branch exists to resolve.

All text is normalized to ASR-like form (lowercase, no punctuation) because at
inference time the model sees streaming ASR output, which carries neither casing
nor reliable punctuation. Skipping this step would let the model cheat on
terminal periods during training and then fail in production.
"""

import random
import re
from collections.abc import Iterable, Iterator

_CONTRACTION = re.compile(r"\s*['’]\s*")
_NON_WORD = re.compile(r"[^a-z0-9' ]+")
_SPACES = re.compile(r"\s+")


def normalize_asr(text: str) -> str:
    t = text.lower()
    t = _CONTRACTION.sub("'", t)  # "i ’ m" -> "i'm" (DailyDialog spaces out contractions)
    t = _NON_WORD.sub(" ", t)
    return _SPACES.sub(" ", t).strip()


def iter_examples(
    dialogues: Iterable[list[str]],
    seed: int = 0,
    negatives_per_positive: int = 1,
    min_words_to_cut: int = 3,
) -> Iterator[dict]:
    """Deterministic for a given seed and input order."""
    rng = random.Random(seed)
    for dialogue in dialogues:
        prev = ""
        for raw in dialogue:
            text = normalize_asr(raw)
            if not text:
                continue
            yield {"context": prev, "text": text, "label": 1}
            words = text.split()
            if len(words) >= min_words_to_cut:
                n_cuts = min(negatives_per_positive, len(words) - 1)
                for cut in rng.sample(range(1, len(words)), k=n_cuts):
                    yield {"context": prev, "text": " ".join(words[:cut]), "label": 0}
            prev = text
