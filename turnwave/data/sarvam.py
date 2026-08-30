"""Throttled, disk-cached Sarvam TTS client.

Its role in this project shrank during planning and that is worth stating: the
corpus already provides real human prosody, and the Indic clips that were
Sarvam's main Phase 5 justification already exist in public turn-detection data.
Synthesising thousands of TTS clips would add TTS prosody to a dataset that has
real prosody — spending money to make the training distribution less like
production. So this ships as working, tested platform code with a pilot mode,
and bulk synthesis waits until a measured gap justifies it.

Two things it must get right when it is used:

* **Throttling.** The Starter tier allows 30 requests/minute, which is the real
  constraint — not cost. Fanning out with unbounded async earns 429s, not speed.
* **Caching.** Every clip is keyed by a hash of its request, so re-running a
  build costs nothing. Without this, one interrupted run is one wasted bill.

`bulbul:v2` rather than v3 deliberately: v3 dropped `pitch` and `loudness`, which
are exactly the knobs worth varying to produce prosodic diversity.
"""

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

ENDPOINT = "https://api.sarvam.ai/text-to-speech"
DEFAULT_MODEL = "bulbul:v2"
RETRY_STATUS = {429, 500, 502, 503, 504}


@dataclass
class TTSRequest:
    text: str
    language_code: str = "en-IN"
    speaker: str = "anushka"
    model: str = DEFAULT_MODEL
    pace: float = 1.0
    pitch: float = 0.0
    loudness: float = 1.0
    sample_rate: int = 16000  # match the corpus; the documented default differs

    def payload(self) -> dict:
        return {
            "text": self.text,
            "target_language_code": self.language_code,
            "speaker": self.speaker,
            "model": self.model,
            "pace": self.pace,
            "pitch": self.pitch,
            "loudness": self.loudness,
            "speech_sample_rate": self.sample_rate,
        }

    def cache_key(self) -> str:
        blob = json.dumps(self.payload(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:24]


@dataclass
class RateLimiter:
    """Simple sliding window. 30 req/min is the documented Starter limit."""

    max_per_minute: int = 30
    _timestamps: list[float] = field(default_factory=list)

    def acquire(self, sleep=time.sleep, now=time.monotonic) -> float:
        current = now()
        self._timestamps = [t for t in self._timestamps if current - t < 60.0]
        waited = 0.0
        if len(self._timestamps) >= self.max_per_minute:
            waited = 60.0 - (current - self._timestamps[0]) + 0.01
            if waited > 0:
                sleep(waited)
                current = now()
                self._timestamps = [t for t in self._timestamps if current - t < 60.0]
        self._timestamps.append(current)
        return waited


def decode_audio(response_json: dict) -> bytes:
    """Sarvam returns `{"audios": ["<base64>", ...]}` — a list, not one blob.

    Each chunk is decoded separately and the *bytes* concatenated. Joining the
    base64 strings first looks equivalent and is not: a padded chunk ends in "="
    and decoding stops there, silently yielding only the first chunk's audio.
    """
    audios = response_json.get("audios")
    if not audios:
        raise ValueError(f"no audio in response: {list(response_json)}")
    return b"".join(base64.b64decode(chunk) for chunk in audios)


class SarvamTTS:
    def __init__(self, api_key: str | None = None, cache_dir: str | Path = "data/tts_cache",
                 max_per_minute: int = 30, max_retries: int = 4, session=None):
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError("set SARVAM_API_KEY (see .env.example)")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(max_per_minute)
        self.max_retries = max_retries
        self._session = session
        self.stats = {"cached": 0, "fetched": 0, "retries": 0, "characters": 0}

    @property
    def session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def synthesize(self, request: TTSRequest, sleep=time.sleep) -> Path:
        """Returns the path to the wav, fetching only on a cache miss."""
        path = self.cache_dir / f"{request.cache_key()}.wav"
        if path.exists():
            self.stats["cached"] += 1
            return path

        headers = {"api-subscription-key": self.api_key, "Content-Type": "application/json"}
        for attempt in range(self.max_retries):
            self.limiter.acquire(sleep=sleep)
            response = self.session.post(ENDPOINT, headers=headers,
                                         json=request.payload(), timeout=60)
            if response.status_code == 200:
                path.write_bytes(decode_audio(response.json()))
                self.stats["fetched"] += 1
                self.stats["characters"] += len(request.text)
                return path
            if response.status_code in RETRY_STATUS and attempt < self.max_retries - 1:
                self.stats["retries"] += 1
                sleep(2.0 ** attempt)  # exponential backoff, as the docs prescribe
                continue
            raise RuntimeError(f"sarvam tts failed [{response.status_code}]: "
                               f"{response.text[:200]}")
        raise RuntimeError(f"sarvam tts gave up after {self.max_retries} attempts")

    def estimated_cost_inr(self) -> float:
        """Sarvam bills TTS at Rs 30 per 10,000 characters."""
        return self.stats["characters"] * 30.0 / 10_000.0
