"""Sarvam client tests. No network: a fake session stands in for requests."""

import base64
import json

import pytest

from turnwave.data.sarvam import RateLimiter, SarvamTTS, TTSRequest, decode_audio


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.responses.pop(0)


def wav_response(data=b"RIFFfake"):
    return FakeResponse(200, {"audios": [base64.b64encode(data).decode()]})


def test_decode_audio_concatenates_chunks():
    """Joining the base64 strings first would stop at the first chunk's padding."""
    chunks = [base64.b64encode(b"ab").decode(), base64.b64encode(b"cd").decode()]
    assert decode_audio({"audios": chunks}) == b"abcd"


def test_decode_audio_rejects_an_empty_response():
    with pytest.raises(ValueError, match="no audio"):
        decode_audio({"request_id": "x"})


def test_request_uses_v2_for_the_prosody_controls():
    """v3 dropped pitch and loudness, which are the knobs worth varying."""
    payload = TTSRequest(text="hi").payload()
    assert payload["model"] == "bulbul:v2"
    assert "pitch" in payload and "loudness" in payload
    assert payload["speech_sample_rate"] == 16000  # matches the corpus


def test_cache_key_tracks_every_parameter():
    base = TTSRequest(text="hello")
    assert base.cache_key() == TTSRequest(text="hello").cache_key()
    for changed in (TTSRequest(text="hello", pitch=0.5),
                    TTSRequest(text="hello", speaker="vidya"),
                    TTSRequest(text="goodbye")):
        assert changed.cache_key() != base.cache_key()


def test_rate_limiter_waits_once_the_window_is_full():
    slept, clock = [], [0.0]
    limiter = RateLimiter(max_per_minute=2)
    for _ in range(2):
        assert limiter.acquire(sleep=slept.append, now=lambda: clock[0]) == 0.0
    assert limiter.acquire(sleep=slept.append, now=lambda: clock[0]) > 0
    assert slept and slept[0] == pytest.approx(60.0, abs=0.1)


def test_synthesize_writes_and_then_reuses_the_cache(tmp_path):
    session = FakeSession([wav_response(b"AUDIO")])
    tts = SarvamTTS(api_key="k", cache_dir=tmp_path, session=session)
    request = TTSRequest(text="hello there")

    path = tts.synthesize(request, sleep=lambda s: None)
    assert path.read_bytes() == b"AUDIO"
    assert tts.stats == {"cached": 0, "fetched": 1, "retries": 0, "characters": 11}

    # second call must not hit the network — the fake would raise IndexError
    assert tts.synthesize(request, sleep=lambda s: None) == path
    assert tts.stats["cached"] == 1
    assert len(session.calls) == 1


def test_synthesize_sends_the_documented_auth_header(tmp_path):
    session = FakeSession([wav_response()])
    SarvamTTS(api_key="secret", cache_dir=tmp_path, session=session).synthesize(
        TTSRequest(text="hi"), sleep=lambda s: None)
    assert session.calls[0]["headers"]["api-subscription-key"] == "secret"


def test_synthesize_retries_a_429_with_backoff(tmp_path):
    session = FakeSession([FakeResponse(429, text="rate_limit_exceeded_error"), wav_response()])
    tts = SarvamTTS(api_key="k", cache_dir=tmp_path, session=session)
    slept = []
    assert tts.synthesize(TTSRequest(text="hi"), sleep=slept.append).exists()
    assert tts.stats["retries"] == 1
    assert len(session.calls) == 2


def test_synthesize_raises_on_a_non_retryable_error(tmp_path):
    session = FakeSession([FakeResponse(401, text="unauthorized")])
    tts = SarvamTTS(api_key="k", cache_dir=tmp_path, session=session)
    with pytest.raises(RuntimeError, match=r"\[401\]"):
        tts.synthesize(TTSRequest(text="hi"), sleep=lambda s: None)


def test_missing_api_key_fails_immediately(monkeypatch, tmp_path):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SARVAM_API_KEY"):
        SarvamTTS(cache_dir=tmp_path)


def test_cost_estimate_matches_the_published_rate(tmp_path):
    """Rs 30 per 10,000 characters."""
    tts = SarvamTTS(api_key="k", cache_dir=tmp_path, session=FakeSession([]))
    tts.stats["characters"] = 10_000
    assert tts.estimated_cost_inr() == pytest.approx(30.0)
