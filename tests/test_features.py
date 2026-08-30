"""The filterbank tests are the point of this file.

An empty low-frequency filter is invisible at training time — the loss still
falls, the model just never sees the bottom of the spectrum. These assertions are
what make that failure loud.
"""

import math

import pytest
import torch

from turnwave.data.features import (
    LogMel,
    MelConfig,
    fit_window,
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
)

CFG = MelConfig()


def test_mel_hz_roundtrip():
    for hz in (0.0, 100.0, 700.0, 1000.0, 8000.0):
        assert mel_to_hz(hz_to_mel(hz)).item() == pytest.approx(hz, abs=1e-6)


def test_mel_scale_reference_values():
    # 700 Hz is the mel break frequency: hz_to_mel(700) = 2595 * log10(2)
    assert hz_to_mel(700.0).item() == pytest.approx(2595.0 * math.log10(2.0), abs=1e-9)
    assert hz_to_mel(0.0).item() == pytest.approx(0.0, abs=1e-12)


def test_every_filter_is_nonempty():
    """The regression this module exists to prevent.

    Binning mel edges onto FFT indices with floor() leaves the lowest filters
    empty (62/64 non-zero at these settings). Building the triangles on a
    continuous frequency axis fixes it.
    """
    fb = mel_filterbank(CFG)
    assert fb.shape == (CFG.n_mels, CFG.n_fft // 2 + 1)
    per_filter = fb.sum(dim=1)
    empty = (per_filter <= 0).nonzero().flatten().tolist()
    assert not empty, f"empty mel filters at indices {empty}"


def test_filters_are_nonnegative_and_bounded():
    fb = mel_filterbank(CFG)
    assert (fb >= 0).all()
    # Peaks are <= 1: a triangle's apex sits at a mel edge, which generally falls
    # between two FFT bins, so the sampled maximum approaches 1 without reaching it.
    assert fb.max().item() <= 1.0
    assert fb.max().item() > 0.9


def test_adjacent_filters_overlap():
    """Triangles must share support, or there are frequencies no filter covers."""
    fb = mel_filterbank(CFG)
    for m in range(CFG.n_mels - 1):
        shared = ((fb[m] > 0) & (fb[m + 1] > 0)).sum().item()
        assert shared > 0, f"filters {m} and {m + 1} do not overlap"


def test_filters_are_ordered_by_frequency():
    """Filter m's centre of mass must sit below filter m+1's."""
    fb = mel_filterbank(CFG)
    freqs = torch.linspace(0.0, CFG.sample_rate / 2, CFG.n_fft // 2 + 1)
    centroids = (fb * freqs).sum(1) / fb.sum(1)
    assert torch.all(centroids[1:] > centroids[:-1])


def test_logmel_shape_matches_config():
    out = LogMel(CFG)(torch.randn(CFG.n_samples))
    assert out.shape == (CFG.n_mels, CFG.n_frames)


def test_logmel_batched():
    out = LogMel(CFG)(torch.randn(3, CFG.n_samples))
    assert out.shape == (3, CFG.n_mels, CFG.n_frames)


def test_logmel_is_deterministic():
    wav = torch.randn(CFG.n_samples)
    lm = LogMel(CFG)
    assert torch.equal(lm(wav), lm(wav))


def test_logmel_responds_to_frequency():
    """A 300 Hz tone must excite lower mel bins than a 3000 Hz tone."""
    t = torch.arange(CFG.n_samples) / CFG.sample_rate
    lm = LogMel(CFG)
    low = lm(torch.sin(2 * math.pi * 300 * t)).mean(dim=1)
    high = lm(torch.sin(2 * math.pi * 3000 * t)).mean(dim=1)
    assert low.argmax().item() < high.argmax().item()


def test_silence_hits_the_log_floor():
    out = LogMel(CFG)(torch.zeros(CFG.n_samples))
    assert out.max().item() == pytest.approx(math.log(CFG.log_floor), abs=1e-4)


def test_fit_window_truncates_from_the_end():
    """The cut point must stay at the window's right edge."""
    wav = torch.arange(100, dtype=torch.float32)
    assert torch.equal(fit_window(wav, 10), torch.arange(90, 100, dtype=torch.float32))


def test_fit_window_left_pads_short_clips():
    wav = torch.ones(4)
    out = fit_window(wav, 10)
    assert out.shape == (10,)
    assert torch.equal(out[:6], torch.zeros(6))
    assert torch.equal(out[6:], torch.ones(4))


def test_fit_window_exact_length_is_untouched():
    wav = torch.randn(10)
    assert torch.equal(fit_window(wav, 10), wav)
