"""Log-mel spectrogram features, computed from scratch.

`torch.stft` plus a hand-built mel filterbank — no torchaudio, no librosa. The
filterbank is the one piece worth writing carefully: the obvious implementation
(floor() mel points onto FFT bin indices) silently produces empty filters at the
low end, because several low-frequency mel points land on the same bin. Empty
filters mean the model never sees the bottom of the spectrum, which is exactly
where pitch lives — and pitch is the signal this whole branch exists to capture.
So the triangles are built on a continuous frequency axis instead, the way
librosa does it, and `tests/test_features.py` asserts every filter is non-zero.
"""

from dataclasses import dataclass

import torch

_MEL_BREAK_HZ = 700.0
_MEL_BREAK_MEL = 2595.0


@dataclass(frozen=True)
class MelConfig:
    sample_rate: int = 16000
    # 32 ms window: a 25 ms (n_fft=400) window gives 40 Hz bins, but the narrowest
    # mel filters here are ~56 Hz wide, so adjacent low-frequency triangles can end
    # up sharing no FFT bin — gaps in coverage. 512 makes the bank clean at 64 mels
    # (see tests), and 32 ms is still well inside the 100 ms+ scale of prosodic events.
    n_fft: int = 512
    hop_length: int = 160  # 10 ms
    n_mels: int = 64
    # 60 Hz floor: below any human fundamental, so nothing but rumble is discarded.
    f_min: float = 60.0
    f_max: float = 7600.0
    window_seconds: float = 2.0
    log_floor: float = 1e-6

    @property
    def n_frames(self) -> int:
        """Frames produced by a window_seconds clip with center padding."""
        return int(self.window_seconds * self.sample_rate) // self.hop_length + 1

    @property
    def n_samples(self) -> int:
        return int(self.window_seconds * self.sample_rate)


def hz_to_mel(hz: torch.Tensor | float) -> torch.Tensor:
    hz = torch.as_tensor(hz, dtype=torch.float64)
    return _MEL_BREAK_MEL * torch.log10(1.0 + hz / _MEL_BREAK_HZ)


def mel_to_hz(mel: torch.Tensor | float) -> torch.Tensor:
    mel = torch.as_tensor(mel, dtype=torch.float64)
    return _MEL_BREAK_HZ * (10.0 ** (mel / _MEL_BREAK_MEL) - 1.0)


def mel_filterbank(cfg: MelConfig) -> torch.Tensor:
    """[n_mels, n_fft // 2 + 1] triangular filters on a continuous frequency axis.

    Each filter m rises linearly from edge[m] to edge[m+1] and falls to edge[m+2],
    evaluated at the true FFT bin frequencies rather than at rounded bin indices.
    """
    n_freqs = cfg.n_fft // 2 + 1
    fft_hz = torch.linspace(0.0, cfg.sample_rate / 2.0, n_freqs, dtype=torch.float64)
    edges_hz = mel_to_hz(torch.linspace(
        float(hz_to_mel(cfg.f_min)), float(hz_to_mel(cfg.f_max)), cfg.n_mels + 2,
        dtype=torch.float64,
    ))

    # ramps[m, k] = fft_hz[k] - edges_hz[m]
    ramps = fft_hz.unsqueeze(0) - edges_hz.unsqueeze(1)
    widths = edges_hz[1:] - edges_hz[:-1]
    rising = ramps[:-2] / widths[:-1].unsqueeze(1)
    falling = -ramps[2:] / widths[1:].unsqueeze(1)
    fb = torch.clamp(torch.minimum(rising, falling), min=0.0)
    return fb.to(torch.float32)


class LogMel:
    """Callable feature extractor. Holds the window and filterbank so a batch of
    clips reuses them instead of rebuilding per call."""

    def __init__(self, cfg: MelConfig = MelConfig()):
        self.cfg = cfg
        self.window = torch.hann_window(cfg.n_fft)
        self.filterbank = mel_filterbank(cfg)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """[..., n_samples] float waveform -> [..., n_mels, n_frames] log-mel."""
        spec = torch.stft(
            waveform.to(torch.float32),
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = spec.real.pow(2) + spec.imag.pow(2)
        mel = self.filterbank @ power
        return torch.log(mel + self.cfg.log_floor)


def fit_window(waveform: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Take the last n_samples, left-padding with silence if the clip is shorter.

    Left padding (rather than right) keeps the cut point at the end of the window:
    the model always looks at the moment the speaker stopped, never past it.
    """
    if waveform.numel() >= n_samples:
        return waveform[-n_samples:]
    pad = torch.zeros(n_samples - waveform.numel(), dtype=waveform.dtype)
    return torch.cat([pad, waveform])
