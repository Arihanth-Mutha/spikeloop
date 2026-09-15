"""Real-time spike detection for multi-electrode arrays.

>>> import spikeloop as sl
>>> data, truth, stims = sl.synthesize(n_channels=64, duration_s=2.0)
>>> det = sl.Detector(n_channels=64, sample_rate=25_000)
>>> for s in stims:
...     det.stimulate(int(s))          # blank the stimulation artifact
>>> spikes = det.process(data) + det.flush()
>>> len(spikes) > 0
True

Feed it 1 ms chunks from a live stream and get the same spikes as one big call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from . import _core

__version__ = _core.__version__
__all__ = ["Detector", "Spikes", "bandpass", "detect", "score", "synthesize"]


@dataclass(frozen=True)
class Spikes:
    """Detected spikes as parallel arrays, aligned to each spike's peak."""

    frame: np.ndarray
    channel: np.ndarray
    amplitude: np.ndarray

    @classmethod
    def _from(cls, d: dict) -> "Spikes":
        return cls(d["frame"], d["channel"], d["amplitude"])

    @classmethod
    def empty(cls) -> "Spikes":
        return cls(np.empty(0, np.uint64), np.empty(0, np.uint32), np.empty(0, np.float32))

    def __len__(self) -> int:
        return len(self.frame)

    def __add__(self, other: "Spikes") -> "Spikes":
        return Spikes(
            np.concatenate([self.frame, other.frame]),
            np.concatenate([self.channel, other.channel]),
            np.concatenate([self.amplitude, other.amplitude]),
        )

    def sorted(self) -> "Spikes":
        order = np.lexsort((self.channel, self.frame))
        return Spikes(self.frame[order], self.channel[order], self.amplitude[order])

    def times(self, sample_rate: float) -> np.ndarray:
        """Spike times in seconds."""
        return self.frame / float(sample_rate)

    def counts(self, n_channels: int) -> np.ndarray:
        """Spikes per channel."""
        return np.bincount(self.channel, minlength=n_channels)


def _as_frames(data: np.ndarray, n_channels: int) -> np.ndarray:
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim == 1 and n_channels == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D (n_frames, n_channels), got shape {data.shape}")
    return data


class Detector:
    """Streaming detector: band-pass, robust per-channel thresholds, peak alignment,
    refractory period and stimulation-artifact blanking.

    Parameters mirror ``spikeloop_core::Config``. Times are in milliseconds, voltages in
    the data's units (microvolts by convention). ``threshold`` is in multiples of each
    channel's robust noise sigma, ``median(|x|) / 0.6745``.
    """

    def __init__(
        self,
        n_channels: int,
        sample_rate: float,
        *,
        band: Tuple[float, float] = (300.0, 3000.0),
        threshold: float = 4.5,
        polarity: str = "negative",
        peak_window_ms: float = 0.5,
        refractory_ms: float = 2.0,
        warmup_ms: float = 250.0,
        noise_tau_s: float = 2.0,
        blank_ms: float = 2.5,
        artifact_threshold: Optional[float] = 500.0,
    ) -> None:
        self._inner = _core.Detector(
            n_channels, float(sample_rate), band=band, threshold=threshold, polarity=polarity,
            peak_window_ms=peak_window_ms, refractory_ms=refractory_ms, warmup_ms=warmup_ms,
            noise_tau_s=noise_tau_s, blank_ms=blank_ms, artifact_threshold=artifact_threshold,
        )

    def process(self, data: np.ndarray) -> Spikes:
        """Process a ``(n_frames, n_channels)`` chunk. The GIL is released while it runs."""
        return Spikes._from(self._inner.process(_as_frames(data, self.n_channels)))

    def flush(self) -> Spikes:
        """Emit spikes still inside their peak window (end of a recording)."""
        return Spikes._from(self._inner.flush())

    def stimulate(self, frame: int) -> None:
        """Announce a stimulus at absolute ``frame``; blanks ``blank_ms`` from there."""
        self._inner.stimulate(int(frame))

    def blank(self, start: int, length: int) -> None:
        self._inner.blank(int(start), int(length))

    def reset(self) -> None:
        self._inner.reset()

    @property
    def frame(self) -> int:
        """Absolute index of the next frame to be processed."""
        return self._inner.frame

    @property
    def ready(self) -> bool:
        """False during the noise warm-up, when no spikes are reported."""
        return self._inner.ready

    @property
    def n_channels(self) -> int:
        return self._inner.n_channels

    @property
    def sample_rate(self) -> float:
        return self._inner.sample_rate

    @property
    def noise(self) -> np.ndarray:
        """Robust noise sigma per channel."""
        return self._inner.noise

    @property
    def thresholds(self) -> np.ndarray:
        return self._inner.thresholds

    @property
    def stats(self) -> dict:
        return self._inner.stats

    def __repr__(self) -> str:
        return (f"Detector(n_channels={self.n_channels}, sample_rate={self.sample_rate:g}, "
                f"frame={self.frame}, ready={self.ready})")


def detect(
    data: np.ndarray,
    sample_rate: float,
    stim_frames: Optional[Iterable[int]] = None,
    **options,
) -> Spikes:
    """Detect spikes in a whole recording at once."""
    data = _as_frames(data, data.shape[1] if np.ndim(data) == 2 else 1)
    det = Detector(data.shape[1], sample_rate, **options)
    for s in stim_frames if stim_frames is not None else ():
        det.stimulate(int(s))
    return (det.process(data) + det.flush()).sorted()


def bandpass(
    data: np.ndarray,
    sample_rate: float,
    band: Tuple[float, float] = (300.0, 3000.0),
    blank: Optional[Sequence[Tuple[int, int]]] = None,
) -> np.ndarray:
    """The detector's band-pass filter (causal 2nd-order Butterworth high- and low-pass).

    ``blank`` takes ``(start_frame, length)`` windows, held the way the detector holds
    them, so the result is exactly the signal the detector thresholds.
    """
    data = _as_frames(data, data.shape[1] if np.ndim(data) == 2 else 1)
    windows = None if blank is None else [(int(s), int(n)) for s, n in blank]
    return _core.bandpass(data, float(sample_rate), band, windows)


def synthesize(
    n_channels: int = 64,
    sample_rate: float = 25_000.0,
    duration_s: float = 10.0,
    *,
    seed: int = 42,
    stim_times_s: Optional[Sequence[float]] = None,
    noise_uv: float = 6.0,
) -> Tuple[np.ndarray, Spikes, np.ndarray]:
    """Synthetic MEA recording with exact ground truth: ``(data, truth, stim_frames)``.

    White noise, LFP and mains hum, one Poisson unit on most channels, and a decaying
    artifact on every channel at each stimulus. A test signal, not a biophysical model.
    """
    data, truth, stims = _core.synthesize(
        n_channels, float(sample_rate), float(duration_s), seed=seed,
        stim_times_s=None if stim_times_s is None else [float(t) for t in stim_times_s],
        noise_uv=noise_uv,
    )
    return data, Spikes._from(truth), stims


def score(detected: Spikes, truth: Spikes, tolerance: int = 10,
          exclude: Optional[Sequence[Tuple[int, int]]] = None) -> dict:
    """Match detections to ground truth within ``tolerance`` frames, one to one."""
    return _core.score(
        np.ascontiguousarray(detected.frame, np.uint64),
        np.ascontiguousarray(detected.channel, np.uint32),
        np.ascontiguousarray(truth.frame, np.uint64),
        np.ascontiguousarray(truth.channel, np.uint32),
        int(tolerance),
        None if exclude is None else [(int(a), int(b)) for a, b in exclude],
    )
