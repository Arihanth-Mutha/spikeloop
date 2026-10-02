"""spikeloop vs a NumPy/SciPy pipeline on the same recording.

The SciPy reference does *less* work than spikeloop: band-pass (sosfilt, carrying
filter state across chunks), a fixed MAD threshold from the first 250 ms, and
threshold crossings with a refractory period. No peak alignment, no adaptive noise
tracking, no artifact blanking. It is a fair lower bound on what an equivalent
Python pipeline costs.

    python bench/compare_scipy.py
"""

from __future__ import annotations

import platform
import time

import numpy as np
from scipy import signal

import spikeloop as sl

FS = 25_000
N_CHANNELS = 64
DURATION_S = 10.0


class ScipyReference:
    def __init__(self, n_channels: int, fs: float):
        self.sos = np.vstack([
            signal.butter(2, 300, "highpass", fs=fs, output="sos"),
            signal.butter(2, 3000, "lowpass", fs=fs, output="sos"),
        ])
        self.zi = np.zeros((self.sos.shape[0], 2, n_channels))
        self.warmup = int(0.25 * fs)
        self.buffer: list[np.ndarray] = []
        self.threshold: np.ndarray | None = None
        self.refractory = int(2e-3 * fs)
        self.last = np.full(n_channels, -10**9)
        self.frame = 0

    def process(self, chunk: np.ndarray) -> int:
        y, self.zi = signal.sosfilt(self.sos, chunk, axis=0, zi=self.zi)
        start = self.frame
        self.frame += len(chunk)
        if self.threshold is None:
            self.buffer.append(y)
            seen = sum(len(b) for b in self.buffer)
            if seen < self.warmup:
                return 0
            w = np.concatenate(self.buffer)[: self.warmup]
            self.threshold = 4.5 * np.median(np.abs(w), axis=0) / 0.6745
            self.buffer = []
            return 0
        below = -y > self.threshold
        prev = np.vstack([np.zeros((1, y.shape[1]), bool), below[:-1]])
        frames, channels = np.nonzero(below & ~prev)
        count = 0
        for f, c in zip(frames + start, channels):
            if f - self.last[c] > self.refractory:
                self.last[c] = f
                count += 1
        return count


def timed(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def main() -> None:
    data, truth, stims = sl.synthesize(N_CHANNELS, FS, DURATION_S)
    data64 = data.astype(np.float64)
    print(f"{N_CHANNELS} channels x {DURATION_S:.0f} s at {FS} Hz | "
          f"{platform.machine()} | Python {platform.python_version()}\n")
    print(f"{'chunk':<16}{'pipeline':<22}{'per chunk':>12}{'x real time':>14}{'speed-up':>10}")
    for chunk in (25, 250, 2500):
        n_chunks = int(np.ceil(len(data) / chunk))

        def run_spikeloop():
            det = sl.Detector(N_CHANNELS, FS)
            for s in stims:
                det.stimulate(int(s))
            for i in range(0, len(data), chunk):
                det.process(data[i:i + chunk])

        def run_scipy():
            ref = ScipyReference(N_CHANNELS, FS)
            for i in range(0, len(data64), chunk):
                ref.process(data64[i:i + chunk])

        t_sf = timed(run_spikeloop)
        t_sp = timed(run_scipy)
        label = f"{chunk} ({chunk / FS * 1000:g} ms)"
        print(f"{label:<16}{'spikeloop':<22}{t_sf / n_chunks * 1e6:>10.1f}us"
              f"{DURATION_S / t_sf:>14.0f}{t_sp / t_sf:>9.1f}x")
        print(f"{'':<16}{'numpy/scipy (less work)':<22}{t_sp / n_chunks * 1e6:>10.1f}us"
              f"{DURATION_S / t_sp:>14.0f}{'':>10}")


if __name__ == "__main__":
    main()
