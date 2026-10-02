"""Render docs/pipeline.png: raw voltage, filtered signal, thresholds and detections
around a stimulus, with and without artifact blanking.

    python bench/figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import spikeloop as sl

FS = 25_000
CHANNEL = None  # pick the busiest channel automatically

data, truth, stims = sl.synthesize(n_channels=16, duration_s=3.0, seed=11,
                                   stim_times_s=[1.5, 2.25])
channel = CHANNEL if CHANNEL is not None else int(np.bincount(truth.channel).argmax())
stim = int(stims[0])
lo, hi = stim - int(0.030 * FS), stim + int(0.045 * FS)
t_ms = (np.arange(lo, hi) - stim) / FS * 1000

blanked = sl.detect(data, FS, stim_frames=stims)
naive = sl.detect(data, FS, artifact_threshold=None)
detector = sl.Detector(16, FS)
detector.process(data[: stim - 1])
threshold = detector.thresholds[channel]
blank = round(2.5e-3 * FS)
plain = sl.bandpass(data, FS)[:, channel]
held = sl.bandpass(data, FS, blank=[(int(s), blank) for s in stims])[:, channel]


def in_view(spikes: sl.Spikes) -> np.ndarray:
    m = (spikes.channel == channel) & (spikes.frame >= lo) & (spikes.frame < hi)
    return spikes.frame[m].astype(np.int64)


ink, muted, accent, bad, good = "#1d1f23", "#8a8f98", "#2f6fdb", "#c2313a", "#1f7a4d"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
                     "xtick.color": muted, "ytick.color": muted,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True,
                         gridspec_kw={"height_ratios": [1.1, 1, 1]})

ax = axes[0]
ax.plot(t_ms, data[lo:hi, channel], color=ink, lw=0.8)
ax.set_ylabel("raw (uV)")
ax.set_title(f"channel {channel}: stimulus at t = 0 ms", loc="left", fontsize=11, color=ink)
ax.set_ylim(np.percentile(data[lo:hi, channel], 0.5) - 40, np.percentile(data[lo:hi, channel], 99.5) + 40)
edge = ax.get_ylim()[0] if data[stim, channel] < 0 else ax.get_ylim()[1]
ax.annotate("stimulation artifact (off scale)", xy=(0.4, edge), xytext=(6, edge * 0.6),
            color=bad, fontsize=9, arrowprops={"arrowstyle": "->", "color": bad})

for ax, spikes, filtered, title in (
        (axes[1], naive, plain, "no blanking: the filter rings on the artifact"),
        (axes[2], blanked, held, "stimulus announced: 2.5 ms blanked, filter held")):
    ax.plot(t_ms, filtered[lo:hi], color=ink, lw=0.8)
    ax.axhline(-threshold, color=accent, lw=1, ls="--")
    ax.text(t_ms[0], -threshold * 1.6, f"-{threshold:.1f} uV (4.5 sigma, MAD)", color=accent,
            fontsize=8.5)
    frames = in_view(spikes)
    true_frames = truth.frame[(truth.channel == channel) & (truth.frame >= lo)
                              & (truth.frame < hi)].astype(np.int64)
    hits = [f for f in frames if np.any(np.abs(true_frames - f) <= 10)]
    misses = [f for f in frames if f not in hits]
    ax.scatter((np.array(hits) - stim) / FS * 1000, filtered[hits], color=good, s=28, zorder=3,
               label="detected, matches ground truth")
    if misses:
        ax.scatter((np.array(misses) - stim) / FS * 1000, filtered[misses], color=bad,
                   marker="x", s=40, zorder=3, label="false detection")
    if "blanked" in title:
        ax.axvspan(0, 2.5, color=accent, alpha=0.12, lw=0, label="blanked")
    ax.set_ylim(-max(80, 3 * threshold), max(60, 2 * threshold))
    ax.set_ylabel("300-3000 Hz (uV)")
    ax.set_title(title, loc="left", fontsize=10, color=muted)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
axes[2].set_xlabel("time from stimulus (ms)")
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "docs" / "pipeline.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=160)
print(out)
