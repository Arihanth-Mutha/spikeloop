import threading

import numpy as np
import pytest

import spikeloop as sl

FS = 25_000


@pytest.fixture(scope="module")
def recording():
    return sl.synthesize(n_channels=16, duration_s=4.0, seed=3)


def test_synthesize_shapes(recording):
    data, truth, stims = recording
    assert data.shape == (4 * FS, 16)
    assert data.dtype == np.float32 and data.flags.c_contiguous
    assert len(truth) > 100
    assert truth.frame.dtype == np.uint64 and truth.channel.max() < 16
    assert stims.dtype == np.uint64 and len(stims) > 0


def test_accuracy(recording):
    data, truth, stims = recording
    spikes = sl.detect(data, FS, stim_frames=stims)
    blank = round(2.5e-3 * FS)
    exclude = [(int(s), int(s) + blank) for s in stims] + [(0, round(0.25 * FS))]
    result = sl.score(spikes, truth, tolerance=10, exclude=exclude)
    assert result["precision"] > 0.98, result
    assert result["recall"] > 0.98, result


def test_one_ms_chunks_match_one_call(recording):
    data, _, stims = recording
    whole = sl.detect(data, FS, stim_frames=stims)
    det = sl.Detector(16, FS)
    for s in stims:
        det.stimulate(int(s))
    parts = sl.Spikes.empty()
    for start in range(0, len(data), 25):
        parts = parts + det.process(data[start:start + 25])
    parts = (parts + det.flush()).sorted()
    np.testing.assert_array_equal(whole.frame, parts.frame)
    np.testing.assert_array_equal(whole.channel, parts.channel)
    np.testing.assert_array_equal(whole.amplitude, parts.amplitude)


def test_bandpass_matches_scipy():
    signal = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(0)
    x = rng.normal(0, 10, (FS, 4)).astype(np.float32)
    sos = np.vstack([
        signal.butter(2, 300, "highpass", fs=FS, output="sos"),
        signal.butter(2, 3000, "lowpass", fs=FS, output="sos"),
    ])
    expected = signal.sosfilt(sos, x.astype(np.float64), axis=0)
    got = sl.bandpass(x, FS)
    np.testing.assert_allclose(got, expected, atol=2e-3 * np.abs(expected).max())


def test_noise_estimate_is_robust(recording):
    data, _, _ = recording
    det = sl.Detector(16, FS)
    det.process(data)
    assert det.ready
    filtered_sigma = sl.bandpass(data, FS)[FS:].std(axis=0)
    # MAD-based sigma ignores spikes, so it sits at or below the raw std.
    assert np.all(det.noise > 0)
    assert np.all(det.noise <= filtered_sigma * 1.05)
    np.testing.assert_allclose(det.thresholds, det.noise * 4.5, rtol=1e-6)


def test_no_spikes_during_warmup():
    det = sl.Detector(2, FS, warmup_ms=100)
    out = det.process(np.full((FS // 20, 2), -1000.0, np.float32))
    assert len(out) == 0 and not det.ready


def test_artifact_is_blanked_without_announcement():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 5, (FS, 8)).astype(np.float32)
    x[20_000:20_050] += 3000 * np.exp(-np.arange(50) / 10)[:, None]
    det = sl.Detector(8, FS)
    spikes = det.process(x) + det.flush()
    assert not np.any((spikes.frame >= 20_000) & (spikes.frame < 20_100))
    assert det.stats["artifacts"] == 1


def test_rejects_wrong_channel_count_and_bad_config():
    det = sl.Detector(4, FS)
    with pytest.raises(ValueError, match="expected 4 channels"):
        det.process(np.zeros((10, 3), np.float32))
    with pytest.raises(ValueError, match="band"):
        sl.Detector(4, FS, band=(3000, 300))
    with pytest.raises(ValueError, match="polarity"):
        sl.Detector(4, FS, polarity="sideways")


def test_accepts_float64_and_fortran_order(recording):
    data, _, _ = recording
    a = sl.detect(data[:FS], FS)
    b = sl.detect(np.asfortranarray(data[:FS].astype(np.float64)), FS)
    np.testing.assert_array_equal(a.frame, b.frame)


def test_releases_the_gil():
    # Two detectors on two threads must both finish (and not deadlock).
    data, _, _ = sl.synthesize(n_channels=32, duration_s=2.0)
    results = []

    def work():
        results.append(len(sl.detect(data, FS)))

    threads = [threading.Thread(target=work) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 2 and results[0] == results[1]
