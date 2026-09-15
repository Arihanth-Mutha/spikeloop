//! Python bindings. The Python package in `python/spikeloop` wraps these with
//! friendlier types; everything heavy runs here with the GIL released.

use numpy::{
    IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use spikeloop_core::filter::FilterBank;
use spikeloop_core::{Config, Detector, Polarity, Spike, SynthConfig};

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn spikes_to_dict<'py>(py: Python<'py>, spikes: &[Spike]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("frame", spikes.iter().map(|s| s.frame).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("channel", spikes.iter().map(|s| s.channel).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("amplitude", spikes.iter().map(|s| s.amplitude).collect::<Vec<_>>().into_pyarray(py))?;
    Ok(d)
}

fn spikes_from_arrays(
    frame: PyReadonlyArray1<'_, u64>,
    channel: PyReadonlyArray1<'_, u32>,
) -> PyResult<Vec<Spike>> {
    let (f, c) = (frame.as_array(), channel.as_array());
    if f.len() != c.len() {
        return Err(value_error("frame and channel arrays differ in length"));
    }
    Ok(f.iter().zip(c.iter()).map(|(&frame, &channel)| Spike { frame, channel, amplitude: 0.0 }).collect())
}

fn contiguous<'a>(data: &'a PyReadonlyArray2<'_, f32>, n_channels: usize) -> PyResult<&'a [f32]> {
    let shape = data.shape();
    if shape[1] != n_channels {
        return Err(value_error(format!(
            "expected {n_channels} channels (shape (n_frames, {n_channels})), got shape {shape:?}"
        )));
    }
    data.as_slice().map_err(|_| value_error("data must be C-contiguous float32"))
}

#[pyclass(name = "Detector", module = "spikeloop._core")]
struct PyDetector {
    inner: Detector,
    buffer: Vec<Spike>,
}

#[pymethods]
impl PyDetector {
    #[new]
    #[pyo3(signature = (
        n_channels, sample_rate, *, band=(300.0, 3000.0), threshold=4.5, polarity="negative",
        peak_window_ms=0.5, refractory_ms=2.0, warmup_ms=250.0, noise_tau_s=2.0, blank_ms=2.5,
        artifact_threshold=Some(500.0)
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_channels: usize,
        sample_rate: f64,
        band: (f64, f64),
        threshold: f32,
        polarity: &str,
        peak_window_ms: f64,
        refractory_ms: f64,
        warmup_ms: f64,
        noise_tau_s: f64,
        blank_ms: f64,
        artifact_threshold: Option<f32>,
    ) -> PyResult<Self> {
        let polarity = match polarity {
            "negative" => Polarity::Negative,
            "positive" => Polarity::Positive,
            "both" => Polarity::Both,
            other => {
                return Err(value_error(format!(
                    "polarity must be negative, positive or both, got {other:?}"
                )))
            }
        };
        let cfg = Config {
            n_channels,
            sample_rate,
            band_hz: band,
            threshold,
            polarity,
            peak_window_ms,
            refractory_ms,
            warmup_ms,
            noise_tau_s,
            blank_ms,
            artifact_threshold,
        };
        Ok(Self { inner: Detector::new(cfg).map_err(value_error)?, buffer: Vec::new() })
    }

    /// Process a (n_frames, n_channels) float32 chunk; returns the spikes it completed.
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let slice = contiguous(&data, self.inner.config().n_channels)?;
        self.buffer.clear();
        let (inner, buffer) = (&mut self.inner, &mut self.buffer);
        py.allow_threads(|| inner.process(slice, buffer)).map_err(value_error)?;
        spikes_to_dict(py, &self.buffer)
    }

    /// Emit spikes whose peak window is still open (call at the end of a recording).
    fn flush<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.buffer.clear();
        self.inner.flush(&mut self.buffer);
        spikes_to_dict(py, &self.buffer)
    }

    fn stimulate(&mut self, frame: u64) {
        self.inner.stimulate(frame);
    }

    fn blank(&mut self, start: u64, length: u64) {
        self.inner.blank(start, length);
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn frame(&self) -> u64 {
        self.inner.frame()
    }

    #[getter]
    fn ready(&self) -> bool {
        self.inner.ready()
    }

    #[getter]
    fn n_channels(&self) -> usize {
        self.inner.config().n_channels
    }

    #[getter]
    fn sample_rate(&self) -> f64 {
        self.inner.config().sample_rate
    }

    #[getter]
    fn noise<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.noise_sigmas().into_pyarray(py)
    }

    #[getter]
    fn thresholds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.thresholds().into_pyarray(py)
    }

    #[getter]
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stats();
        let d = PyDict::new(py);
        d.set_item("frames", s.frames)?;
        d.set_item("blanked_frames", s.blanked_frames)?;
        d.set_item("artifacts", s.artifacts)?;
        d.set_item("spikes", s.spikes)?;
        Ok(d)
    }
}

/// Band-pass filter a (n_frames, n_channels) float32 array with the detector's filter.
/// Frames inside `blank` windows `(start, length)` are held exactly as the detector
/// holds them, so the output is the signal the detector actually thresholds.
#[pyfunction]
#[pyo3(signature = (data, sample_rate, band=(300.0, 3000.0), blank=None))]
fn bandpass<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    sample_rate: f64,
    band: (f64, f64),
    blank: Option<Vec<(u64, u64)>>,
) -> PyResult<Bound<'py, PyAny>> {
    let n = data.shape()[1];
    if !(band.0 > 0.0 && band.0 < band.1 && band.1 < sample_rate / 2.0) {
        return Err(value_error("band must satisfy 0 < low < high < sample_rate / 2"));
    }
    let slice = contiguous(&data, n)?;
    let windows = blank.unwrap_or_default();
    let mut out = vec![0.0f32; slice.len()];
    py.allow_threads(|| {
        let mut bank = FilterBank::bandpass(n, sample_rate, band.0, band.1);
        for (f, (raw, filtered)) in slice.chunks_exact(n).zip(out.chunks_exact_mut(n)).enumerate() {
            let f = f as u64;
            if windows.iter().any(|&(start, len)| f >= start && f < start + len) {
                bank.hold_frame(raw, filtered);
            } else {
                bank.process_frame(raw, filtered);
            }
        }
    });
    let frames = slice.len() / n.max(1);
    Ok(out.into_pyarray(py).reshape([frames, n])?.into_any())
}

/// `(data, truth, stim_frames)` as handed to Python.
type SynthOutput<'py> = (Bound<'py, PyAny>, Bound<'py, PyDict>, Bound<'py, PyArray1<u64>>);

/// Synthetic recording with ground truth. Returns (data, truth, stim_frames).
#[pyfunction]
#[pyo3(signature = (n_channels=64, sample_rate=25_000.0, duration_s=10.0, *, seed=42, stim_times_s=None, noise_uv=6.0))]
fn synthesize<'py>(
    py: Python<'py>,
    n_channels: usize,
    sample_rate: f64,
    duration_s: f64,
    seed: u64,
    stim_times_s: Option<Vec<f64>>,
    noise_uv: f64,
) -> PyResult<SynthOutput<'py>> {
    if n_channels == 0 || duration_s <= 0.0 || sample_rate <= 0.0 {
        return Err(value_error("n_channels, sample_rate and duration_s must be positive"));
    }
    let mut cfg = SynthConfig { n_channels, sample_rate, duration_s, seed, noise_uv, ..Default::default() };
    if let Some(t) = stim_times_s {
        cfg.stim_times_s = t;
    }
    let rec = py.allow_threads(|| spikeloop_core::synthesize(&cfg));
    let frames = rec.n_frames();
    let data = rec.data.into_pyarray(py).reshape([frames, n_channels])?.into_any();
    Ok((data, spikes_to_dict(py, &rec.truth)?, rec.stim_frames.into_pyarray(py)))
}

/// Precision, recall and F1 of detected spikes against ground truth.
#[pyfunction]
#[pyo3(signature = (frame, channel, true_frame, true_channel, tolerance, exclude=None))]
fn score<'py>(
    py: Python<'py>,
    frame: PyReadonlyArray1<'py, u64>,
    channel: PyReadonlyArray1<'py, u32>,
    true_frame: PyReadonlyArray1<'py, u64>,
    true_channel: PyReadonlyArray1<'py, u32>,
    tolerance: u64,
    exclude: Option<Vec<(u64, u64)>>,
) -> PyResult<Bound<'py, PyDict>> {
    let detected = spikes_from_arrays(frame, channel)?;
    let truth = spikes_from_arrays(true_frame, true_channel)?;
    let s = spikeloop_core::score(&detected, &truth, tolerance, &exclude.unwrap_or_default());
    let d = PyDict::new(py);
    d.set_item("true_positives", s.true_positives)?;
    d.set_item("false_positives", s.false_positives)?;
    d.set_item("false_negatives", s.false_negatives)?;
    d.set_item("precision", s.precision)?;
    d.set_item("recall", s.recall)?;
    d.set_item("f1", s.f1)?;
    Ok(d)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDetector>()?;
    m.add_function(wrap_pyfunction!(bandpass, m)?)?;
    m.add_function(wrap_pyfunction!(synthesize, m)?)?;
    m.add_function(wrap_pyfunction!(score, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
