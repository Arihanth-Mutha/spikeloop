//! Synthetic MEA recordings with exact ground truth, for tests and benchmarks.
//!
//! Each channel gets white noise, a slow LFP oscillation and mains hum (both outside
//! the spike band), and, on active channels, one unit firing as a Poisson process
//! with a refractory period. Stimuli add a large decaying artifact on every channel
//! and can evoke spikes a few milliseconds later. This is a test signal, not a
//! biophysical model.

use crate::detect::Spike;
use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct SynthConfig {
    pub n_channels: usize,
    pub sample_rate: f64,
    pub duration_s: f64,
    /// White noise standard deviation (raw, before filtering).
    pub noise_uv: f64,
    /// Amplitude of a slow (4-10 Hz) LFP oscillation.
    pub lfp_uv: f64,
    /// Amplitude of 50 Hz mains interference.
    pub mains_uv: f64,
    /// Fraction of channels that carry a unit.
    pub active_fraction: f64,
    /// Firing-rate range across active channels.
    pub rate_hz: (f64, f64),
    /// Spike trough amplitude range across active channels.
    pub amplitude_uv: (f64, f64),
    /// Stimulus times; each adds an artifact on every channel.
    pub stim_times_s: Vec<f64>,
    pub artifact_uv: f64,
    pub artifact_tau_ms: f64,
    /// Probability that an active channel fires 3-12 ms after a stimulus.
    pub evoked_probability: f64,
    pub seed: u64,
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self {
            n_channels: 64,
            sample_rate: 25_000.0,
            duration_s: 10.0,
            noise_uv: 6.0,
            lfp_uv: 60.0,
            mains_uv: 15.0,
            active_fraction: 0.75,
            rate_hz: (2.0, 20.0),
            amplitude_uv: (30.0, 150.0),
            stim_times_s: vec![0.75, 1.5, 2.25, 3.0, 4.5, 6.0, 7.5, 9.0],
            artifact_uv: 1500.0,
            artifact_tau_ms: 0.4,
            evoked_probability: 0.6,
            seed: 42,
        }
    }
}

/// A recording and the spikes that were put into it.
#[derive(Clone, Debug)]
pub struct Recording {
    pub n_channels: usize,
    pub sample_rate: f64,
    /// Interleaved frames: `data[frame * n_channels + channel]`.
    pub data: Vec<f32>,
    /// True spikes, at the trough of each waveform.
    pub truth: Vec<Spike>,
    pub stim_frames: Vec<u64>,
}

impl Recording {
    pub fn n_frames(&self) -> usize {
        self.data.len() / self.n_channels
    }
}

/// Biphasic extracellular waveform: a sharp trough, then a slower positive rebound.
fn waveform(t_ms: f64) -> f64 {
    let trough = (-(t_ms / 0.15).powi(2) / 2.0).exp();
    let rebound = 0.35 * (-((t_ms - 0.45) / 0.25).powi(2) / 2.0).exp();
    -trough + rebound
}

pub fn synthesize(cfg: &SynthConfig) -> Recording {
    let n = cfg.n_channels;
    let fs = cfg.sample_rate;
    let frames = (cfg.duration_s * fs).round() as usize;
    let mut rng = Rng::new(cfg.seed);
    let mut data = vec![0.0f32; frames * n];
    let mut truth = Vec::new();
    let stim_frames: Vec<u64> =
        cfg.stim_times_s.iter().map(|t| (t * fs).round() as u64).filter(|&f| (f as usize) < frames).collect();

    let half = (1.5e-3 * fs) as i64;
    let kernel: Vec<f32> = (-half..=half).map(|i| waveform(i as f64 * 1000.0 / fs) as f32).collect();
    let refractory_s = 2e-3;

    for c in 0..n {
        let lfp_hz = rng.range(4.0, 10.0);
        let lfp_phase = rng.range(0.0, std::f64::consts::TAU);
        let mains_phase = rng.range(0.0, std::f64::consts::TAU);
        let gain = rng.range(0.7, 1.3) * if rng.uniform() < 0.5 { 1.0 } else { -1.0 };
        for f in 0..frames {
            let t = f as f64 / fs;
            let v = cfg.noise_uv * rng.normal()
                + cfg.lfp_uv * (std::f64::consts::TAU * lfp_hz * t + lfp_phase).sin()
                + cfg.mains_uv * (std::f64::consts::TAU * 50.0 * t + mains_phase).sin();
            data[f * n + c] = v as f32;
        }
        for &s in &stim_frames {
            let len = ((10.0 * cfg.artifact_tau_ms) * fs / 1000.0) as usize;
            for i in 0..len.min(frames - s as usize) {
                let decay = (-(i as f64) * 1000.0 / fs / cfg.artifact_tau_ms).exp();
                data[(s as usize + i) * n + c] += (cfg.artifact_uv * gain * decay) as f32;
            }
        }

        if rng.uniform() >= cfg.active_fraction {
            continue;
        }
        let rate = rng.range(cfg.rate_hz.0, cfg.rate_hz.1);
        let amplitude = rng.range(cfg.amplitude_uv.0, cfg.amplitude_uv.1);
        let mut times = Vec::new();
        let mut t = rng.exponential(1.0 / rate);
        while t < cfg.duration_s {
            times.push(t);
            t += refractory_s + rng.exponential(1.0 / rate);
        }
        for &s in &stim_frames {
            if rng.uniform() < cfg.evoked_probability {
                times.push(s as f64 / fs + rng.range(3e-3, 12e-3));
            }
        }
        times.sort_by(f64::total_cmp);
        let mut last = f64::NEG_INFINITY;
        for t in times {
            if t - last < refractory_s {
                continue; // an evoked spike that collides with a spontaneous one
            }
            last = t;
            let centre = (t * fs).round() as i64;
            if centre - half < 0 || centre + half >= frames as i64 {
                continue;
            }
            for (k, w) in kernel.iter().enumerate() {
                let f = (centre - half + k as i64) as usize;
                data[f * n + c] += (amplitude as f32) * w;
            }
            truth.push(Spike { frame: centre as u64, channel: c as u32, amplitude: -amplitude as f32 });
        }
    }
    truth.sort_by_key(|s| (s.frame, s.channel));
    Recording { n_channels: n, sample_rate: fs, data, truth, stim_frames }
}
