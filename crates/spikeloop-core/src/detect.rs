use std::collections::VecDeque;
use std::fmt;

use crate::filter::FilterBank;
use crate::noise::NoiseTracker;

/// Which deflection counts as a spike. Extracellular spikes are usually negative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Polarity {
    Negative,
    Positive,
    Both,
}

/// Detector settings. Times are in milliseconds, voltages in the input's units
/// (typically microvolts).
#[derive(Clone, Debug)]
pub struct Config {
    pub n_channels: usize,
    pub sample_rate: f64,
    /// Band-pass corners in Hz.
    pub band_hz: (f64, f64),
    /// Threshold in multiples of each channel's robust noise sigma.
    pub threshold: f32,
    pub polarity: Polarity,
    /// After a crossing, how long to keep looking for the peak.
    pub peak_window_ms: f64,
    /// Dead time after a detected spike on the same channel.
    pub refractory_ms: f64,
    /// Signal used to estimate noise before detection starts.
    pub warmup_ms: f64,
    /// Time constant of the noise tracker after warm-up (0 freezes it).
    pub noise_tau_s: f64,
    /// Blanking applied after each stimulus or detected artifact.
    pub blank_ms: f64,
    /// Any raw sample beyond this magnitude starts (or extends) blanking on all
    /// channels. `None` disables automatic artifact detection.
    pub artifact_threshold: Option<f32>,
}

impl Config {
    /// Defaults suited to 25 kHz MEA recordings in microvolts.
    pub fn new(n_channels: usize, sample_rate: f64) -> Self {
        Self {
            n_channels,
            sample_rate,
            band_hz: (300.0, 3000.0),
            threshold: 4.5,
            polarity: Polarity::Negative,
            peak_window_ms: 0.5,
            refractory_ms: 2.0,
            warmup_ms: 250.0,
            noise_tau_s: 2.0,
            blank_ms: 2.5,
            artifact_threshold: Some(500.0),
        }
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        let nyquist = self.sample_rate / 2.0;
        let (lo, hi) = self.band_hz;
        let check = |ok: bool, msg: &str| if ok { Ok(()) } else { Err(ConfigError(msg.into())) };
        check(self.n_channels > 0, "n_channels must be at least 1")?;
        check(self.sample_rate.is_finite() && self.sample_rate > 0.0, "sample_rate must be positive")?;
        check(lo > 0.0 && lo < hi && hi < nyquist, "band must satisfy 0 < low < high < sample_rate / 2")?;
        check(self.threshold.is_finite() && self.threshold > 0.0, "threshold must be positive")?;
        check(self.peak_window_ms >= 0.0, "peak_window_ms must be >= 0")?;
        check(self.refractory_ms >= 0.0, "refractory_ms must be >= 0")?;
        check(self.warmup_ms > 0.0, "warmup_ms must be positive")?;
        check(self.noise_tau_s >= 0.0, "noise_tau_s must be >= 0")?;
        check(self.blank_ms >= 0.0, "blank_ms must be >= 0")?;
        check(self.artifact_threshold.map_or(true, |t| t > 0.0), "artifact_threshold must be positive")
    }

    fn frames(&self, ms: f64) -> u64 {
        (ms * self.sample_rate / 1000.0).round() as u64
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

/// One detected spike, aligned to its peak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spike {
    /// Absolute frame index since the detector started (or was reset).
    pub frame: u64,
    pub channel: u32,
    /// Filtered amplitude at the peak (signed).
    pub amplitude: f32,
}

/// Running counters, useful for monitoring a live stream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub frames: u64,
    pub blanked_frames: u64,
    pub artifacts: u64,
    pub spikes: u64,
}

#[derive(Clone, Copy, Debug)]
struct Event {
    peak_frame: u64,
    peak_score: f32,
    amplitude: f32,
    ends: u64,
}

/// Streaming multi-channel spike detector. See the crate docs for the pipeline.
#[derive(Clone, Debug)]
pub struct Detector {
    cfg: Config,
    filter: FilterBank,
    noise: NoiseTracker,
    frame: u64,
    events: Vec<Option<Event>>,
    refractory_until: Vec<u64>,
    blank_until: u64,
    in_artifact: bool,
    scheduled: VecDeque<(u64, u64)>,
    filtered: Vec<f32>,
    peak_window: u64,
    refractory: u64,
    blank_len: u64,
    stats: Stats,
}

impl Detector {
    pub fn new(cfg: Config) -> Result<Self, ConfigError> {
        cfg.validate()?;
        let n = cfg.n_channels;
        let rate =
            if cfg.noise_tau_s > 0.0 { (1.0 / (cfg.sample_rate * cfg.noise_tau_s)) as f32 } else { 0.0 };
        Ok(Self {
            filter: FilterBank::bandpass(n, cfg.sample_rate, cfg.band_hz.0, cfg.band_hz.1),
            noise: NoiseTracker::new(n, cfg.frames(cfg.warmup_ms) as usize, rate),
            frame: 0,
            events: vec![None; n],
            refractory_until: vec![0; n],
            blank_until: 0,
            in_artifact: false,
            scheduled: VecDeque::new(),
            filtered: vec![0.0; n],
            peak_window: cfg.frames(cfg.peak_window_ms),
            refractory: cfg.frames(cfg.refractory_ms),
            blank_len: cfg.frames(cfg.blank_ms),
            stats: Stats::default(),
            cfg,
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Index of the next frame to be processed.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// Whether the noise warm-up has finished and detection is live.
    pub fn ready(&self) -> bool {
        self.noise.ready()
    }

    /// Robust noise sigma per channel (zeros during warm-up).
    pub fn noise_sigmas(&self) -> Vec<f32> {
        self.noise.sigmas()
    }

    /// Current detection threshold per channel, in input units.
    pub fn thresholds(&self) -> Vec<f32> {
        self.noise.sigmas().into_iter().map(|s| s * self.cfg.threshold).collect()
    }

    /// Announce a stimulus delivered at absolute `frame`: blank `blank_ms` from there.
    pub fn stimulate(&mut self, frame: u64) {
        self.blank(frame, self.blank_len);
    }

    /// Blank `[start, start + len)` on all channels. Windows may be scheduled ahead.
    pub fn blank(&mut self, start: u64, len: u64) {
        let window = (start, start.saturating_add(len));
        let at = self.scheduled.partition_point(|w| w.0 <= start);
        self.scheduled.insert(at, window);
    }

    /// Process interleaved frames (`data.len()` must be a multiple of `n_channels`),
    /// appending detected spikes to `out`. Returns the number of spikes added.
    pub fn process(&mut self, data: &[f32], out: &mut Vec<Spike>) -> Result<usize, ConfigError> {
        let n = self.cfg.n_channels;
        if data.len() % n != 0 {
            return Err(ConfigError(format!("got {} samples, not a multiple of {} channels", data.len(), n)));
        }
        let before = out.len();
        for raw in data.chunks_exact(n) {
            self.step(raw, out);
        }
        let added = out.len() - before;
        self.stats.spikes += added as u64;
        Ok(added)
    }

    /// Emit spikes whose peak window has not closed yet (end of a recording).
    pub fn flush(&mut self, out: &mut Vec<Spike>) -> usize {
        let before = out.len();
        for (c, ev) in self.events.iter_mut().enumerate() {
            if let Some(e) = ev.take() {
                out.push(Spike { frame: e.peak_frame, channel: c as u32, amplitude: e.amplitude });
                self.refractory_until[c] = e.peak_frame + self.refractory + 1;
            }
        }
        let added = out.len() - before;
        self.stats.spikes += added as u64;
        added
    }

    /// Forget all state, including the noise estimate.
    pub fn reset(&mut self) {
        *self = Self::new(self.cfg.clone()).expect("config was already validated");
    }

    #[inline]
    fn step(&mut self, raw: &[f32], out: &mut Vec<Spike>) {
        let f = self.frame;
        self.frame += 1;
        self.stats.frames += 1;

        if let Some(limit) = self.cfg.artifact_threshold {
            let hit = raw.iter().any(|x| x.abs() > limit);
            if hit {
                if !self.in_artifact {
                    self.stats.artifacts += 1;
                }
                self.blank_until = self.blank_until.max(f + self.blank_len.max(1));
            }
            self.in_artifact = hit;
        }
        while let Some(&(start, end)) = self.scheduled.front() {
            if start > f {
                break;
            }
            self.scheduled.pop_front();
            self.blank_until = self.blank_until.max(end);
        }

        if f < self.blank_until {
            self.stats.blanked_frames += 1;
            self.filter.hold_frame(raw, &mut self.filtered);
            // A spike in progress when the artifact arrived cannot be trusted.
            self.events.iter_mut().for_each(|e| *e = None);
            return;
        }

        self.filter.process_frame(raw, &mut self.filtered);
        self.noise.update(&self.filtered);
        if !self.noise.ready() {
            return;
        }

        let k = self.cfg.threshold;
        for c in 0..self.filtered.len() {
            let y = self.filtered[c];
            let score = match self.cfg.polarity {
                Polarity::Negative => -y,
                Polarity::Positive => y,
                Polarity::Both => y.abs(),
            };
            match self.events[c] {
                Some(mut ev) => {
                    if score > ev.peak_score {
                        ev.peak_score = score;
                        ev.peak_frame = f;
                        ev.amplitude = y;
                    }
                    if f >= ev.ends {
                        out.push(Spike { frame: ev.peak_frame, channel: c as u32, amplitude: ev.amplitude });
                        self.refractory_until[c] = ev.peak_frame + self.refractory + 1;
                        self.events[c] = None;
                    } else {
                        self.events[c] = Some(ev);
                    }
                }
                None => {
                    if f >= self.refractory_until[c] && score > k * self.noise.sigma(c) {
                        self.events[c] = Some(Event {
                            peak_frame: f,
                            peak_score: score,
                            amplitude: y,
                            ends: f + self.peak_window,
                        });
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n: usize) -> Config {
        let mut c = Config::new(n, 25_000.0);
        c.warmup_ms = 20.0;
        c
    }

    fn noise(n_frames: usize, n: usize, seed: u64) -> Vec<f32> {
        let mut rng = crate::rng::Rng::new(seed);
        (0..n_frames * n).map(|_| (rng.normal() * 5.0) as f32).collect()
    }

    #[test]
    fn rejects_bad_config_and_ragged_input() {
        let mut bad = cfg(4);
        bad.band_hz = (3000.0, 300.0);
        assert!(Detector::new(bad).is_err());
        let mut d = Detector::new(cfg(4)).unwrap();
        assert!(d.process(&[0.0; 7], &mut Vec::new()).is_err());
    }

    /// Add a 1 ms Gaussian trough centred on `centre`.
    fn inject(data: &mut [f32], n: usize, channel: usize, centre: usize, amplitude: f32) {
        for i in 0..25usize {
            let t = (i as f32 - 12.0) / 4.0;
            data[(centre - 12 + i) * n + channel] -= amplitude * (-t * t / 2.0).exp();
        }
    }

    #[test]
    fn detects_an_injected_spike_at_its_peak() {
        let n = 2;
        let mut data = noise(10_000, n, 3);
        inject(&mut data, n, 1, 8000, 120.0);
        let mut d = Detector::new(cfg(n)).unwrap();
        let mut spikes = Vec::new();
        d.process(&data, &mut spikes).unwrap();
        d.flush(&mut spikes);
        let on_1: Vec<_> = spikes.iter().filter(|s| s.channel == 1).collect();
        assert!(on_1.iter().any(|s| (s.frame as i64 - 8000).abs() <= 6), "{spikes:?}");
        assert!(on_1.iter().all(|s| s.amplitude < 0.0));
    }

    #[test]
    fn chunking_does_not_change_the_result() {
        let n = 8;
        let rec = crate::synth::synthesize(&crate::synth::SynthConfig {
            n_channels: n,
            duration_s: 2.0,
            ..Default::default()
        });
        let run = |chunk_frames: usize| {
            let mut d = Detector::new(Config::new(n, rec.sample_rate)).unwrap();
            for &s in &rec.stim_frames {
                d.stimulate(s);
            }
            let mut out = Vec::new();
            for chunk in rec.data.chunks(chunk_frames * n) {
                d.process(chunk, &mut out).unwrap();
            }
            d.flush(&mut out);
            out
        };
        let whole = run(usize::MAX / n);
        assert!(!whole.is_empty());
        assert_eq!(whole, run(25));
        assert_eq!(whole, run(1));
        assert_eq!(whole, run(997));
    }

    #[test]
    fn stimulation_artifact_is_blanked() {
        let n = 4;
        let mut data = noise(20_000, n, 5);
        for f in 15_000..15_050usize {
            let decay = (-((f - 15_000) as f32) / 10.0).exp();
            for c in 0..n {
                data[f * n + c] += 3000.0 * decay;
            }
        }
        let count = |artifact_threshold: Option<f32>, stim: bool| {
            let mut c = cfg(n);
            c.artifact_threshold = artifact_threshold;
            let mut d = Detector::new(c).unwrap();
            if stim {
                d.stimulate(15_000);
            }
            let mut out = Vec::new();
            d.process(&data, &mut out).unwrap();
            out.iter().filter(|s| (14_990..15_300).contains(&s.frame)).count()
        };
        assert!(count(None, false) >= n, "unblanked artifact should look like spikes");
        assert_eq!(count(None, true), 0, "announced stimulus is blanked");
        assert_eq!(count(Some(500.0), false), 0, "detected artifact is blanked");
    }

    #[test]
    fn refractory_period_suppresses_double_counting() {
        let n = 1;
        let mut data = noise(10_000, n, 9);
        for f in [6000usize, 6030, 6400] {
            inject(&mut data, n, 0, f, 60.0);
        }
        let mut c = cfg(n);
        c.refractory_ms = 2.0; // 50 frames: the spike at 6030 falls inside it
        let mut d = Detector::new(c).unwrap();
        let mut out = Vec::new();
        d.process(&data, &mut out).unwrap();
        let near: Vec<_> = out.iter().filter(|s| s.frame >= 5990).map(|s| s.frame).collect();
        assert_eq!(near.len(), 2, "{near:?}");
        assert!((near[0] as i64 - 6000).abs() <= 6 && (near[1] as i64 - 6400).abs() <= 6);
    }
}
