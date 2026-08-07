//! Per-channel noise level from the median absolute value of the filtered signal.
//!
//! `sigma = median(|x|) / 0.6745` is the standard robust estimate for spike detection
//! (Quiroga et al., 2004): unlike the standard deviation, it is barely moved by the
//! spikes themselves. The median is computed exactly over a warm-up window, then
//! tracked with an O(1) stochastic quantile update so thresholds follow slow drift.

/// `median(|x|) / sigma` for Gaussian noise.
pub const MAD_TO_SIGMA: f32 = 0.674_489_75;

#[derive(Clone, Debug)]
pub struct NoiseTracker {
    median_abs: Vec<f32>,
    warmup: Vec<Vec<f32>>,
    warmup_frames: usize,
    rate: f32,
    ready: bool,
}

impl NoiseTracker {
    /// `rate` is the relative step of the quantile tracker per sample; 0 freezes the
    /// estimate after warm-up.
    pub fn new(n_channels: usize, warmup_frames: usize, rate: f32) -> Self {
        let warmup_frames = warmup_frames.max(1);
        Self {
            median_abs: vec![0.0; n_channels],
            warmup: (0..n_channels).map(|_| Vec::with_capacity(warmup_frames)).collect(),
            warmup_frames,
            rate,
            ready: false,
        }
    }

    pub fn ready(&self) -> bool {
        self.ready
    }

    #[inline]
    pub fn update(&mut self, frame: &[f32]) {
        if !self.ready {
            for (buf, &x) in self.warmup.iter_mut().zip(frame) {
                buf.push(x.abs());
            }
            if self.warmup[0].len() >= self.warmup_frames {
                self.finish_warmup();
            }
            return;
        }
        if self.rate == 0.0 {
            return;
        }
        let up = 1.0 + self.rate;
        let down = 1.0 - self.rate;
        for (m, &x) in self.median_abs.iter_mut().zip(frame) {
            // Converges where P(|x| > m) = 1/2, i.e. to the median of |x|.
            *m *= if x.abs() > *m { up } else { down };
        }
    }

    fn finish_warmup(&mut self) {
        for (m, buf) in self.median_abs.iter_mut().zip(self.warmup.iter_mut()) {
            let mid = buf.len() / 2;
            let (_, median, _) = buf.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
            // A dead channel would pin the tracker at zero; give it a floor.
            *m = median.max(f32::MIN_POSITIVE);
        }
        self.warmup = Vec::new();
        self.ready = true;
    }

    /// Robust noise standard deviation for `channel` (0 until warm-up completes).
    pub fn sigma(&self, channel: usize) -> f32 {
        self.median_abs[channel] / MAD_TO_SIGMA
    }

    pub fn sigmas(&self) -> Vec<f32> {
        (0..self.median_abs.len()).map(|c| self.sigma(c)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn recovers_gaussian_sigma_despite_spikes() {
        let mut rng = Rng::new(1);
        let mut tracker = NoiseTracker::new(1, 20_000, 0.0);
        for i in 0..20_000 {
            let spike = if i % 400 == 0 { -150.0 } else { 0.0 };
            tracker.update(&[(rng.normal() * 5.0) as f32 + spike]);
        }
        assert!(tracker.ready());
        let s = tracker.sigma(0);
        assert!((s - 5.0).abs() < 0.2, "sigma {s}");
    }

    #[test]
    fn tracker_follows_a_level_change() {
        let mut rng = Rng::new(2);
        let mut tracker = NoiseTracker::new(1, 5_000, 2e-4);
        for _ in 0..5_000 {
            tracker.update(&[(rng.normal() * 5.0) as f32]);
        }
        for _ in 0..200_000 {
            tracker.update(&[(rng.normal() * 10.0) as f32]);
        }
        let s = tracker.sigma(0);
        assert!((s - 10.0).abs() < 0.8, "sigma {s}");
    }
}
