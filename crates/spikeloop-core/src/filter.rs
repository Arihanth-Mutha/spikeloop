//! Biquad IIR filters with per-channel state stored structure-of-arrays.
//!
//! Each section is a transposed direct form II biquad. State for all channels lives in
//! contiguous `Vec<f32>`s, so the per-frame loop over channels has no cross-channel
//! dependency and compiles to SIMD.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// Normalised biquad coefficients (`a0 == 1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Biquad {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

impl Biquad {
    /// Second-order high-pass (Q = 1/sqrt 2 gives a Butterworth response).
    pub fn highpass(cutoff_hz: f64, sample_rate: f64, q: f64) -> Self {
        let (cos_w, alpha) = Self::prewarp(cutoff_hz, sample_rate, q);
        let a0 = 1.0 + alpha;
        Self::normalised(
            (1.0 + cos_w) / 2.0,
            -(1.0 + cos_w),
            (1.0 + cos_w) / 2.0,
            a0,
            -2.0 * cos_w,
            1.0 - alpha,
        )
    }

    /// Second-order low-pass (Q = 1/sqrt 2 gives a Butterworth response).
    pub fn lowpass(cutoff_hz: f64, sample_rate: f64, q: f64) -> Self {
        let (cos_w, alpha) = Self::prewarp(cutoff_hz, sample_rate, q);
        let a0 = 1.0 + alpha;
        Self::normalised((1.0 - cos_w) / 2.0, 1.0 - cos_w, (1.0 - cos_w) / 2.0, a0, -2.0 * cos_w, 1.0 - alpha)
    }

    fn prewarp(cutoff_hz: f64, sample_rate: f64, q: f64) -> (f64, f64) {
        let w0 = 2.0 * PI * cutoff_hz / sample_rate;
        (w0.cos(), w0.sin() / (2.0 * q))
    }

    fn normalised(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0: (b0 / a0) as f32,
            b1: (b1 / a0) as f32,
            b2: (b2 / a0) as f32,
            a1: (a1 / a0) as f32,
            a2: (a2 / a0) as f32,
        }
    }

    /// Gain at 0 Hz.
    pub fn dc_gain(&self) -> f32 {
        (self.b0 + self.b1 + self.b2) / (1.0 + self.a1 + self.a2)
    }

    /// Magnitude response at `freq_hz`.
    pub fn magnitude(&self, freq_hz: f64, sample_rate: f64) -> f64 {
        let w = 2.0 * PI * freq_hz / sample_rate;
        let (c1, s1, c2, s2) = (w.cos(), w.sin(), (2.0 * w).cos(), (2.0 * w).sin());
        let (b0, b1, b2) = (self.b0 as f64, self.b1 as f64, self.b2 as f64);
        let (a1, a2) = (self.a1 as f64, self.a2 as f64);
        let num = (b0 + b1 * c1 + b2 * c2).hypot(-(b1 * s1 + b2 * s2));
        let den = (1.0 + a1 * c1 + a2 * c2).hypot(-(a1 * s1 + a2 * s2));
        num / den
    }
}

/// A cascade of biquads applied independently to every channel.
#[derive(Clone, Debug)]
pub struct FilterBank {
    sections: Vec<Biquad>,
    z1: Vec<Vec<f32>>,
    z2: Vec<Vec<f32>>,
}

impl FilterBank {
    pub fn new(sections: Vec<Biquad>, n_channels: usize) -> Self {
        let n = sections.len();
        Self { sections, z1: vec![vec![0.0; n_channels]; n], z2: vec![vec![0.0; n_channels]; n] }
    }

    /// Butterworth band-pass built from a 2nd-order high-pass and a 2nd-order low-pass.
    pub fn bandpass(n_channels: usize, sample_rate: f64, low_hz: f64, high_hz: f64) -> Self {
        Self::new(
            vec![
                Biquad::highpass(low_hz, sample_rate, FRAC_1_SQRT_2),
                Biquad::lowpass(high_hz, sample_rate, FRAC_1_SQRT_2),
            ],
            n_channels,
        )
    }

    pub fn sections(&self) -> &[Biquad] {
        &self.sections
    }

    /// Filter one frame (one sample per channel).
    #[inline]
    pub fn process_frame(&mut self, input: &[f32], output: &mut [f32]) {
        output.copy_from_slice(input);
        for (s, (z1, z2)) in self.sections.iter().zip(self.z1.iter_mut().zip(self.z2.iter_mut())) {
            let (b0, b1, b2, a1, a2) = (s.b0, s.b1, s.b2, s.a1, s.a2);
            for ((y, z1), z2) in output.iter_mut().zip(z1.iter_mut()).zip(z2.iter_mut()) {
                let x = *y;
                let out = b0 * x + *z1;
                *z1 = b1 * x - a1 * out + *z2;
                *z2 = b2 * x - a2 * out;
                *y = out;
            }
        }
    }

    /// Put every section in the steady state it would reach if `input` had been
    /// constant forever, and write that steady-state output.
    ///
    /// Used while blanking a stimulation artifact: the filter tracks the raw baseline
    /// without being excited by the artifact, so it resumes with no step transient.
    #[inline]
    pub fn hold_frame(&mut self, input: &[f32], output: &mut [f32]) {
        output.copy_from_slice(input);
        for (s, (z1, z2)) in self.sections.iter().zip(self.z1.iter_mut().zip(self.z2.iter_mut())) {
            let g = s.dc_gain();
            for ((y, z1), z2) in output.iter_mut().zip(z1.iter_mut()).zip(z2.iter_mut()) {
                let u = *y;
                let out = g * u;
                *z1 = out - s.b0 * u;
                *z2 = s.b2 * u - s.a2 * out;
                *y = out;
            }
        }
    }

    pub fn reset(&mut self) {
        for z in self.z1.iter_mut().chain(self.z2.iter_mut()) {
            z.fill(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 25_000.0;

    #[test]
    fn butterworth_is_minus_3db_at_cutoff() {
        let hp = Biquad::highpass(300.0, FS, FRAC_1_SQRT_2);
        let lp = Biquad::lowpass(3000.0, FS, FRAC_1_SQRT_2);
        assert!((hp.magnitude(300.0, FS) - FRAC_1_SQRT_2).abs() < 1e-3);
        assert!((lp.magnitude(3000.0, FS) - FRAC_1_SQRT_2).abs() < 1e-3);
        assert!(hp.magnitude(10.0, FS) < 0.01);
        assert!(lp.magnitude(10_000.0, FS) < 0.05);
    }

    #[test]
    fn highpass_rejects_dc_and_lowpass_passes_it() {
        assert!(Biquad::highpass(300.0, FS, FRAC_1_SQRT_2).dc_gain().abs() < 1e-5);
        assert!((Biquad::lowpass(3000.0, FS, FRAC_1_SQRT_2).dc_gain() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn sine_in_band_passes_and_out_of_band_is_removed() {
        let mut bank = FilterBank::bandpass(2, FS, 300.0, 3000.0);
        let mut out = [0.0f32; 2];
        let (mut peak_in, mut peak_out) = (0.0f32, 0.0f32);
        for i in 0..25_000 {
            let t = i as f64 / FS;
            let frame = [(2.0 * PI * 1000.0 * t).sin() as f32, (2.0 * PI * 20.0 * t).sin() as f32];
            bank.process_frame(&frame, &mut out);
            if i > 5_000 {
                peak_in = peak_in.max(out[0].abs());
                peak_out = peak_out.max(out[1].abs());
            }
        }
        assert!(peak_in > 0.9, "1 kHz should pass, got {peak_in}");
        assert!(peak_out < 0.01, "20 Hz should be removed, got {peak_out}");
    }

    #[test]
    fn hold_then_resume_has_no_transient() {
        let mut bank = FilterBank::bandpass(1, FS, 300.0, 3000.0);
        let mut out = [0.0f32];
        for _ in 0..100 {
            bank.hold_frame(&[250.0], &mut out);
        }
        // Resume on the same baseline: a high-pass must output ~0, not ring.
        let mut worst = 0.0f32;
        for _ in 0..1000 {
            bank.process_frame(&[250.0], &mut out);
            worst = worst.max(out[0].abs());
        }
        assert!(worst < 1e-3, "transient after hold: {worst}");
    }
}
