//! Streaming spike detection for multi-electrode arrays.
//!
//! Raw voltage arrives as interleaved frames (`frame0[ch0..chN], frame1[ch0..chN], ...`),
//! the layout MEA hardware streams. [`Detector`] bandpass-filters each channel, tracks
//! its noise level with a median-absolute-deviation estimate, detects threshold
//! crossings, aligns each spike to its peak, and blanks stimulation artifacts so a
//! closed-loop stimulus does not register as a burst of spikes.
//!
//! State carries across calls, so feeding one-millisecond chunks gives exactly the
//! same spikes as feeding the whole recording at once.
//!
//! ```
//! use spikeloop_core::{Config, Detector};
//!
//! let mut detector = Detector::new(Config::new(64, 25_000.0)).unwrap();
//! let chunk = vec![0.0f32; 64 * 25]; // 1 ms of silence
//! let mut spikes = Vec::new();
//! detector.process(&chunk, &mut spikes).unwrap();
//! assert!(spikes.is_empty());
//! ```

mod detect;
pub mod filter;
pub mod noise;
mod rng;
pub mod score;
pub mod synth;

pub use detect::{Config, ConfigError, Detector, Polarity, Spike, Stats};
pub use score::{score, Score};
pub use synth::{synthesize, Recording, SynthConfig};
