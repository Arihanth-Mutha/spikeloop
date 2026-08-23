//! Accuracy and throughput on a synthetic 64-channel, 25 kHz recording.
//!
//! cargo run --release --example evaluate

use std::time::Instant;

use spikeloop_core::{score, synthesize, Config, Detector, SynthConfig};

fn main() {
    let synth = SynthConfig::default();
    let started = Instant::now();
    let rec = synthesize(&synth);
    let n = rec.n_channels;
    println!(
        "synthetic recording: {} channels x {:.0} s at {:.0} Hz, {} true spikes, {} stimuli ({:.1} s to generate)\n",
        n,
        synth.duration_s,
        rec.sample_rate,
        rec.truth.len(),
        rec.stim_frames.len(),
        started.elapsed().as_secs_f64()
    );

    let cfg = Config::new(n, rec.sample_rate);
    let blank = (cfg.blank_ms * rec.sample_rate / 1000.0).round() as u64;
    let warmup = (cfg.warmup_ms * rec.sample_rate / 1000.0).round() as u64;
    let tolerance = (0.4e-3 * rec.sample_rate) as u64;
    let mut excluded: Vec<(u64, u64)> = rec.stim_frames.iter().map(|&s| (s, s + blank)).collect();
    excluded.push((0, warmup));

    println!(
        "{:<36} {:>9} {:>9} {:>9} {:>7} {:>16}",
        "artifact handling", "precision", "recall", "F1", "false+", "in stim windows"
    );
    for (label, announce, auto) in [
        ("none", false, None),
        ("automatic (|raw| > 500 uV)", false, Some(500.0)),
        ("stimuli announced by the host", true, None),
    ] {
        let mut c = cfg.clone();
        c.artifact_threshold = auto;
        let mut d = Detector::new(c).unwrap();
        if announce {
            for &s in &rec.stim_frames {
                d.stimulate(s);
            }
        }
        let mut spikes = Vec::new();
        d.process(&rec.data, &mut spikes).unwrap();
        d.flush(&mut spikes);
        let s = score(&spikes, &rec.truth, tolerance, &excluded);
        // Evoked spikes arrive 3 ms or more after a stimulus, so detections in the
        // first blank_ms are almost all the artifact itself.
        let in_windows = spikes
            .iter()
            .filter(|sp| rec.stim_frames.iter().any(|&st| sp.frame >= st && sp.frame < st + blank))
            .count();
        println!(
            "{:<36} {:>9.3} {:>9.3} {:>9.3} {:>7} {:>16}",
            label, s.precision, s.recall, s.f1, s.false_positives, in_windows
        );
    }

    println!("\n{:<44} {:>12} {:>14} {:>12}", "chunk size", "Msamples/s", "x real time", "us / chunk");
    let recording_seconds = rec.n_frames() as f64 / rec.sample_rate;
    for chunk_frames in [1usize, 25, 250, 2500] {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut d = Detector::new(cfg.clone()).unwrap();
            let mut out = Vec::with_capacity(1 << 16);
            let t = Instant::now();
            for chunk in rec.data.chunks(chunk_frames * n) {
                d.process(chunk, &mut out).unwrap();
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        let chunks = rec.n_frames().div_ceil(chunk_frames) as f64;
        println!(
            "{:<44} {:>12.0} {:>14.0} {:>12.2}",
            format!("{chunk_frames} frames ({:.2} ms)", chunk_frames as f64 * 1000.0 / rec.sample_rate),
            rec.data.len() as f64 / best / 1e6,
            recording_seconds / best,
            best / chunks * 1e6
        );
    }
}
