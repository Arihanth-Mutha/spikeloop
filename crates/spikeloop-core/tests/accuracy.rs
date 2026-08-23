use spikeloop_core::{score, synthesize, Config, Detector, SynthConfig};

fn recording() -> spikeloop_core::Recording {
    synthesize(&SynthConfig { n_channels: 16, duration_s: 4.0, ..Default::default() })
}

fn run(rec: &spikeloop_core::Recording, announce: bool, auto: Option<f32>) -> Vec<spikeloop_core::Spike> {
    let mut cfg = Config::new(rec.n_channels, rec.sample_rate);
    cfg.artifact_threshold = auto;
    let mut d = Detector::new(cfg).unwrap();
    if announce {
        for &s in &rec.stim_frames {
            d.stimulate(s);
        }
    }
    let mut out = Vec::new();
    d.process(&rec.data, &mut out).unwrap();
    d.flush(&mut out);
    out
}

#[test]
fn high_precision_and_recall_on_synthetic_recording() {
    let rec = recording();
    let blank = 63; // 2.5 ms at 25 kHz
    let mut excluded: Vec<(u64, u64)> = rec.stim_frames.iter().map(|&s| (s, s + blank)).collect();
    excluded.push((0, 6250)); // warm-up
    let s = score(&run(&rec, true, Some(500.0)), &rec.truth, 10, &excluded);
    assert!(s.precision > 0.98, "{s:?}");
    assert!(s.recall > 0.98, "{s:?}");
}

#[test]
fn blanking_removes_artifact_detections() {
    let rec = recording();
    let in_windows = |spikes: &[spikeloop_core::Spike]| {
        spikes
            .iter()
            .filter(|sp| rec.stim_frames.iter().any(|&st| sp.frame >= st && sp.frame < st + 63))
            .count()
    };
    assert!(in_windows(&run(&rec, false, None)) > rec.n_channels);
    assert_eq!(in_windows(&run(&rec, true, None)), 0);
    assert_eq!(in_windows(&run(&rec, false, Some(500.0))), 0);
}

#[test]
fn evoked_spikes_right_after_blanking_are_still_found() {
    let rec = recording();
    let detected = run(&rec, true, None);
    // True spikes 3-12 ms after a stimulus are the evoked responses.
    let evoked: Vec<_> = rec
        .truth
        .iter()
        .filter(|t| rec.stim_frames.iter().any(|&s| t.frame >= s + 75 && t.frame < s + 300))
        .collect();
    assert!(!evoked.is_empty());
    let found = evoked
        .iter()
        .filter(|t| detected.iter().any(|d| d.channel == t.channel && d.frame.abs_diff(t.frame) <= 10))
        .count();
    assert!(found as f64 / evoked.len() as f64 > 0.95, "{found}/{}", evoked.len());
}
