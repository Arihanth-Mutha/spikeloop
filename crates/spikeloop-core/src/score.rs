//! Match detected spikes to ground truth.

use crate::detect::Spike;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Score {
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

/// One-to-one matching on each channel: a detection matches the nearest unmatched true
/// spike within `tolerance` frames. Spikes whose frame falls in any `exclude` window
/// (for example, blanked stimulation periods) are ignored on both sides.
pub fn score(detected: &[Spike], truth: &[Spike], tolerance: u64, exclude: &[(u64, u64)]) -> Score {
    let excluded = |f: u64| exclude.iter().any(|&(a, b)| f >= a && f < b);
    let n_channels = detected.iter().chain(truth).map(|s| s.channel as usize + 1).max().unwrap_or(0);
    let mut det: Vec<Vec<u64>> = vec![Vec::new(); n_channels];
    let mut tru: Vec<Vec<u64>> = vec![Vec::new(); n_channels];
    for s in detected.iter().filter(|s| !excluded(s.frame)) {
        det[s.channel as usize].push(s.frame);
    }
    for s in truth.iter().filter(|s| !excluded(s.frame)) {
        tru[s.channel as usize].push(s.frame);
    }

    let (mut tp, mut fp, mut fn_) = (0, 0, 0);
    for (d, t) in det.iter_mut().zip(tru.iter_mut()) {
        d.sort_unstable();
        t.sort_unstable();
        let mut used = vec![false; t.len()];
        let mut lo = 0;
        for &f in d.iter() {
            while lo < t.len() && t[lo] + tolerance < f {
                lo += 1;
            }
            let best = (lo..t.len())
                .take_while(|&j| t[j] <= f + tolerance)
                .filter(|&j| !used[j])
                .min_by_key(|&j| t[j].abs_diff(f));
            match best {
                Some(j) => {
                    used[j] = true;
                    tp += 1;
                }
                None => fp += 1,
            }
        }
        fn_ += used.iter().filter(|u| !**u).count();
    }
    let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 1.0 };
    let recall = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 1.0 };
    let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
    Score { true_positives: tp, false_positives: fp, false_negatives: fn_, precision, recall, f1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(frame: u64, channel: u32) -> Spike {
        Spike { frame, channel, amplitude: -1.0 }
    }

    #[test]
    fn matches_within_tolerance_once() {
        let truth = [s(100, 0), s(200, 0), s(100, 1)];
        let det = [s(103, 0), s(104, 0), s(300, 0), s(100, 1)];
        let r = score(&det, &truth, 5, &[]);
        assert_eq!((r.true_positives, r.false_positives, r.false_negatives), (2, 2, 1));
    }

    #[test]
    fn excluded_windows_are_ignored() {
        let r = score(&[s(50, 0)], &[s(10, 0)], 5, &[(0, 100)]);
        assert_eq!((r.true_positives, r.false_positives, r.false_negatives), (0, 0, 0));
    }
}
