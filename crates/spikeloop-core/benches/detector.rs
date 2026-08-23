use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use spikeloop_core::{synthesize, Config, Detector, SynthConfig};

fn bench(c: &mut Criterion) {
    let rec = synthesize(&SynthConfig { duration_s: 2.0, ..Default::default() });
    let n = rec.n_channels;
    let mut group = c.benchmark_group("detector_64ch_25khz");
    group.throughput(Throughput::Elements(rec.data.len() as u64));
    for chunk_frames in [25usize, 250, 2500] {
        group.bench_function(format!("chunk_{chunk_frames}_frames"), |b| {
            b.iter(|| {
                let mut d = Detector::new(Config::new(n, rec.sample_rate)).unwrap();
                let mut out = Vec::with_capacity(4096);
                for chunk in rec.data.chunks(chunk_frames * n) {
                    d.process(black_box(chunk), &mut out).unwrap();
                }
                black_box(out.len())
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
