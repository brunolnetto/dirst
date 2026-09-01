use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dirst::geodesic::{GeodesicCurve, Surface, PARALLEL_ENABLED};

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    let d = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + i as f64 * d).collect()
}

fn mode_label() -> &'static str {
    if PARALLEL_ENABLED {
        "parallel"
    } else {
        "serial"
    }
}

fn geodesic_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("geodesic");

    let sphere = GeodesicCurve::new(Surface::sphere(1.0));
    let paraboloid = GeodesicCurve::new(Surface::paraboloid());
    let t_eval = linspace(0.0, 1.0, 201);
    let y0 = [0.0, 0.0, 1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()];

    group.bench_with_input(
        BenchmarkId::new("christoffel_symbols_at", mode_label()),
        &sphere,
        |b, gc| {
            b.iter(|| {
                let gamma = gc.christoffel_symbols_at(black_box(&[0.7, 1.1]));
                black_box(gamma);
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("solve_ivp", mode_label()),
        &paraboloid,
        |b, gc| {
            b.iter(|| {
                let sol = gc.solve_ivp(black_box(&t_eval), black_box(&y0));
                black_box(sol.t.len());
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("curve_length", mode_label()),
        &paraboloid,
        |b, gc| {
            let sol = gc.solve_ivp(&t_eval, &y0);
            b.iter(|| {
                let len = gc.curve_length(black_box(&sol));
                black_box(len);
            })
        },
    );

    group.finish();
}

criterion_group!(benches, geodesic_benches);
criterion_main!(benches);
