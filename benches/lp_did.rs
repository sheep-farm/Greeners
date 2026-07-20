use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use greeners::{DataFrame, LpDid};
use indexmap::IndexMap;
use ndarray::Array1;
use ndarray_rand::rand::{distributions::Distribution, rngs::StdRng, SeedableRng};
use ndarray_rand::rand_distr::Normal;

fn make_panel(n_units: usize, n_periods: usize, seed: u64) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(seed);
    let norm = Normal::new(0.0, 1.0).unwrap();
    let cohort_choices = [0.0, 6.0, 9.0, 12.0];

    let mut id = Vec::new();
    let mut t = Vec::new();
    let mut g = Vec::new();
    let mut y = Vec::new();

    for i in 0..n_units {
        let cohort = cohort_choices[i % cohort_choices.len()];
        let ufe = norm.sample(&mut rng);
        for tt in 1..=n_periods {
            let treated = if cohort > 0.0 && (tt as f64) >= cohort {
                1.0
            } else {
                0.0
            };
            let value = ufe + 0.3 * (tt as f64) + 2.0 * treated + norm.sample(&mut rng);
            id.push(i as f64);
            t.push(tt as f64);
            g.push(cohort);
            y.push(value);
        }
    }

    let mut cols = IndexMap::new();
    cols.insert("id".to_string(), Array1::from(id));
    cols.insert("t".to_string(), Array1::from(t));
    cols.insert("g".to_string(), Array1::from(g));
    cols.insert("y".to_string(), Array1::from(y));
    DataFrame::new(cols).unwrap()
}

fn bench_lp_did(c: &mut Criterion) {
    let mut group = c.benchmark_group("lpdid");
    for n_units in [200, 500] {
        let df = make_panel(n_units, 15, 0);
        group.bench_with_input(BenchmarkId::new("vw", n_units), &n_units, |b, _| {
            b.iter(|| {
                let _ = black_box(
                    LpDid::new()
                        .with_target_estimand("vw")
                        .with_max_pre(Some(5))
                        .with_max_post(Some(8))
                        .with_base_period_int(-1)
                        .with_clean_control("not_yet_treated")
                        .fit(&df, "y", "id", "t", Some("g"), None, None),
                );
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_lp_did);
criterion_main!(benches);
