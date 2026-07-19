use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use indexmap::IndexMap;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand::{distributions::Distribution, rngs::StdRng, SeedableRng};
use ndarray_rand::rand_distr::Normal;

use greeners::linalg::{LinalgCholesky, LinalgInverse, UPLO};
use greeners::{CovarianceType, DataFrame, Formula, Logit, Probit, QuantileReg, IV, OLS, VAR};

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

fn norm() -> Normal<f64> {
    Normal::new(0.0, 1.0).unwrap()
}

fn generate_ols_data(n: usize, k: usize) -> (Array1<f64>, Array2<f64>) {
    let mut rng = rng();
    let norm = norm();
    let x = Array2::from_shape_fn(
        (n, k + 1),
        |(_, j)| {
            if j == 0 {
                1.0
            } else {
                norm.sample(&mut rng)
            }
        },
    );
    let noise = Array1::from_shape_fn(n, |_| norm.sample(&mut rng));
    let y = x.slice(ndarray::s![.., 1..]).sum_axis(Axis(1)) + &noise;
    (y, x)
}

fn generate_binary_data(n: usize, k: usize) -> (Array1<f64>, Array2<f64>) {
    let (y_cont, x) = generate_ols_data(n, k);
    let y = y_cont.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
    (y, x)
}

fn generate_var_data(n: usize) -> Array2<f64> {
    let mut rng = rng();
    let norm = norm();
    Array2::from_shape_fn((n, 2), |_| norm.sample(&mut rng))
}

fn generate_spd_matrix(k: usize) -> Array2<f64> {
    let mut rng = rng();
    let norm = norm();
    // Build a positive definite matrix as X'X from an overdetermined random X.
    let x = Array2::from_shape_fn((2 * k.max(1), k), |_| norm.sample(&mut rng));
    x.t().dot(&x)
}

fn make_dataframe(y: &Array1<f64>, x: &Array2<f64>) -> DataFrame {
    let n = y.len();
    let mut columns: IndexMap<String, Array1<f64>> = IndexMap::new();
    columns.insert("y".to_string(), y.clone());
    for j in 0..x.ncols() {
        if j == 0 {
            continue; // intercept is implicit in formula, skip as named column
        }
        let col = x.slice(ndarray::s![.., j]).to_owned();
        columns.insert(format!("x{j}"), col);
    }
    // Add enough rows guarantee: y and all x columns have length n
    if n == 0 {
        // Defensive; benchmarks never use n=0.
    }
    DataFrame::new(columns).unwrap()
}

fn write_csv(n: usize) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("greeners_bench_csv_{n}.csv"));
    if path.exists() {
        return path;
    }

    let mut rng = rng();
    let norm = norm();
    let file = File::create(&path).expect("create csv");
    let mut w = BufWriter::new(file);
    writeln!(w, "y,x1,x2,x3").unwrap();
    for _ in 0..n {
        let x1 = norm.sample(&mut rng);
        let x2 = norm.sample(&mut rng);
        let x3 = norm.sample(&mut rng);
        let y = x1 + x2 + x3 + norm.sample(&mut rng);
        writeln!(w, "{y},{x1},{x2},{x3}").unwrap();
    }
    w.flush().unwrap();
    path
}

fn bench_ols(c: &mut Criterion) {
    let mut group = c.benchmark_group("ols");
    for n in [100, 1_000, 10_000, 100_000] {
        let (y, x) = generate_ols_data(n, 3);
        group.bench_with_input(BenchmarkId::new("fit", n), &n, |b, _| {
            b.iter(|| OLS::fit(black_box(&y), black_box(&x), CovarianceType::NonRobust).unwrap())
        });
    }
    group.finish();
}

fn bench_design_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("design_matrix");
    for n in [100, 1_000, 10_000, 100_000] {
        let (y, x) = generate_ols_data(n, 3);
        let df = make_dataframe(&y, &x);
        let formula = Formula::parse("y ~ x1 + x2 + x3").unwrap();
        group.bench_with_input(BenchmarkId::new("to_design_matrix", n), &n, |b, _| {
            b.iter(|| df.to_design_matrix(black_box(&formula)).unwrap())
        });
    }
    group.finish();
}

fn bench_ols_formula(c: &mut Criterion) {
    let mut group = c.benchmark_group("ols_formula");
    for n in [100, 1_000, 10_000, 100_000] {
        let (y, x) = generate_ols_data(n, 3);
        let df = make_dataframe(&y, &x);
        let formula = Formula::parse("y ~ x1 + x2 + x3").unwrap();
        group.bench_with_input(BenchmarkId::new("from_formula", n), &n, |b, _| {
            b.iter(|| {
                let (y, x) = df.to_design_matrix(black_box(&formula)).unwrap();
                OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap()
            })
        });
    }
    group.finish();
}

fn bench_iv(c: &mut Criterion) {
    let mut group = c.benchmark_group("iv");
    for n in [1_000, 10_000] {
        let (y, x) = generate_ols_data(n, 1);
        // Build a proper instrument: z = x1 + noise, so first-stage is strong.
        let x1 = x.slice(ndarray::s![.., 1]).to_owned();
        let z = &x1 + Array1::from_shape_fn(n, |_| norm().sample(&mut rng()));

        let mut columns: IndexMap<String, Array1<f64>> = IndexMap::new();
        columns.insert("y".to_string(), y.clone());
        columns.insert("x1".to_string(), x1);
        columns.insert("z1".to_string(), z);
        let df = DataFrame::new(columns).unwrap();

        let endog = Formula::parse("y ~ x1").unwrap();
        let instr = Formula {
            dependent: "y".to_string(),
            independents: vec!["z1".to_string()],
            intercept: true,
        };

        group.bench_with_input(BenchmarkId::new("from_formula", n), &n, |b, _| {
            b.iter(|| {
                IV::from_formula(
                    black_box(&endog),
                    black_box(&instr),
                    black_box(&df),
                    CovarianceType::NonRobust,
                )
                .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_quantile(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantile");
    for n in [1_000, 10_000] {
        let (y, x) = generate_ols_data(n, 3);
        let df = make_dataframe(&y, &x);
        let formula = Formula::parse("y ~ x1 + x2 + x3").unwrap();
        group.bench_with_input(BenchmarkId::new("fit_boot0", n), &n, |b, _| {
            b.iter(|| {
                QuantileReg::from_formula(black_box(&formula), black_box(&df), 0.5, 0).unwrap()
            })
        });
    }
    group.finish();
}

fn bench_var(c: &mut Criterion) {
    let mut group = c.benchmark_group("var");
    for n in [1_000, 10_000] {
        let data = generate_var_data(n);
        group.bench_with_input(BenchmarkId::new("fit", n), &n, |b, _| {
            b.iter(|| VAR::fit(black_box(&data), 1, None).unwrap())
        });
    }
    group.finish();
}

fn bench_discrete(c: &mut Criterion) {
    let mut group = c.benchmark_group("discrete");
    for n in [1_000, 10_000] {
        let (y, x) = generate_binary_data(n, 3);
        group.bench_with_input(BenchmarkId::new("logit", n), &n, |b, _| {
            b.iter(|| Logit::fit(black_box(&y), black_box(&x)).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("probit", n), &n, |b, _| {
            b.iter(|| Probit::fit(black_box(&y), black_box(&x)).unwrap())
        });
    }
    group.finish();
}

fn bench_csv_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("csv_parse");
    for n in [1_000, 10_000, 100_000] {
        let path = write_csv(n);
        group.bench_with_input(BenchmarkId::new("from_csv", n), &n, |b, _| {
            b.iter(|| DataFrame::from_csv(black_box(&path)).unwrap())
        });
    }
    group.finish();
}

fn bench_matrix_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_inverse");
    for k in [10, 50, 100, 200] {
        let a = generate_spd_matrix(k);
        group.bench_with_input(BenchmarkId::new("inv", k), &k, |b, _| {
            b.iter(|| black_box(&a).inv().unwrap())
        });
    }
    group.finish();
}

fn bench_cholesky(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky");
    for k in [10, 50, 100, 200, 500] {
        let a = generate_spd_matrix(k);
        group.bench_with_input(BenchmarkId::new("llt", k), &k, |b, _| {
            b.iter(|| black_box(&a).cholesky(UPLO::Lower).unwrap())
        });
    }
    group.finish();
}

fn bench_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_normal");
    let norm = norm();
    for n in [100_000, 1_000_000] {
        group.bench_with_input(BenchmarkId::new("array1", n), &n, |b, &n| {
            b.iter(|| {
                let mut rng = rng();
                black_box(Array1::<f64>::from_shape_fn(n, |_| norm.sample(&mut rng)))
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_ols,
    bench_design_matrix,
    bench_ols_formula,
    bench_iv,
    bench_quantile,
    bench_var,
    bench_discrete,
    bench_csv_parse,
    bench_matrix_inverse,
    bench_cholesky,
    bench_random,
);
criterion_main!(benches);
