use greeners_core::linalg::LinalgInverse as _;
use greeners_panel::panel::FixedEffects;
use indexmap::IndexMap;
use ndarray::{Array1, Array2};

fn approx_zero(v: f64, tol: f64) {
    assert!(v.abs() < tol, "expected ~0, got {}", v);
}

/// Hand-crafted panel:
///   y_it = a_i + 2 * x_it
/// Entity 1 (a=3): x=[1,2] -> y=[5,7]
/// Entity 2 (a=5): x=[3,4] -> y=[11,13]
/// Within slope = 2, df = n - k - (N - 1) = 4 - 1 - 1 = 2.
fn exact_panel() -> (Array1<f64>, Array2<f64>, Vec<usize>) {
    let y = Array1::from(vec![5.0, 7.0, 11.0, 13.0]);
    let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let groups = vec![1, 1, 2, 2];
    (y, x, groups)
}

/// Helper: within-transform a matrix by subtracting group means.
fn manual_within_transform(data: &Array2<f64>, groups: &[usize]) -> Array2<f64> {
    let mut out = data.clone();
    let mut group_sums: IndexMap<usize, Array1<f64>> = IndexMap::new();
    let mut group_counts: IndexMap<usize, usize> = IndexMap::new();

    for (i, &g) in groups.iter().enumerate() {
        let row = data.row(i).to_owned();
        group_sums
            .entry(g)
            .and_modify(|s| *s = &*s + &row)
            .or_insert(row);
        *group_counts.entry(g).or_insert(0) += 1;
    }

    for (i, &g) in groups.iter().enumerate() {
        let mean = &group_sums[&g] / group_counts[&g] as f64;
        out.row_mut(i).assign(&(&data.row(i) - &mean));
    }

    out
}

#[test]
fn test_fe_within_transform_zero_group_means() {
    let (y, x, groups) = exact_panel();
    let y_mat = y.view().insert_axis(ndarray::Axis(1)).to_owned();

    let y_dm = manual_within_transform(&y_mat, &groups);
    let x_dm = manual_within_transform(&x, &groups);

    // Group means of demeaned y and demeaned x must be zero for every group.
    let mut y_sums: IndexMap<usize, (f64, usize)> = IndexMap::new();
    let mut x_sums: IndexMap<usize, (f64, usize)> = IndexMap::new();

    for (i, &g) in groups.iter().enumerate() {
        y_sums
            .entry(g)
            .and_modify(|(s, c)| {
                *s += y_dm[[i, 0]];
                *c += 1;
            })
            .or_insert((y_dm[[i, 0]], 1));
        x_sums
            .entry(g)
            .and_modify(|(s, c)| {
                *s += x_dm[[i, 0]];
                *c += 1;
            })
            .or_insert((x_dm[[i, 0]], 1));
    }
    for (_g, (sum, count)) in y_sums.iter() {
        approx_zero(sum / (*count as f64), 1e-14);
    }
    for (_g, (sum, count)) in x_sums.iter() {
        approx_zero(sum / (*count as f64), 1e-14);
    }
}

#[test]
fn test_fe_beta_equals_ols_on_demeaned_data() {
    let (y, x, groups) = exact_panel();
    let y_mat = y.view().insert_axis(ndarray::Axis(1)).to_owned();

    let y_dm_mat = manual_within_transform(&y_mat, &groups);
    let x_dm = manual_within_transform(&x, &groups);
    let y_dm = y_dm_mat.column(0).to_owned();

    // OLS of y_dm on x_dm (no intercept)
    let xtx = x_dm.t().dot(&x_dm);
    let xty = x_dm.t().dot(&y_dm);
    let xtx_inv = xtx.inv().unwrap();
    let expected_beta = xtx_inv.dot(&xty);

    let result = FixedEffects::fit(&y, &x, &groups).unwrap();
    for i in 0..result.params.len() {
        approx_zero((result.params[i] - expected_beta[i]).abs(), 1e-12);
    }
}

#[test]
fn test_fe_within_residuals_sum_to_zero_by_group() {
    let (y, x, groups) = exact_panel();
    let y_mat = y.view().insert_axis(ndarray::Axis(1)).to_owned();
    let result = FixedEffects::fit(&y, &x, &groups).unwrap();

    // Within residuals: (y - y_bar) - (X - X_bar) β
    let y_dm = manual_within_transform(&y_mat, &groups)
        .column(0)
        .to_owned();
    let x_dm = manual_within_transform(&x, &groups);
    let within_residuals = &y_dm - &x_dm.dot(&result.params);

    // Within residuals must have zero mean within each entity.
    let mut group_sums: IndexMap<usize, (f64, usize)> = IndexMap::new();
    for (i, &g) in groups.iter().enumerate() {
        group_sums
            .entry(g)
            .and_modify(|(s, c)| {
                *s += within_residuals[i];
                *c += 1;
            })
            .or_insert((within_residuals[i], 1));
    }
    for (_g, (sum, count)) in group_sums.iter() {
        approx_zero(sum / (*count as f64), 1e-12);
    }
}

#[test]
fn test_fe_exact_small_panel() {
    let (y, x, groups) = exact_panel();
    let result = FixedEffects::fit(&y, &x, &groups).unwrap();

    assert_eq!(result.params.len(), 1);
    assert!(
        (result.params[0] - 2.0).abs() < 1e-12,
        "slope: {}",
        result.params[0]
    );
    assert_eq!(result.n_entities, 2);
    assert_eq!(result.df_resid, 2);
}

#[test]
fn test_fe_time_invariant_variable_drops() {
    // Entity 1: z = 10, Entity 2: z = 20. After demeaning z becomes all zeros,
    // so the within estimator cannot identify its coefficient.
    let y = Array1::from(vec![5.0, 6.0, 15.0, 16.0]); // y = a_i + small_trend
    let mut x = Array2::zeros((4, 1));
    for (i, &g) in [1, 1, 2, 2].iter().enumerate() {
        x[[i, 0]] = if g == 1 { 10.0 } else { 20.0 };
    }
    let groups = vec![1, 1, 2, 2];

    let result = FixedEffects::fit(&y, &x, &groups);
    assert!(result.is_err());
}
