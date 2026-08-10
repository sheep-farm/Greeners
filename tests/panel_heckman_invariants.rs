use greeners::PanelHeckman;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn panel_heckman_runs_and_recovers_parameters() {
    let n_panels = 30;
    let t = 10;
    let n = n_panels * t;
    let mut rng = StdRng::seed_from_u64(11);
    let noise = Normal::new(0.0, 0.5).unwrap();

    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n * 2);
    let mut x = Vec::with_capacity(n * 2);
    let mut z = Vec::with_capacity(n);
    let mut panel_ids = Vec::with_capacity(n);

    for p in 0..n_panels {
        let alpha = noise.sample(&mut rng);
        for _tt in 0..t {
            panel_ids.push(p as i64);

            let z1 = noise.sample(&mut rng);
            let nu = noise.sample(&mut rng);
            let selected = 0.3 + 0.2 * z1 + nu > 0.0;

            w.push(1.0);
            w.push(z1);

            let x1 = z1 + 0.5 * noise.sample(&mut rng);
            x.push(1.0);
            x.push(x1);

            if selected {
                let y_val = 1.0 + 1.5 * x1 + 0.3 * (z1 + nu) + alpha + noise.sample(&mut rng);
                y.push(y_val);
            } else {
                y.push(0.0);
            }
            z.push(selected);
        }
    }

    let y = Array1::from_vec(y);
    let w = Array2::from_shape_vec((n, 2), w).unwrap();
    let x = Array2::from_shape_vec((n, 2), x).unwrap();

    let result = PanelHeckman::fit(
        &z,
        &y,
        &w,
        &x,
        &panel_ids,
        Some(vec!["const".into(), "z1".into()]),
        Some(vec!["const".into(), "x1".into()]),
    )
    .unwrap();

    assert!(result.gamma.iter().all(|v| v.is_finite()));
    assert!(result.beta.iter().all(|v| v.is_finite()));
    assert_eq!(result.gamma.len(), 2);
    assert_eq!(result.beta.len(), 2);
    assert!(result.n_obs == n);
    assert!(result.n_selected >= 2);
    assert!(result.n_panels == n_panels);
    assert!(result.rho.abs() <= 0.99);
    assert!(result.sigma > 0.0);

    // The outcome slope on x1 should be positive and in the right ballpark.
    assert!(
        result.beta[1] > 0.5 && result.beta[1] < 2.5,
        "x1 coef out of range: {}",
        result.beta[1]
    );
}

#[test]
fn panel_heckman_input_validation() {
    let n = 10;
    let z = vec![true; n];
    let y = Array1::from_vec(vec![0.0; n]);
    let w = Array2::from_shape_vec((n, 2), vec![1.0; n * 2]).unwrap();
    let x = Array2::from_shape_vec((n, 2), vec![1.0; n * 2]).unwrap();
    let panel_ids = vec![0i64; n];

    // Mismatched dimensions
    let short_y = Array1::from_vec(vec![0.0; n - 1]);
    assert!(PanelHeckman::fit(&z, &short_y, &w, &x, &panel_ids, None, None).is_err());

    // No selected observations
    let z_none = vec![false; n];
    assert!(PanelHeckman::fit(&z_none, &y, &w, &x, &panel_ids, None, None).is_err());
}
