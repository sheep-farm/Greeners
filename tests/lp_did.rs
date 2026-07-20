// Test panel generator inspired by the `pylpdid` quickstart example
// (<https://github.com/Daniel-Uhr/pylpdid>), used under the MIT License.
// Original code copyright (c) 2026 Daniel de Abreu Pereira Uhr.

use greeners::{DataFrame, LpDid};
use ndarray_rand::rand::{rngs::StdRng, Rng, SeedableRng};

fn make_absorbing_panel(n_units: usize, n_periods: usize, seed: u64) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut id = Vec::new();
    let mut t = Vec::new();
    let mut g = Vec::new();
    let mut y = Vec::new();

    let cohorts = [0.0_f64, 5.0, 8.0, 11.0];
    for i in 0..n_units {
        let cohort = cohorts[rng.gen::<usize>() % cohorts.len()];
        let ufe = rng.gen::<f64>() * 2.0 - 1.0; // ~U(-1,1)
        for tt in 1..=n_periods {
            let d = if cohort > 0.0 && (tt as f64) >= cohort {
                1.0
            } else {
                0.0
            };
            let yy = ufe + 0.25 * (tt as f64) + 2.0 * d + (rng.gen::<f64>() - 0.5) * 0.8;
            id.push(i as f64);
            t.push(tt as f64);
            g.push(cohort);
            y.push(yy);
        }
    }

    DataFrame::builder()
        .add_column("id", id)
        .add_column("t", t)
        .add_column("g", g)
        .add_column("y", y)
        .build()
        .unwrap()
}

#[test]
fn lp_did_vw_recovers_constant_effect() {
    let df = make_absorbing_panel(150, 14, 0);
    let res = LpDid::new()
        .with_max_pre(Some(4))
        .with_max_post(Some(6))
        .fit(&df, "y", "id", "t", Some("g"), None, None)
        .unwrap();

    let horizons = &res.horizons;
    let estimates = res.estimates.as_slice().unwrap();
    for (&h, &est) in horizons.iter().zip(estimates.iter()) {
        if h == 0 || h == 4 {
            assert!(
                (est - 2.0).abs() < 0.25,
                "expected effect ~2.0 at h={}, got {}",
                h,
                est
            );
        }
    }
    assert!(res
        .scalar_estimates
        .as_slice()
        .unwrap()
        .iter()
        .any(|&x| (x - 2.0).abs() < 0.25));
}

#[test]
fn lp_did_rw_recovers_constant_effect() {
    let df = make_absorbing_panel(150, 14, 1);
    let res = LpDid::new()
        .with_target_estimand("rw")
        .with_max_pre(Some(4))
        .with_max_post(Some(6))
        .fit(&df, "y", "id", "t", Some("g"), None, None)
        .unwrap();

    let horizons = &res.horizons;
    let estimates = res.estimates.as_slice().unwrap();
    for (&h, &est) in horizons.iter().zip(estimates.iter()) {
        if h == 0 || h == 4 {
            assert!(
                (est - 2.0).abs() < 0.25,
                "expected effect ~2.0 at h={}, got {}",
                h,
                est
            );
        }
    }
}

#[test]
fn lp_did_ra_recovers_constant_effect() {
    let df = make_absorbing_panel(150, 14, 2);
    let res = LpDid::new()
        .with_target_estimand("ra")
        .with_max_pre(Some(4))
        .with_max_post(Some(6))
        .fit(&df, "y", "id", "t", Some("g"), None, None)
        .unwrap();

    let horizons = &res.horizons;
    let estimates = res.estimates.as_slice().unwrap();
    for (&h, &est) in horizons.iter().zip(estimates.iter()) {
        if h == 0 || h == 4 {
            assert!(
                (est - 2.0).abs() < 0.3,
                "expected effect ~2.0 at h={}, got {}",
                h,
                est
            );
        }
    }
}

#[test]
fn lp_did_base_period_normalization() {
    let df = make_absorbing_panel(150, 14, 3);
    let res = LpDid::new()
        .with_base_period_int(-1)
        .with_max_pre(Some(4))
        .with_max_post(Some(6))
        .fit(&df, "y", "id", "t", Some("g"), None, None)
        .unwrap();

    let horizons = &res.horizons;
    let estimates = res.estimates.as_slice().unwrap();
    for (&h, &est) in horizons.iter().zip(estimates.iter()) {
        if h == -1 {
            assert!(
                est.abs() < 1e-10,
                "base period estimate should be zero, got {}",
                est
            );
        }
    }
}

#[test]
fn lp_did_fixed_composition_runs() {
    let df = make_absorbing_panel(150, 14, 4);
    let res = LpDid::new()
        .with_fixed_composition(true)
        .with_max_pre(Some(3))
        .with_max_post(Some(5))
        .fit(&df, "y", "id", "t", Some("g"), None, None)
        .unwrap();

    assert!(res
        .estimates
        .as_slice()
        .unwrap()
        .iter()
        .any(|x| x.is_finite()));
}

#[test]
fn lp_did_pooled_scalar_present() {
    let df = make_absorbing_panel(150, 14, 5);
    let res = LpDid::new()
        .with_max_pre(Some(4))
        .with_max_post(Some(6))
        .fit(&df, "y", "id", "t", Some("g"), None, None)
        .unwrap();

    assert!(res.scalar_terms.contains(&"ATT pooled".to_string()));
    let idx = res
        .scalar_terms
        .iter()
        .position(|t| t == "ATT pooled")
        .unwrap();
    let pooled = res.scalar_estimates[idx];
    assert!(
        (pooled - 2.0).abs() < 0.25,
        "expected pooled ~2.0, got {}",
        pooled
    );
}

#[test]
fn lp_did_bootstrap_runs_and_produces_finite_se() {
    let df = make_absorbing_panel(80, 12, 3);
    let res = LpDid::new()
        .with_max_pre(Some(3))
        .with_max_post(Some(5))
        .with_inference("cluster_bootstrap")
        .with_n_bootstrap(50)
        .with_seed(42)
        .fit(&df, "y", "id", "t", Some("g"), None, None)
        .unwrap();

    for (&h, &se) in res.horizons.iter().zip(res.standard_errors.iter()) {
        if h != -1 {
            assert!(
                se.is_finite() && se >= 0.0,
                "expected finite SE at h={}, got {}",
                h,
                se
            );
        }
    }
    for se in res.scalar_standard_errors.iter() {
        assert!(
            se.is_finite() && *se > 0.0,
            "expected finite positive scalar SE, got {}",
            se
        );
    }
}
