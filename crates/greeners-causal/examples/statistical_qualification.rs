//! Bounded Monte Carlo diagnostics, not a general estimator qualification gate.
//!
//! Run from the workspace: cargo run --locked --offline -p greeners-causal
//! --example statistical_qualification. Four designs use 200 independent samples
//! of size 500, seed 20261005. DR intervals require both nuisance models correct;
//! the one-correct designs diagnose coverage outside that inference contract.

use greeners_causal::dr_learner::DRLearner;
use greeners_core::CovarianceType;
use greeners_ols::OLS;
use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::{json, Value};
use std::collections::BTreeMap;

const REPLICATIONS: usize = 200;
const SAMPLE_SIZE: usize = 500;
const SEED: u64 = 20261005;

fn run_design(design: &str, stream: u64) -> Value {
    let mut rng = StdRng::seed_from_u64(SEED + stream);
    let mut errors = Vec::new();
    let mut covered = 0;
    let mut failures = BTreeMap::<String, usize>::new();
    for _ in 0..REPLICATIONS {
        let mut y = Array1::zeros(SAMPLE_SIZE);
        let k = if design == "ols_hc1" { 2 } else { 1 };
        let mut x = Array2::zeros((SAMPLE_SIZE, k));
        let mut treatment = Vec::with_capacity(SAMPLE_SIZE);
        for i in 0..SAMPLE_SIZE {
            let noise: f64 = StandardNormal.sample(&mut rng);
            if design == "ols_hc1" {
                let xi: f64 = rng.gen_range(-1.0..1.0);
                x[(i, 0)] = 1.0;
                x[(i, 1)] = xi;
                y[i] = 0.5 + xi + (0.2 + 2.0 * xi.abs()) * noise;
            } else {
                let xi = rng.gen_range(0..3) as f64 - 1.0;
                x[(i, 0)] = xi;
                let propensity = if design == "dr_outcome_correct" {
                    if xi < 0.0 {
                        0.2
                    } else {
                        0.8
                    }
                } else {
                    0.5 + 0.2 * xi
                };
                let treated = rng.gen_bool(propensity);
                treatment.push(treated);
                let (mu0, tau) = if design == "dr_propensity_correct" {
                    (xi * xi, 1.0 + 0.5 * xi * xi)
                } else {
                    (0.4 + 0.5 * xi, 1.0 + 0.2 * xi)
                };
                y[i] = mu0 + if treated { tau } else { 0.0 } + noise;
            }
        }
        let target = if design == "dr_propensity_correct" {
            4.0 / 3.0
        } else {
            1.0
        };
        let estimate = if design == "ols_hc1" {
            OLS::fit(&y, &x, CovarianceType::HC1)
                .map(|r| (r.params[1], r.conf_lower[1], r.conf_upper[1]))
        } else {
            DRLearner::fit(&y, &treatment, &x, Some(3), None)
                .map(|r| (r.ate, r.ate_ci[0], r.ate_ci[1]))
        };
        match estimate {
            Ok((point, lower, upper)) if [point, lower, upper].iter().all(|v| v.is_finite()) => {
                errors.push(point - target);
                if lower <= target && target <= upper {
                    covered += 1;
                }
            }
            Ok(_) => {
                *failures
                    .entry("non-finite reported estimate or interval".into())
                    .or_default() += 1
            }
            Err(error) => *failures.entry(error.to_string()).or_default() += 1,
        }
    }
    let m = errors.len();
    if m == 0 {
        return json!({"design": design, "planned": REPLICATIONS, "successful": 0, "failures": failures});
    }
    let bias = errors.iter().sum::<f64>() / m as f64;
    let rmse = (errors.iter().map(|e| e * e).sum::<f64>() / m as f64).sqrt();
    let bias_mcse = if m > 1 {
        Some((errors.iter().map(|e| (e - bias).powi(2)).sum::<f64>() / ((m - 1) * m) as f64).sqrt())
    } else {
        None
    };
    let coverage = covered as f64 / m as f64;
    let planned_coverage = covered as f64 / REPLICATIONS as f64;
    json!({
        "design": design, "planned": REPLICATIONS, "successful": m,
        "failed": REPLICATIONS - m, "failure_reasons": failures,
        "sample_size": SAMPLE_SIZE, "seed": SEED + stream,
        "bias": bias, "bias_mcse": bias_mcse, "rmse": rmse,
        "coverage_successful": coverage,
        "coverage_successful_mcse": (coverage * (1.0 - coverage) / m as f64).sqrt(),
        "coverage_planned_denominator": planned_coverage,
        "coverage_planned_mcse": (planned_coverage * (1.0 - planned_coverage) / REPLICATIONS as f64).sqrt(),
        "inference_contract": if design == "ols_hc1" || design == "dr_both_correct" {
            "asymptotic inference control; bounded finite-sample diagnostic"
        } else {
            "point-estimate robustness branch; interval coverage diagnostic only"
        }
    })
}

fn main() {
    let designs = [
        "ols_hc1",
        "dr_outcome_correct",
        "dr_propensity_correct",
        "dr_both_correct",
    ];
    let results: Vec<_> = designs
        .iter()
        .enumerate()
        .map(|(i, name)| run_design(name, i as u64))
        .collect();
    println!(
        "{}",
        json!({"replications_per_design": REPLICATIONS, "results": results})
    );
}
