use greeners::Tobit;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

fn lcg_sequence(n: usize, seed: u64) -> Vec<f64> {
    let mut current = seed;
    let mut res = Vec::with_capacity(n);
    for _ in 0..n {
        current = (current.wrapping_mul(6364136223846793005).wrapping_add(1)) % 4294967296;
        res.push(current as f64 / 4294967296.0);
    }
    res
}

#[test]
fn test_tobit_simulated() {
    let n = 300;
    let u1 = lcg_sequence(n, 42);
    let u2 = lcg_sequence(n, 123);
    let x_val = lcg_sequence(n, 777);

    // Box-Muller for normal errors: mean = 0, std = 1.5
    let sigma_true = 1.5;
    let mut err = vec![0.0; n];
    for i in 0..n {
        let r1 = u1[i].max(1e-15);
        let r2 = u2[i];
        let r = (-2.0 * r1.ln()).sqrt();
        let theta = 2.0 * PI * r2;
        let z0 = r * theta.cos();
        err[i] = z0 * sigma_true;
    }

    // Tobit equation: y_star = -0.5 + 2.0 * x_i + err_i
    // Censored at ll = 0.0
    let ll = 0.0;
    let mut y = Array1::<f64>::zeros(n);
    let mut x = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        x[[i, 0]] = 1.0; // intercept
        x[[i, 1]] = x_val[i];
        let y_star = -0.5 + 2.0 * x_val[i] + err[i];
        y[i] = if y_star > ll { y_star } else { ll };
    }

    let res = Tobit::fit(&y, &x, ll, None).unwrap();

    assert!(res.n_censored > 0);
    assert!(res.params.len() == 2);
    assert!(res.sigma > 0.0);
    assert!(res.std_errors[0] > 0.0);
    assert!(res.std_errors[1] > 0.0);
    assert!(res.log_likelihood.is_finite());

    // Also run a test with extreme values to verify numerical robustness (underflow avoidance)
    // Create an observation with huge x_i such that xb - ll / sigma > 40.0
    // This previously caused underflow in standard norm_cdf(-a) and phi(a) / norm_cdf(-a).
    let mut y_ext = Array1::<f64>::zeros(10);
    let mut x_ext = Array2::<f64>::zeros((10, 2));
    for i in 0..10 {
        x_ext[[i, 0]] = 1.0;
        x_ext[[i, 1]] = -50.0; // large negative value -> extremely high probability of censoring!
        y_ext[i] = 0.0; // censored
    }

    // Combine them with the original sample
    let mut y_comb = Array1::<f64>::zeros(n + 10);
    let mut x_comb = Array2::<f64>::zeros((n + 10, 2));
    y_comb.slice_mut(ndarray::s![..n]).assign(&y);
    y_comb.slice_mut(ndarray::s![n..]).assign(&y_ext);
    x_comb.slice_mut(ndarray::s![..n, ..]).assign(&x);
    x_comb.slice_mut(ndarray::s![n.., ..]).assign(&x_ext);

    // This should fit successfully without dividing by zero, optimization failure, or standard error NaNs!
    let res_comb = Tobit::fit(&y_comb, &x_comb, ll, None).unwrap();
    assert!(res_comb.n_censored > res.n_censored);
    assert!(res_comb.params[0].is_finite());
    assert!(res_comb.std_errors[0].is_finite());
}
