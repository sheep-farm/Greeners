use greeners::Heckman;
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
fn test_heckman_simulated() {
    let n = 500;
    let u1 = lcg_sequence(n, 42);
    let u2 = lcg_sequence(n, 123);
    let w_val = lcg_sequence(n, 999);
    let x_val = lcg_sequence(n, 777);

    // Box-Muller for normal errors
    let mut z0 = vec![0.0; n];
    let mut z1 = vec![0.0; n];
    for i in 0..n {
        let r1 = u1[i].max(1e-15);
        let r2 = u2[i];
        let r = (-2.0 * r1.ln()).sqrt();
        let theta = 2.0 * PI * r2;
        z0[i] = r * theta.cos();
        z1[i] = r * theta.sin();
    }

    // Correlation rho = 0.6
    let rho = 0.6;
    let mut err_u = vec![0.0; n]; // selection error
    let mut err_e = vec![0.0; n]; // outcome error
    for i in 0..n {
        err_u[i] = z0[i];
        err_e[i] = rho * z0[i] + (1.0 - rho * rho).sqrt() * z1[i];
    }

    // Selection equation: z_i* = -0.5 + 1.5 * w_i + err_u_i
    let mut z = Array1::<f64>::zeros(n);
    let mut x_sel = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        x_sel[[i, 0]] = 1.0; // intercept
        x_sel[[i, 1]] = w_val[i];
        let zstar = -0.5 + 1.5 * w_val[i] + err_u[i];
        z[i] = if zstar > 0.0 { 1.0 } else { 0.0 };
    }

    // Outcome equation: y_i = 1.0 + 2.0 * x_i + err_e_i
    let mut y = Array1::<f64>::zeros(n);
    let mut x_out = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        x_out[[i, 0]] = 1.0; // intercept
        x_out[[i, 1]] = x_val[i];
        let y_val_actual = 1.0 + 2.0 * x_val[i] + err_e[i];
        // If not selected, y_i is unobserved (we can store 0.0, as fit ignores it)
        y[i] = if z[i] == 1.0 { y_val_actual } else { 0.0 };
    }

    let res = Heckman::fit(&y, &x_out, &z, &x_sel, None, None).unwrap();

    assert!(res.n_selected > 0);
    assert!(res.params.len() == 2);
    assert!(res.rho.abs() <= 1.0);
    assert!(res.sigma > 0.0);
    assert!(res.std_errors[0] > 0.0);
    assert!(res.std_errors[1] > 0.0);
    assert!(res.select_params.len() == 2);
    assert!(res.select_se.len() == 2);

    // OLS on selected observations should be biased (since rho = 0.6)
    // Heckman should correct this bias. Let's make sure it estimates parameters successfully.
    println!("Heckman params: {:?}", res.params);
    println!("Heckman rho: {}", res.rho);
}
