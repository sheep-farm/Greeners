use greeners::{CovarianceType, IV, OLS};
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use statrs::distribution::Normal; // Import Rng and Distribution for .sample() to work

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1000;
    let mut rng = rand::thread_rng();

    // 1. Configure Normal Distribution (Mean 0, Std 1)
    let normal = Normal::new(0.0, 1.0).unwrap();

    // 2. Generate Data
    let mut z_vec = Vec::with_capacity(n);
    let mut u_vec = Vec::with_capacity(n);
    let mut v_vec = Vec::with_capacity(n);

    for _ in 0..n {
        // We use the Distribution trait from rand to sample
        z_vec.push(normal.sample(&mut rng));
        u_vec.push(normal.sample(&mut rng));
        v_vec.push(normal.sample(&mut rng));
    }

    let z_array = Array1::from(z_vec);
    let u_array = Array1::from(u_vec);
    let v_array = Array1::from(v_vec);

    // 3. Create Endogeneity
    // X depends on Z (relevance) AND on u (endogeneity)
    // Real coefficient of Z in X = 0.8
    let x_array = 1.0 + &z_array * 0.8 + (&u_array * 0.5 + &v_array * 0.5);

    // 4. Generate Y (True Model)
    // Real coefficient of X in Y = 3.0
    // Note: 'u_array' appears here and in the X equation above -> this causes the bias!
    let y_array = 2.0 + &x_array * 3.0 + &u_array;

    // --- PREPARE MATRICES ---

    // Matrix X (Regressors): [1, x]
    let ones = Array2::ones((n, 1));
    // insert_axis(Axis(1)) transforms Array1 (vector) into Array2 (column) for concatenation
    let x_col = x_array.view().insert_axis(Axis(1));
    let x_mat = ndarray::concatenate(Axis(1), &[ones.view(), x_col])?;

    // Matrix Z (Instruments): [1, z]
    // The intercept is instrument of itself. Z replaces X.
    let z_col = z_array.view().insert_axis(Axis(1));
    let z_mat = ndarray::concatenate(Axis(1), &[ones.view(), z_col])?;

    // --- ESTIMATION ---

    println!("True Beta of X: 3.0");

    // 1. Run OLS (Should be BIASED)
    let ols_res = OLS::fit(&y_array, &x_mat, CovarianceType::HC1)?;
    println!("\n--- OLS (Biased) ---");
    // params[1] is the coefficient of x (params[0] is the intercept)
    println!(
        "Estimated Beta of X: {:.4} (Should be 3.0)",
        ols_res.params[1]
    );
    println!("OLS overestimates because X and error are positively correlated.");

    // 2. Run IV (Should be CONSISTENT)
    let iv_res = IV::fit(&y_array, &x_mat, &z_mat, CovarianceType::HC1)?;
    println!("\n--- IV / 2SLS (Consistent) ---");
    println!(
        "Estimated Beta of X: {:.4} (Recovered the true value!)",
        iv_res.params[1]
    );

    println!("{}", iv_res);

    Ok(())
}
