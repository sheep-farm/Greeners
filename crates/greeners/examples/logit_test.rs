use greeners::Logit;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use statrs::distribution::Normal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1000;
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // 1. Vector Configuration
    // x_data will be a flat vector that we will later transform into a matrix (n x 3)
    let mut x_data = Vec::with_capacity(n * 3);
    let mut y_data = Vec::with_capacity(n);

    println!("--- Logistic Regression (Greeners MLE) ---\n");
    println!("True Parameters:");
    println!("Intercept: -1.0");
    println!("Beta 1:      1.5");
    println!("Beta 2:      0.8\n");

    // 2. Data Generation Loop
    for _ in 0..n {
        let x1 = normal.sample(&mut rng);
        let x2 = normal.sample(&mut rng);

        // Latent Model
        let z = -1.0 + 1.5 * x1 + 0.8 * x2;

        // Probability (Sigmoid)
        let prob = 1.0 / (1.0 + (-z).exp());

        // FIX: We use r#gen to escape the reserved word 'gen' from Rust 2024
        let y_val = if rng.r#gen::<f64>() < prob { 1.0 } else { 0.0 };
        y_data.push(y_val);

        // Construction of matrix X row (Intercept, x1, x2)
        x_data.push(1.0); // Constant
        x_data.push(x1);
        x_data.push(x2);
    }

    // 3. Conversion to Ndarray
    let y = Array1::from(y_data);
    // from_shape_vec takes the flat vector and "folds" it into format (n, 3)
    let x = Array2::from_shape_vec((n, 3), x_data)?;

    // 4. Estimate using Greeners Logit
    let result = Logit::fit(&y, &x)?;

    println!("{}", result);

    Ok(())
}
