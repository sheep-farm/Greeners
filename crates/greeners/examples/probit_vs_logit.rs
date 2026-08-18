use greeners::{Logit, Probit};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use statrs::distribution::Normal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1000;
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut x_data = Vec::with_capacity(n * 2);
    let mut y_data = Vec::with_capacity(n);

    // Generate data based on a NORMAL latent process (Probit is the "true" model)
    // y* = -1.0 + 2.0*x
    for _ in 0..n {
        let x = normal.sample(&mut rng);

        // Latent Probit Model (Normal Error)
        let z = -1.0 + 2.0 * x + normal.sample(&mut rng);

        let y_val = if z > 0.0 { 1.0 } else { 0.0 };
        y_data.push(y_val);

        x_data.push(1.0); // Constant
        x_data.push(x); // X
    }

    let y = Array1::from(y_data);
    let x = Array2::from_shape_vec((n, 2), x_data)?;

    println!("--- Binary Comparison: Probit vs Logit ---\n");
    println!("True Model (Latent): Beta = 2.0 (Generated with Normal error)");

    // 1. Probit
    let res_probit = Probit::fit(&y, &x)?;
    println!("\n>>> 1. PROBIT Results");
    println!("Estimated Beta: {:.4}", res_probit.params[1]);
    println!("Log-Likelihood: {:.4}", res_probit.log_likelihood);

    // 2. Logit
    let res_logit = Logit::fit(&y, &x)?;
    println!("\n>>> 2. LOGIT Results");
    println!("Estimated Beta: {:.4}", res_logit.params[1]);
    println!("Log-Likelihood: {:.4}", res_logit.log_likelihood);

    // Comparison
    let ratio = res_logit.params[1] / res_probit.params[1];
    println!("\n--- Analysis ---");
    println!("Logit/Probit Ratio: {:.4} (Theory says ~1.6 to 1.8)", ratio);

    if res_probit.log_likelihood > res_logit.log_likelihood {
        println!("Better fit: PROBIT (Expected, since the data is Normal).");
    } else {
        println!("Better fit: LOGIT.");
    }

    Ok(())
}
