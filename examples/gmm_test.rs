use greeners::GMM;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use statrs::distribution::Normal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1000;
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // 1. Generate Instruments (Exogenous)
    let mut z1_vec = Vec::new();
    let mut z2_vec = Vec::new();
    let mut u_vec = Vec::new(); // Structural error
    let mut v_vec = Vec::new(); // Reduced form error

    for _ in 0..n {
        z1_vec.push(normal.sample(&mut rng));
        z2_vec.push(normal.sample(&mut rng));
        u_vec.push(normal.sample(&mut rng));
        v_vec.push(normal.sample(&mut rng));
    }

    // Converting to ndarray vectors to facilitate arithmetic
    let z1 = Array1::from(z1_vec.clone());
    let z2 = Array1::from(z2_vec.clone());
    let u = Array1::from(u_vec);
    let v = Array1::from(v_vec);

    // 2. Generate Endogenous X
    // X depends on Z1 and Z2 (Relevant) and on u (Endogenous)
    // Real coefficient: 0.5 for each Z
    let x_endog = &z1 * 0.5 + &z2 * 0.5 + &u * 0.5 + &v * 0.5;

    // 3. Generate Y
    // y = 2.0 + 1.5*x + u
    let y = 2.0 + &x_endog * 1.5 + &u;

    // --- Assemble Matrices ---

    // Matrix X (Regressors): [1, x]
    let mut x_mat = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        x_mat[[i, 0]] = 1.0; // Intercept
        x_mat[[i, 1]] = x_endog[i]; // Endogenous X
    }

    // Matrix Z (Instruments): [1, z1, z2]
    // The constant is instrument of itself.
    let mut z_mat = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        z_mat[[i, 0]] = 1.0;
        z_mat[[i, 1]] = z1_vec[i];
        z_mat[[i, 2]] = z2_vec[i];
    }

    println!("--- GMM Estimation (Two-Step Efficient) ---\n");
    println!("True Beta (Slope): 1.5");
    println!("Instruments Z: [1, z1, z2] (Over-identified, L=3, K=2)");

    // Run GMM
    let gmm_res = GMM::fit(&y, &x_mat, &z_mat)?;

    println!("{}", gmm_res);

    println!("Analysis:");
    println!(
        "1. Estimated Coefficient: {:.4} (Expected: 1.5)",
        gmm_res.params[1]
    );
    println!("2. J-Statistic P-value:  {:.4}", gmm_res.j_p_value);

    if gmm_res.j_p_value > 0.05 {
        println!("   -> High P-value (> 0.05). We do not reject H0.");
        println!("   -> Conclusion: Instruments are valid (orthogonal to error).");
    } else {
        println!("   -> Low P-value. We reject H0.");
        println!("   -> Conclusion: Invalid instruments or misspecified model.");
    }

    Ok(())
}
