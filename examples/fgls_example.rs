use greeners::{CovarianceType, FGLS, OLS};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use statrs::distribution::Normal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 200;
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Gerar AR(1) Error Term
    // u_t = 0.8 * u_{t-1} + e_t
    let rho_real = 0.8;
    let mut u = vec![0.0; n];
    let mut e_vec = Vec::new();

    // Inicializar erro
    u[0] = normal.sample(&mut rng);

    for i in 1..n {
        let e = normal.sample(&mut rng);
        u[i] = rho_real * u[i - 1] + e;
        e_vec.push(e);
    }

    // Gerar Dados
    // y = 2.0 + 1.5 * x + u
    let mut x_vec = Vec::new();
    let mut y_vec = Vec::new();
    let mut x_flat = Vec::new();

    for i in 0..n {
        let x_val = i as f64 * 0.1 + normal.sample(&mut rng); // Trend + Noise
        let y_val = 2.0 + 1.5 * x_val + u[i];

        y_vec.push(y_val);
        x_vec.push(x_val);

        x_flat.push(1.0); // Constant
        x_flat.push(x_val);
    }

    let y = Array1::from(y_vec);
    let x = Array2::from_shape_vec((n, 2), x_flat)?;

    println!("--- Comparação OLS vs FGLS (Cochrane-Orcutt) ---");
    println!("Real Rho: 0.8");
    println!("Real Beta: 1.5\n");

    // 1. OLS Padrão
    let ols = OLS::fit(&y, &x, CovarianceType::NonRobust)?;
    println!(">>> OLS:");
    println!("Beta: {:.4}", ols.params[1]);
    println!(
        "StdErr: {:.4} (Provavelmente subestimado/viesado)",
        ols.std_errors[1]
    );

    // 2. FGLS
    let fgls = FGLS::cochrane_orcutt(&y, &x)?;
    println!("\n>>> FGLS (Cochrane-Orcutt):");
    println!("{}", fgls);

    println!("Análise:");
    println!("Rho Estimado: {:.4} (Esperado 0.8)", fgls.rho.unwrap());

    Ok(())
}
