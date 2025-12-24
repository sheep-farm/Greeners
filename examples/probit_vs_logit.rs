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

    // Gerar dados baseados em um processo latente NORMAL (Probit é o modelo "verdadeiro")
    // y* = -1.0 + 2.0*x
    for _ in 0..n {
        let x = normal.sample(&mut rng);

        // Modelo Latente Probit (Erro Normal)
        let z = -1.0 + 2.0 * x + normal.sample(&mut rng);

        let y_val = if z > 0.0 { 1.0 } else { 0.0 };
        y_data.push(y_val);

        x_data.push(1.0); // Constante
        x_data.push(x); // X
    }

    let y = Array1::from(y_data);
    let x = Array2::from_shape_vec((n, 2), x_data)?;

    println!("--- Comparação Binária: Probit vs Logit ---\n");
    println!("Modelo Verdadeiro (Latente): Beta = 2.0 (Gerado com erro Normal)");

    // 1. Probit
    let res_probit = Probit::fit(&y, &x)?;
    println!("\n>>> 1. Resultados PROBIT");
    println!("Beta Estimado: {:.4}", res_probit.params[1]);
    println!("Log-Likelihood: {:.4}", res_probit.log_likelihood);

    // 2. Logit
    let res_logit = Logit::fit(&y, &x)?;
    println!("\n>>> 2. Resultados LOGIT");
    println!("Beta Estimado: {:.4}", res_logit.params[1]);
    println!("Log-Likelihood: {:.4}", res_logit.log_likelihood);

    // Comparação
    let ratio = res_logit.params[1] / res_probit.params[1];
    println!("\n--- Análise ---");
    println!("Razão Logit/Probit: {:.4} (Teoria diz ~1.6 a 1.8)", ratio);

    if res_probit.log_likelihood > res_logit.log_likelihood {
        println!("Melhor ajuste: PROBIT (Esperado, pois os dados são Normais).");
    } else {
        println!("Melhor ajuste: LOGIT.");
    }

    Ok(())
}
