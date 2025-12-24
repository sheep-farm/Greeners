use greeners::{CovarianceType, Diagnostics, OLS};
use ndarray::{Array1, Array2, Axis};
// use rand::prelude::*;
use rand::distributions::Distribution;
use statrs::distribution::Normal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 200;
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // 1. Gerar Erros Autocorrelacionados (AR(1))
    // u_t = 0.7 * u_{t-1} + noise
    let mut u_vec = vec![0.0; n];
    let mut noise_vec = Vec::new();
    for _ in 0..n {
        noise_vec.push(normal.sample(&mut rng));
    }

    u_vec[0] = noise_vec[0];
    for t in 1..n {
        u_vec[t] = 0.7 * u_vec[t - 1] + noise_vec[t];
    }
    let u = Array1::from(u_vec);

    // 2. Gerar X (Aleatório) e Y
    // Modelo Verdadeiro: y = 1.0 + 2.0*x + u
    let mut x_vec = Vec::new();
    for _ in 0..n {
        x_vec.push(normal.sample(&mut rng) + 2.0);
    } // X média 2
    let x_arr = Array1::from(x_vec);

    let y = 1.0 + &x_arr * 2.0 + &u;

    // Preparar Matriz X
    let ones = Array2::ones((n, 1));
    let x_col = x_arr.view().insert_axis(Axis(1));
    let x_mat = ndarray::concatenate(Axis(1), &[ones.view(), x_col])?;

    println!("--- Análise de Série Temporal (Autocorrelação) ---\n");

    // 3. OLS "Ingênuo"
    let ols_res = OLS::fit(&y, &x_mat, CovarianceType::NonRobust)?;
    println!("1. OLS (Non-Robust)");
    println!("   Beta (x): {:.4}", ols_res.params[1]);
    println!(
        "   Std Err:  {:.4} (Provavelmente subestimado)",
        ols_res.std_errors[1]
    );

    // 4. Diagnóstico Durbin-Watson
    // Recalcular resíduos (poderíamos pegar do OLS se expuséssemos, mas vamos recalcular)
    let predicted = x_mat.dot(&ols_res.params);
    let residuals = &y - &predicted;

    let dw = Diagnostics::durbin_watson(&residuals);
    println!("\n2. Diagnóstico Durbin-Watson");
    println!("   DW Stat: {:.4}", dw);
    if dw < 1.5 {
        println!("   -> Alerta! DW baixo indica forte Autocorrelação Positiva.");
    }

    // 5. OLS com Newey-West (HAC)reset
    // Lags = 4 (regra de bolso para n=200, aprox n^0.25)
    let nw_res = OLS::fit(&y, &x_mat, CovarianceType::NeweyWest(4))?;
    println!("\n3. Newey-West (HAC, Lags=4)");
    println!("   Beta (x): {:.4} (Igual ao OLS)", nw_res.params[1]);
    println!(
        "   Std Err:  {:.4} (Corrigido para autocorrelação)",
        nw_res.std_errors[1]
    );

    let ratio = nw_res.std_errors[1] / ols_res.std_errors[1];
    println!(
        "\nCorreção: O Erro Padrão aumentou em {:.1}%",
        (ratio - 1.0) * 100.0
    );

    Ok(())
}
