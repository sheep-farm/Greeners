use greeners::{CovarianceType, SurEquation, OLS, SUR};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::thread_rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 500;
    let mut rng = thread_rng();

    // --- 1. Gerar Regressores DISTINTOS ---
    let x1_raw = Array1::from_iter((0..n).map(|_| Normal::new(2.0, 1.0).unwrap().sample(&mut rng)));
    let x2_raw = Array1::from_iter((0..n).map(|_| Normal::new(5.0, 2.0).unwrap().sample(&mut rng)));

    // --- 2. Gerar Erros Correlacionados ---
    let dist = Normal::new(0.0, 1.0).unwrap();
    let common = Array1::from_iter((0..n).map(|_| dist.sample(&mut rng)));
    let idio1 = Array1::from_iter((0..n).map(|_| dist.sample(&mut rng)));
    let idio2 = Array1::from_iter((0..n).map(|_| dist.sample(&mut rng)));

    let u1 = &common * 0.8 + &idio1 * 0.6;
    let u2 = &common * 0.8 + &idio2 * 0.6;

    // --- 3. Gerar Y ---
    // AQUI ESTAVA O ERRO: Adicionamos : Array1<f64>
    let y1: Array1<f64> = 10.0 + &x1_raw * 2.0 + &u1;
    let y2: Array1<f64> = 20.0 + &x2_raw * 0.5 + &u2;

    // --- 4. Montar Matrizes X ---
    let mut x1_mat_vec = Vec::new();
    for i in 0..n {
        x1_mat_vec.push(1.0);
        x1_mat_vec.push(x1_raw[i]);
    }
    let x1_mat = Array2::from_shape_vec((n, 2), x1_mat_vec)?;

    let mut x2_mat_vec = Vec::new();
    for i in 0..n {
        x2_mat_vec.push(1.0);
        x2_mat_vec.push(x2_raw[i]);
    }
    let x2_mat = Array2::from_shape_vec((n, 2), x2_mat_vec)?;

    // --- 5. Executar SUR ---
    let eq1 = SurEquation {
        name: "Empresa A".into(),
        y: y1.clone(),
        x: x1_mat.clone(),
    };
    let eq2 = SurEquation {
        name: "Empresa B".into(),
        y: y2.clone(),
        x: x2_mat.clone(),
    };

    println!("--- OLS Individual (Benchmark) ---");
    let ols1 = OLS::fit(&y1, &x1_mat, CovarianceType::NonRobust)?;
    let ols2 = OLS::fit(&y2, &x2_mat, CovarianceType::NonRobust)?;
    println!(
        "OLS Eq 1 Beta: {:.4} (StdErr: {:.4})",
        ols1.params[1], ols1.std_errors[1]
    );
    println!(
        "OLS Eq 2 Beta: {:.4} (StdErr: {:.4})",
        ols2.params[1], ols2.std_errors[1]
    );

    println!("\n--- SUR (Zellner) ---");
    let sur = SUR::fit(&[eq1, eq2])?;
    println!("{}", sur);

    // Comparação de Eficiência
    let eff_gain1 = 1.0 - (sur.equations[0].std_errors[1] / ols1.std_errors[1]);
    let eff_gain2 = 1.0 - (sur.equations[1].std_errors[1] / ols2.std_errors[1]);

    println!("Ganho de Eficiência (Redução no Erro Padrão):");
    println!("Eq 1: {:.2}%", eff_gain1 * 100.0);
    println!("Eq 2: {:.2}%", eff_gain2 * 100.0);

    Ok(())
}
