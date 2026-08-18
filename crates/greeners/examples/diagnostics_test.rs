use greeners::{CovarianceType, Diagnostics, OLS};
use ndarray::{Array1, Array2, Axis};
use std::error::Error;
use std::fs::File;

fn main() -> Result<(), Box<dyn Error>> {
    // 1. Leitura rápida (reaproveitando lógica anterior)
    let file = File::open("dataset.csv")?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let mut y_vec = Vec::new();
    let mut x_flat = Vec::new();
    let mut n = 0;

    for result in rdr.records() {
        let record = result?;
        y_vec.push(record[0].parse::<f64>()?); // income
        x_flat.push(record[1].parse::<f64>()?); // education
        x_flat.push(record[2].parse::<f64>()?); // age
        x_flat.push(record[3].parse::<f64>()?); // experience
        n += 1;
    }

    let y = Array1::from(y_vec);
    let x_raw = Array2::from_shape_vec((n, 3), x_flat)?;
    let ones = Array2::ones((n, 1));
    let x = ndarray::concatenate(Axis(1), &[ones.view(), x_raw.view()])?;

    // 2. Rodar OLS
    println!("Rodando OLS...");
    let model = OLS::fit(&y, &x, CovarianceType::NonRobust)?; // Começamos com OLS Padrão

    // 3. Recuperar Resíduos
    // Precisamos recalcular os resíduos aqui pois não expusemos no struct de resultado (ainda)
    // u = y - X*beta
    let predicted = x.dot(&model.params);
    let residuals = &y - &predicted;

    println!("\n=== Testes de Diagnóstico ===");

    // A. Jarque-Bera (Normalidade)
    let (jb, jb_p) = Diagnostics::jarque_bera(&residuals)?;
    println!(
        "Jarque-Bera (Normalidade): Stat={:.4}, P-value={:.4}",
        jb, jb_p
    );
    if jb_p < 0.05 {
        println!("-> Rejeita H0: Resíduos NÃO são normais.");
    } else {
        println!("-> Falha em rejeitar H0: Resíduos parecem normais.");
    }

    // B. Breusch-Pagan (Heterocedasticidade)
    let (bp, bp_p) = Diagnostics::breusch_pagan(&residuals, &x)?;
    println!(
        "\nBreusch-Pagan (Heterocedasticidade): Stat={:.4}, P-value={:.4}",
        bp, bp_p
    );
    if bp_p < 0.05 {
        println!("-> Rejeita H0: Existe Heterocedasticidade!");
        println!("-> Recomendação: Use CovarianceType::HC1 (Robust Errors).");
    } else {
        println!("-> Falha em rejeitar H0: Variância parece constante (Homocedasticidade).");
    }

    Ok(())
}
