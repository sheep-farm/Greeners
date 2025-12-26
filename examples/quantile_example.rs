use greeners::QuantileReg;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use statrs::distribution::Normal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 500;
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Gerar Dados Heterocedásticos
    // y = 10 + 2*x + (1 + x)*erro
    // O erro aumenta conforme X aumenta!
    let mut x_vec = Vec::new();
    let mut y_vec = Vec::new();
    let mut x_mat_vec = Vec::new();

    for _ in 0..n {
        let x_val = rng.gen_range(1.0..10.0); // X entre 1 e 10
        let error = normal.sample(&mut rng);
        
        // A variância do erro depende de X (Heterocedasticidade)
        let y_val = 10.0 + 2.0 * x_val + (1.0 + 0.5 * x_val) * error;

        x_vec.push(x_val);
        y_vec.push(y_val);

        x_mat_vec.push(1.0); // Intercepto
        x_mat_vec.push(x_val);
    }

    let y = Array1::from(y_vec);
    let x = Array2::from_shape_vec((n, 2), x_mat_vec)?;

    println!("--- Regressão Quantílica (Curva de Engel Simulada) ---");
    println!("Modelo Verdadeiro: y = 10 + 2x + Heterocedasticidade\n");

    // 1. Quantil 0.10 (Os "Poupadores" / Limite Inferior)
    // A inclinação deve ser MENOR que 2.0, pois o erro negativo cresce com X
    let q10 = QuantileReg::fit(&y, &x, 0.10, 100)?; // 100 bootstraps para rapidez
    println!("{}", q10);

    // 2. Quantil 0.50 (Mediana)
    // A inclinação deve ser próxima de 2.0 (igual ao OLS)
    let q50 = QuantileReg::fit(&y, &x, 0.50, 100)?;
    println!("{}", q50);

    // 3. Quantil 0.90 (Os "Gastadores" / Limite Superior)
    // A inclinação deve ser MAIOR que 2.0, pois o erro positivo cresce com X
    let q90 = QuantileReg::fit(&y, &x, 0.90, 100)?;
    println!("{}", q90);

    println!("Análise:");
    println!("Inclinação q10: {:.4}", q10.params[1]);
    println!("Inclinação q50: {:.4}", q50.params[1]);
    println!("Inclinação q90: {:.4}", q90.params[1]);
    println!("Diferença (q90 - q10) captura a heterocedasticidade!");

    Ok(())
}