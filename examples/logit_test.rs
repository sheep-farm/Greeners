use greeners::Logit;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use statrs::distribution::Normal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1000;
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // 1. Configuração dos Vetores
    // x_data será um vetor plano que depois transformaremos em matriz (n x 3)
    let mut x_data = Vec::with_capacity(n * 3);
    let mut y_data = Vec::with_capacity(n);

    println!("--- Regressão Logística (Greeners MLE) ---\n");
    println!("Parâmetros Verdadeiros:");
    println!("Intercepto: -1.0");
    println!("Beta 1:      1.5");
    println!("Beta 2:      0.8\n");

    // 2. Loop de Geração de Dados
    for _ in 0..n {
        let x1 = normal.sample(&mut rng);
        let x2 = normal.sample(&mut rng);

        // Modelo Latente
        let z = -1.0 + 1.5 * x1 + 0.8 * x2;

        // Probabilidade (Sigmoide)
        let prob = 1.0 / (1.0 + (-z).exp());

        // CORREÇÃO: Usamos r#gen para escapar a palavra reservada 'gen' do Rust 2024
        let y_val = if rng.r#gen::<f64>() < prob { 1.0 } else { 0.0 };
        y_data.push(y_val);

        // Construção da linha da matriz X (Intercepto, x1, x2)
        x_data.push(1.0); // Constante
        x_data.push(x1);
        x_data.push(x2);
    }

    // 3. Conversão para Ndarray
    let y = Array1::from(y_data);
    // from_shape_vec pega o vetor plano e "dobra" no formato (n, 3)
    let x = Array2::from_shape_vec((n, 3), x_data)?;

    // 4. Estimar usando Greeners Logit
    let result = Logit::fit(&y, &x)?;

    println!("{}", result);

    Ok(())
}
