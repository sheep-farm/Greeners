use greeners::VECM;
use ndarray::Array2;
use ndarray_rand::rand_distr::Normal;
use rand::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t_obs = 500;
    let mut rng = rand::thread_rng();
    let norm = Normal::new(0.0, 1.0).unwrap();

    let mut data = Array2::<f64>::zeros((t_obs, 2));

    // Simular Cointegração
    // Common Trend (W_t) -> Random Walk
    let mut w = 0.0;

    for t in 0..t_obs {
        w += norm.sample(&mut rng); // Passeio aleatório

        let noise_y = norm.sample(&mut rng) * 0.5;
        let noise_x = norm.sample(&mut rng) * 0.5;

        // X = W + noise
        data[[t, 0]] = w + noise_x;

        // Y = 2*W + noise (Y é o dobro de X no longo prazo)
        // Logo Y - 2X ~ 0
        data[[t, 1]] = 2.0 * w + noise_y;
    }

    println!("--- VECM Simulation (Cointegration) ---");
    println!("System: Y and X are random walks, but tied together.");
    println!("Relationship: Y = 2*X => Y - 2*X = 0");
    println!("Expected Cointegration Vector (Beta): Proportional to [1, -0.5] or [-2, 1]\n");

    // Estimar VECM com Rank=1
    let model = VECM::fit(&data, 2, 1)?; // 2 lags no nível = 1 lag na diferença
    println!("{}", model);

    // Normalizar Beta para facilitar leitura
    // Dividir tudo pelo primeiro elemento de Beta
    let beta_0 = model.beta[[0, 0]];
    let beta_1 = model.beta[[1, 0]];

    println!("Normalized Beta (Div by first element):");
    println!("1.0000");
    println!("{:.4}", beta_1 / beta_0);

    println!("\nConclusion:");
    println!("If the second value is approx -0.5 (if X is var1) or -2.0 (if Y is var1), cointegration was found.");

    Ok(())
}
