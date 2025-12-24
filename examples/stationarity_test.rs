use greeners::TimeSeries;
use ndarray::Array1;
use rand::distributions::Distribution;
use statrs::distribution::Normal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 200;
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // 1. Simular Random Walk (Não-Estacionário)
    // y_t = y_{t-1} + e_t
    let mut rw_vec = vec![0.0; n];
    for t in 1..n {
        rw_vec[t] = rw_vec[t - 1] + normal.sample(&mut rng);
    }
    let rw_series = Array1::from(rw_vec);

    // 2. Simular Ruído Branco (Estacionário)
    let mut wn_vec = Vec::new();
    for _ in 0..n {
        wn_vec.push(normal.sample(&mut rng));
    }
    let wn_series = Array1::from(wn_vec);

    println!("--- Teste de Raiz Unitária (ADF) ---\n");

    // Teste 1: Random Walk
    let res_rw = TimeSeries::adf(&rw_series, None)?;
    println!("Série: Random Walk");
    println!("ADF Stat: {:.4}", res_rw.test_statistic);
    println!("Crítico 5%: {:.4}", res_rw.critical_values.1);
    println!("Estacionária? {}\n", res_rw.is_stationary);
    // Deve ser FALSE (Stat > Critico, ou seja, menos negativo)

    // Teste 2: White Noise
    let res_wn = TimeSeries::adf(&wn_series, None)?;
    println!("Série: Ruído Branco");
    println!("ADF Stat: {:.4}", res_wn.test_statistic);
    println!("Crítico 5%: {:.4}", res_wn.critical_values.1);
    println!("Estacionária? {}", res_wn.is_stationary);
    // Deve ser TRUE (Stat < Critico, bem negativo, ex: -7.0)

    Ok(())
}
