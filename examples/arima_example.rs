use greeners::ARIMA;
use ndarray::{Array1, Array2};

fn main() {
    // --- Example 1: Simple AR(1) ---
    println!("=== ARIMA(1,0,0) on AR(1) process ===\n");
    let n = 300;
    let mut y_vec = vec![0.0; n];
    let mut rng: u64 = 42;
    for t in 1..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let e = (rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        y_vec[t] = 0.1 + 0.7 * y_vec[t - 1] + e * 0.3;
    }
    let y = Array1::from_vec(y_vec);

    let result = ARIMA::fit(&y, (1, 0, 0)).unwrap();
    println!("{}", result);

    let forecast = result.predict(5, None).unwrap();
    println!("5-step forecast: {:?}\n", forecast);

    // --- Example 2: ARIMA(1,1,1) on integrated process ---
    println!("=== ARIMA(1,1,1) on random walk + drift ===\n");
    let mut rw = vec![0.0; n];
    rng = 77;
    for t in 1..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let e = (rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        let ar_part = if t > 1 {
            0.4 * (rw[t - 1] - rw[t - 2])
        } else {
            0.0
        };
        rw[t] = rw[t - 1] + 0.05 + ar_part + e * 0.2;
    }
    let y_rw = Array1::from_vec(rw);
    let result2 = ARIMA::fit(&y_rw, (1, 1, 1)).unwrap();
    println!("{}", result2);

    let forecast2 = result2.predict(5, None).unwrap();
    println!("5-step forecast: {:?}\n", forecast2);

    // --- Example 3: SARIMAX with seasonal component ---
    println!("=== SARIMAX(1,0,0)(1,0,0,12) ===\n");
    let n = 360;
    let mut seasonal = vec![0.0; n];
    rng = 999;
    for t in 12..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let e = (rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        seasonal[t] = 0.3 * seasonal[t - 1] + 0.5 * seasonal[t - 12] + e * 0.2;
    }
    let y_seasonal = Array1::from_vec(seasonal);
    let result3 = ARIMA::fit_sarimax(&y_seasonal, (1, 0, 0), (1, 0, 0, 12), None).unwrap();
    println!("{}", result3);

    // --- Example 4: ARIMAX with exogenous regressor ---
    println!("=== ARIMAX(1,0,0) with exogenous variable ===\n");
    let n = 200;
    let mut y4 = vec![0.0; n];
    let mut x4 = vec![0.0; n];
    rng = 55;
    for t in 0..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let e = (rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        x4[t] = e;
        y4[t] = 1.5 * x4[t] + if t > 0 { 0.4 * y4[t - 1] } else { 0.0 } + e * 0.1;
    }
    let y_exog = Array1::from_vec(y4);
    let exog = Array2::from_shape_vec((n, 1), x4).unwrap();
    let result4 = ARIMA::fit_sarimax(&y_exog, (1, 0, 0), (0, 0, 0, 1), Some(&exog)).unwrap();
    println!("{}", result4);
}
