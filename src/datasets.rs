use crate::error::GreenersError;
use crate::DataFrame;
use ndarray::Array1;
use indexmap::IndexMap;

/// Built-in datasets for examples and testing.
pub struct Datasets;

impl Datasets {
    /// Longley dataset (macroeconomic, n=16, 7 variables).
    /// Classic dataset for testing multicollinearity.
    pub fn longley() -> Result<DataFrame, GreenersError> {
        let gnp_deflator = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];
        let gnp = vec![
            234289.0, 259426.0, 258054.0, 284599.0, 328975.0, 346999.0, 365385.0, 363112.0,
            397469.0, 419180.0, 442769.0, 444546.0, 482704.0, 502601.0, 518173.0, 554894.0,
        ];
        let unemployed = vec![
            235.6, 232.5, 368.2, 335.1, 209.9, 193.2, 187.0, 357.8, 290.4, 282.2, 293.6, 468.1,
            381.3, 393.1, 480.6, 400.7,
        ];
        let armed_forces = vec![
            159.0, 145.6, 161.6, 165.0, 309.9, 359.4, 354.7, 335.0, 304.8, 285.7, 279.8, 263.7,
            255.2, 251.4, 257.2, 282.7,
        ];
        let population = vec![
            107.608, 108.632, 109.773, 110.929, 112.075, 113.270, 115.094, 116.219, 117.388,
            118.734, 120.445, 121.950, 123.366, 125.368, 127.852, 130.081,
        ];
        let year = vec![
            1947.0, 1948.0, 1949.0, 1950.0, 1951.0, 1952.0, 1953.0, 1954.0, 1955.0, 1956.0, 1957.0,
            1958.0, 1959.0, 1960.0, 1961.0, 1962.0,
        ];
        let employed = vec![
            60.323, 61.122, 60.171, 61.187, 63.221, 63.639, 64.989, 63.761, 66.019, 67.857, 68.169,
            66.513, 68.655, 69.564, 69.331, 70.551,
        ];

        let mut data = IndexMap::new();
        data.insert("gnp_deflator".to_string(), Array1::from(gnp_deflator));
        data.insert("gnp".to_string(), Array1::from(gnp));
        data.insert("unemployed".to_string(), Array1::from(unemployed));
        data.insert("armed_forces".to_string(), Array1::from(armed_forces));
        data.insert("population".to_string(), Array1::from(population));
        data.insert("year".to_string(), Array1::from(year));
        data.insert("employed".to_string(), Array1::from(employed));

        DataFrame::new(data)
    }

    /// Stackloss dataset (n=21, 4 variables).
    /// Industrial chemical process data.
    pub fn stackloss() -> Result<DataFrame, GreenersError> {
        let air_flow = vec![
            80.0, 80.0, 75.0, 62.0, 62.0, 62.0, 62.0, 62.0, 58.0, 58.0, 58.0, 58.0, 58.0, 58.0,
            50.0, 50.0, 50.0, 50.0, 50.0, 56.0, 70.0,
        ];
        let water_temp = vec![
            27.0, 27.0, 25.0, 24.0, 22.0, 23.0, 24.0, 24.0, 23.0, 18.0, 18.0, 17.0, 18.0, 19.0,
            18.0, 18.0, 19.0, 19.0, 20.0, 20.0, 20.0,
        ];
        let acid_conc = vec![
            89.0, 88.0, 90.0, 87.0, 87.0, 87.0, 93.0, 93.0, 87.0, 80.0, 89.0, 88.0, 82.0, 93.0,
            89.0, 86.0, 72.0, 79.0, 80.0, 82.0, 91.0,
        ];
        let stackloss = vec![
            42.0, 37.0, 37.0, 28.0, 18.0, 18.0, 19.0, 20.0, 15.0, 14.0, 14.0, 13.0, 11.0, 12.0,
            8.0, 7.0, 8.0, 8.0, 9.0, 15.0, 15.0,
        ];

        let mut data = IndexMap::new();
        data.insert("air_flow".to_string(), Array1::from(air_flow));
        data.insert("water_temp".to_string(), Array1::from(water_temp));
        data.insert("acid_conc".to_string(), Array1::from(acid_conc));
        data.insert("stackloss".to_string(), Array1::from(stackloss));

        DataFrame::new(data)
    }

    /// Simple simulated data for quick testing.
    /// Linear model: y = 1 + 2*x1 + 0.5*x2 + noise.
    pub fn simulated_linear(n: usize, seed: u64) -> Result<DataFrame, GreenersError> {
        // Simple LCG for reproducibility without extra deps
        let mut rng_state = seed;
        let mut next_rand = || -> f64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Map to [-1, 1] range roughly
            ((rng_state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };

        let mut x1_vec = Vec::with_capacity(n);
        let mut x2_vec = Vec::with_capacity(n);
        let mut y_vec = Vec::with_capacity(n);

        for _ in 0..n {
            let x1 = next_rand() * 5.0;
            let x2 = next_rand() * 3.0;
            let noise = next_rand() * 0.5;
            let y = 1.0 + 2.0 * x1 + 0.5 * x2 + noise;
            x1_vec.push(x1);
            x2_vec.push(x2);
            y_vec.push(y);
        }

        let mut data = IndexMap::new();
        data.insert("y".to_string(), Array1::from(y_vec));
        data.insert("x1".to_string(), Array1::from(x1_vec));
        data.insert("x2".to_string(), Array1::from(x2_vec));

        DataFrame::new(data)
    }

    /// Simulated panel data (balanced).
    /// y_it = alpha_i + beta * x_it + epsilon_it
    pub fn simulated_panel(
        n_entities: usize,
        n_periods: usize,
        seed: u64,
    ) -> Result<DataFrame, GreenersError> {
        let n = n_entities * n_periods;
        let mut rng_state = seed;
        let mut next_rand = || -> f64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };

        let mut entity_vec = Vec::with_capacity(n);
        let mut time_vec = Vec::with_capacity(n);
        let mut x_vec = Vec::with_capacity(n);
        let mut y_vec = Vec::with_capacity(n);

        for i in 0..n_entities {
            let alpha = next_rand() * 2.0;
            for t in 0..n_periods {
                let x = next_rand() * 5.0;
                let eps = next_rand() * 0.5;
                let y = alpha + 1.5 * x + eps;

                entity_vec.push(i as f64);
                time_vec.push(t as f64);
                x_vec.push(x);
                y_vec.push(y);
            }
        }

        let mut data = IndexMap::new();
        data.insert("y".to_string(), Array1::from(y_vec));
        data.insert("x".to_string(), Array1::from(x_vec));
        data.insert("entity".to_string(), Array1::from(entity_vec));
        data.insert("time".to_string(), Array1::from(time_vec));

        DataFrame::new(data)
    }
}
