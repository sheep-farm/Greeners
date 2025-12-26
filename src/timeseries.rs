use crate::{CovarianceType, GreenersError, OLS};
use ndarray::{s, Array1, Array2}; // Axis removed as it was not used

/// Results of the Augmented Dickey-Fuller Test
#[derive(Debug)]
pub struct AdfResult {
    pub test_statistic: f64,
    pub p_value: Option<f64>,
    pub critical_values: (f64, f64, f64), // 1%, 5%, 10%
    pub is_stationary: bool,
    pub lags_used: usize,
    pub n_obs: usize,
}

pub struct TimeSeries;

impl TimeSeries {
    pub fn adf(series: &Array1<f64>, max_lags: Option<usize>) -> Result<AdfResult, GreenersError> {
        let n = series.len();
        if n < 10 {
            return Err(GreenersError::ShapeMismatch(
                "Series too short for ADF".into(),
            ));
        }

        // 1. Define Lags (Rule of thumb: (n-1)^(1/3))
        // FIX: Using native Rust .powf() instead of libm
        let lags = match max_lags {
            Some(l) => l,
            None => ((n - 1) as f64).powf(1.0 / 3.0) as usize,
        };

        // 2. Prepare Variables
        let y_diff = diff(series, 1);
        let effective_n = n - 1 - lags;

        let target_y = y_diff.slice(s![lags..]).to_owned();

        let mut x_mat = Array2::<f64>::zeros((effective_n, 2 + lags));

        // Coluna 0: Constante
        x_mat.column_mut(0).fill(1.0);

        // Column 1: y_{t-1} (Lagged level)
        for i in 0..effective_n {
            x_mat[[i, 1]] = series[lags + i];
        }

        // Column 2..L: Lags of Differences
        for l in 0..lags {
            for i in 0..effective_n {
                x_mat[[i, 2 + l]] = y_diff[lags + i - 1 - l];
            }
        }

        // 3. Run OLS
        let ols_res = OLS::fit(&target_y, &x_mat, CovarianceType::NonRobust)?;

        // 4. ADF Statistic = t-stat of level coefficient (index 1)
        let adf_stat = ols_res.t_values[1];

        // 5. Critical Values (MacKinnon approx. for constant)
        let crit_1pct = -3.43;
        let crit_5pct = -2.86;
        let crit_10pct = -2.57;

        Ok(AdfResult {
            test_statistic: adf_stat,
            p_value: None,
            critical_values: (crit_1pct, crit_5pct, crit_10pct),
            // If Statistic < Critical (ex: -4.0 < -2.86), reject H0 (Stationary)
            is_stationary: adf_stat < crit_5pct,
            lags_used: lags,
            n_obs: effective_n,
        })
    }
}

fn diff(arr: &Array1<f64>, lag: usize) -> Array1<f64> {
    let len = arr.len();
    let mut out = Vec::with_capacity(len - lag);
    for i in lag..len {
        out.push(arr[i] - arr[i - lag]);
    }
    Array1::from(out)
}
