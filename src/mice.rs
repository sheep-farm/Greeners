//! MICE: Multiple Imputation by Chained Equations
//! (van Buuren & Groothuis-Oudshoorn 2011).
//!
//! Iterative imputation for multivariate missing data. Each
//! variable with missing values is imputed using a regression
//! on all other variables, iterated until convergence.
//!
//! Algorithm:
//!   1. Initialize missing values with mean imputation
//!   2. For each variable j with missing values:
//!      a. Regress y_j on all other variables (observed rows only)
//!      b. Predict missing values + add stochastic noise
//!   3. Repeat for M iterations (default 10)
//!   4. Repeat for D imputations (default 5)
//!   5. Pool results using Rubin's rules
//!
//! This implementation handles continuous variables. Returns
//! imputed datasets and pooled statistics.

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// Result of MICE imputation.
#[derive(Debug)]
pub struct MiceResult {
    /// Pooled imputed data (n x k), with missing values replaced
    pub imputed_data: Array2<f64>,
    /// Number of imputations
    pub n_imputations: usize,
    /// Number of iterations per imputation
    pub n_iterations: usize,
    /// Number of missing values imputed
    pub n_missing: usize,
    /// Number of observations
    pub n_obs: usize,
    /// Number of variables
    pub n_vars: usize,
    /// Variable names
    pub variable_names: Vec<String>,
    /// Missing count per variable
    pub missing_per_var: Vec<usize>,
    /// Pooled mean per variable
    pub pooled_mean: Array1<f64>,
    /// Pooled variance per variable (Rubin's rules)
    pub pooled_variance: Array1<f64>,
    /// Within-imputation variance
    pub within_variance: Array1<f64>,
    /// Between-imputation variance
    pub between_variance: Array1<f64>,
    /// Rate of missing information
    pub missing_info_rate: Array1<f64>,
}

impl fmt::Display for MiceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " MICE ")?;
        writeln!(f, "van Buuren & Groothuis-Oudshoorn (2011)")?;
        writeln!(f, "Multiple Imputation by Chained Equations")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Variables:", self.n_vars)?;
        writeln!(f, "{:<20} {:>12}", "Imputations:", self.n_imputations)?;
        writeln!(f, "{:<20} {:>12}", "Iterations:", self.n_iterations)?;
        writeln!(f, "{:<20} {:>12}", "Missing values:", self.n_missing)?;

        // Per-variable summary
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "  {:<10} {:>8} {:>12} {:>12} {:>12}",
            "Variable", "Missing", "Mean", "Variance", "FMI"
        )?;
        writeln!(f, "{:-^78}", "")?;
        for (j, name) in self.variable_names.iter().enumerate() {
            writeln!(
                f,
                "  {:<10} {:>8} {:>12.4} {:>12.4} {:>12.4}",
                name,
                self.missing_per_var[j],
                self.pooled_mean[j],
                self.pooled_variance[j],
                self.missing_info_rate[j]
            )?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct MiceChained;

impl MiceChained {
    /// Perform MICE imputation.
    ///
    /// # Arguments
    /// * `data` - Data matrix (n x k) with NaN for missing values
    /// * `n_imputations` - Number of imputations (default 5)
    /// * `n_iterations` - Iterations per imputation (default 10)
    /// * `variable_names` - Optional variable names
    pub fn fit(
        data: &Array2<f64>,
        n_imputations: Option<usize>,
        n_iterations: Option<usize>,
        variable_names: Option<Vec<String>>,
    ) -> Result<MiceResult, GreenersError> {
        let n = data.nrows();
        let k = data.ncols();
        if n < 5 || k < 2 {
            return Err(GreenersError::InvalidOperation(
                "MICE: need at least 5 observations and 2 variables".into(),
            ));
        }

        let names = variable_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());
        let m = n_imputations.unwrap_or(5);
        let max_iter = n_iterations.unwrap_or(10);

        // Identify missing values
        let mut missing_mask = Array2::from_elem((n, k), false);
        let mut missing_per_var = vec![0_usize; k];
        let mut col_means = vec![0.0_f64; k];
        let mut col_counts = vec![0_usize; k];

        for i in 0..n {
            for j in 0..k {
                if data[(i, j)].is_nan() {
                    missing_mask[(i, j)] = true;
                    missing_per_var[j] += 1;
                } else {
                    col_means[j] += data[(i, j)];
                    col_counts[j] += 1;
                }
            }
        }

        let n_missing: usize = missing_per_var.iter().sum();
        if n_missing == 0 {
            return Err(GreenersError::InvalidOperation(
                "MICE: no missing values found".into(),
            ));
        }

        for j in 0..k {
            if col_counts[j] > 0 {
                col_means[j] /= col_counts[j] as f64;
            }
        }

        // Check if any variable is fully missing
        for j in 0..k {
            if missing_per_var[j] == n {
                return Err(GreenersError::InvalidOperation(format!(
                    "MICE: variable '{}' is fully missing",
                    names[j]
                )));
            }
        }

        // Generate M imputations
        let mut all_imputations: Vec<Array2<f64>> = Vec::with_capacity(m);

        for imp in 0..m {
            // Initialize with mean imputation + noise
            let mut imputed = data.clone();
            for i in 0..n {
                for j in 0..k {
                    if missing_mask[(i, j)] {
                        // Add stochastic noise for different imputations
                        let noise = Self::rand_normal() * col_means[j].abs().max(1.0) * 0.01;
                        imputed[(i, j)] = col_means[j] + noise * (imp as f64 + 1.0) / m as f64;
                    }
                }
            }

            // Chained equations iterations
            for _iter in 0..max_iter {
                for j in 0..k {
                    if missing_per_var[j] == 0 {
                        continue;
                    }

                    // Regress variable j on all other variables
                    // using observed rows of j
                    let n_obs_j = col_counts[j];
                    if n_obs_j < k {
                        continue;
                    }

                    // Build X (other variables) and y (variable j) for observed rows
                    let mut x_obs = Array2::zeros((n_obs_j, k)); // intercept + k-1 others
                    let mut y_obs = Array1::zeros(n_obs_j);
                    let mut row = 0;
                    for i in 0..n {
                        if !missing_mask[(i, j)] {
                            x_obs[(row, 0)] = 1.0;
                            let mut col = 1;
                            for jj in 0..k {
                                if jj != j {
                                    x_obs[(row, col)] = imputed[(i, jj)];
                                    col += 1;
                                }
                            }
                            y_obs[row] = imputed[(i, j)];
                            row += 1;
                        }
                    }

                    // OLS
                    let xt = x_obs.t();
                    let xtx = xt.dot(&x_obs);
                    let xtx_inv = match (&xtx + Array2::<f64>::eye(k) * 1e-6).inv() {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let xty = xt.dot(&y_obs);
                    let beta = xtx_inv.dot(&xty);

                    // Predict missing values
                    let residuals = &y_obs - x_obs.dot(&beta);
                    let sigma2 = residuals.mapv(|r| r * r).sum() / (n_obs_j - k) as f64;
                    let sigma = sigma2.sqrt().max(1e-10);

                    for i in 0..n {
                        if missing_mask[(i, j)] {
                            let mut x_i = Array1::zeros(k);
                            x_i[0] = 1.0;
                            let mut col = 1;
                            for jj in 0..k {
                                if jj != j {
                                    x_i[col] = imputed[(i, jj)];
                                    col += 1;
                                }
                            }
                            let pred = beta.dot(&x_i);
                            // Add stochastic noise
                            let noise = Self::rand_normal() * sigma;
                            imputed[(i, j)] = pred + noise;
                        }
                    }
                }
            }

            all_imputations.push(imputed);
        }

        // Pool results using Rubin's rules
        let mut pooled_mean = Array1::zeros(k);
        let mut within_variance = Array1::zeros(k);
        let mut between_variance = Array1::zeros(k);
        let mut pooled_variance = Array1::zeros(k);
        let mut missing_info_rate = Array1::zeros(k);

        for j in 0..k {
            // Pooled mean: average of imputation means
            let imp_means: Vec<f64> = all_imputations
                .iter()
                .map(|imp| imp.column(j).mean().unwrap_or(0.0))
                .collect();
            pooled_mean[j] = imp_means.iter().sum::<f64>() / m as f64;

            // Within-imputation variance: average of imputation variances
            let imp_vars: Vec<f64> = all_imputations
                .iter()
                .map(|imp| {
                    let col = imp.column(j);
                    let mean = col.mean().unwrap_or(0.0);
                    col.mapv(|v| (v - mean).powi(2)).sum() / (n - 1) as f64
                })
                .collect();
            within_variance[j] = imp_vars.iter().sum::<f64>() / m as f64;

            // Between-imputation variance
            let between: f64 = imp_means
                .iter()
                .map(|&mu| (mu - pooled_mean[j]).powi(2))
                .sum::<f64>()
                / (m - 1) as f64;
            between_variance[j] = between;

            // Pooled variance (Rubin's rules)
            // T = W_bar + (1 + 1/m) * B
            pooled_variance[j] = within_variance[j] + (1.0 + 1.0 / m as f64) * between;

            // Fraction of missing information
            // FMI = (1 + 1/m) * B / T
            missing_info_rate[j] = if pooled_variance[j] > 1e-15 {
                (1.0 + 1.0 / m as f64) * between / pooled_variance[j]
            } else {
                0.0
            };
        }

        // Use first imputation as the imputed dataset
        let imputed_data = all_imputations[0].clone();

        Ok(MiceResult {
            imputed_data,
            n_imputations: m,
            n_iterations: max_iter,
            n_missing,
            n_obs: n,
            n_vars: k,
            variable_names: names,
            missing_per_var,
            pooled_mean,
            pooled_variance,
            within_variance,
            between_variance,
            missing_info_rate,
        })
    }

    fn rand_normal() -> f64 {
        // Box-Muller transform
        let u1 = Self::rand_uniform().max(1e-10);
        let u2 = Self::rand_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn rand_uniform() -> f64 {
        use std::cell::Cell;
        thread_local! {
            static STATE: Cell<u64> = const { Cell::new(2468135790) };
        }
        STATE.with(|s| {
            let mut state = s.get();
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s.set(state);
            ((state >> 11) as f64) / (1u64 << 53) as f64
        })
    }
}
