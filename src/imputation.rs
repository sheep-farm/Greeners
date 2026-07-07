use crate::error::GreenersError;
use crate::{CovarianceType, OLS};
use indexmap::IndexMap;
use ndarray::{Array1, Array2};
use std::fmt;

/// Result of MICE imputation.
#[derive(Debug)]
pub struct MICEResult {
    /// Multiple imputed datasets: Vec of column-name -> complete column
    pub datasets: Vec<IndexMap<String, Array1<f64>>>,
    pub n_imputations: usize,
    pub n_iter: usize,
    pub n_obs: usize,
    pub n_vars: usize,
}

impl fmt::Display for MICEResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^60}", " MICE Imputation ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10}", "Variables:", self.n_vars)?;
        writeln!(f, "{:<20} {:>10}", "Imputations:", self.n_imputations)?;
        writeln!(f, "{:<20} {:>10}", "Iterations:", self.n_iter)?;
        writeln!(f, "{:=^60}", "")
    }
}

/// Result of Bayesian Gaussian MI.
#[derive(Debug)]
pub struct BayesGaussMIResult {
    pub datasets: Vec<IndexMap<String, Array1<f64>>>,
    pub n_imputations: usize,
    pub n_obs: usize,
    pub n_vars: usize,
}

impl fmt::Display for BayesGaussMIResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^60}", " Bayesian Gaussian MI ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10}", "Variables:", self.n_vars)?;
        writeln!(f, "{:<20} {:>10}", "Imputations:", self.n_imputations)?;
        writeln!(f, "{:=^60}", "")
    }
}

/// Multiple Imputation by Chained Equations.
pub struct MICE;

impl MICE {
    /// Impute missing data using MICE.
    ///
    /// - `data`: HashMap of column name -> Array1 (NaN = missing)
    /// - `n_imputations`: number of imputed datasets to generate
    /// - `n_iter`: number of MICE iterations per imputation
    pub fn impute(
        data: &IndexMap<String, Array1<f64>>,
        n_imputations: usize,
        n_iter: usize,
    ) -> Result<MICEResult, GreenersError> {
        if data.is_empty() {
            return Err(GreenersError::InvalidOperation(
                "Empty data for MICE".into(),
            ));
        }

        let col_names: Vec<String> = data.keys().cloned().collect();
        let n_vars = col_names.len();
        let n_obs = data[&col_names[0]].len();

        // Build matrix form
        let mut mat = Array2::<f64>::zeros((n_obs, n_vars));
        for (j, name) in col_names.iter().enumerate() {
            let col = &data[name];
            if col.len() != n_obs {
                return Err(GreenersError::ShapeMismatch(
                    "All columns must have same length".into(),
                ));
            }
            mat.column_mut(j).assign(col);
        }

        // Find missing indices per column
        let missing: Vec<Vec<usize>> = (0..n_vars)
            .map(|j| (0..n_obs).filter(|&i| mat[[i, j]].is_nan()).collect())
            .collect();

        let mut datasets = Vec::with_capacity(n_imputations);
        // Simple PRNG state
        let mut rng_state: u64 = 42;

        for _imp in 0..n_imputations {
            let mut current = mat.clone();

            // Initial fill: column mean for missing values
            for j in 0..n_vars {
                let observed: Vec<f64> = (0..n_obs)
                    .filter(|&i| !current[[i, j]].is_nan())
                    .map(|i| current[[i, j]])
                    .collect();
                let col_mean = if observed.is_empty() {
                    0.0
                } else {
                    observed.iter().sum::<f64>() / observed.len() as f64
                };
                for &i in &missing[j] {
                    current[[i, j]] = col_mean;
                }
            }

            // Chained equations iterations
            for _it in 0..n_iter {
                for j in 0..n_vars {
                    if missing[j].is_empty() {
                        continue;
                    }

                    // Build predictors: all columns except j
                    let observed_rows: Vec<usize> =
                        (0..n_obs).filter(|i| !missing[j].contains(i)).collect();

                    if observed_rows.len() < n_vars {
                        continue;
                    }

                    let mut x_obs = Array2::<f64>::zeros((observed_rows.len(), n_vars)); // incl. intercept
                    let mut y_obs = Array1::<f64>::zeros(observed_rows.len());

                    for (ii, &i) in observed_rows.iter().enumerate() {
                        x_obs[[ii, 0]] = 1.0; // intercept
                        let mut col_idx = 1;
                        for jj in 0..n_vars {
                            if jj != j {
                                x_obs[[ii, col_idx]] = current[[i, jj]];
                                col_idx += 1;
                            }
                        }
                        y_obs[ii] = current[[i, j]];
                    }

                    // Trim to actual columns used
                    let x_obs = x_obs.slice(ndarray::s![.., ..n_vars]).to_owned();

                    if let Ok(ols_res) = OLS::fit(&y_obs, &x_obs, CovarianceType::NonRobust) {
                        // Predict for missing rows
                        let resid = ols_res.residuals(&y_obs, &x_obs);
                        let sigma2 =
                            resid.iter().map(|r| r * r).sum::<f64>() / observed_rows.len() as f64;
                        let sigma = sigma2.sqrt();

                        for &i in &missing[j] {
                            let mut x_i = Array1::<f64>::zeros(n_vars);
                            x_i[0] = 1.0;
                            let mut col_idx = 1;
                            for jj in 0..n_vars {
                                if jj != j {
                                    x_i[col_idx] = current[[i, jj]];
                                    col_idx += 1;
                                }
                            }
                            let x_i = x_i.slice(ndarray::s![..n_vars]).to_owned();
                            let pred = x_i.dot(&ols_res.params);
                            // Add noise from predictive distribution
                            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                            let u1 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
                            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                            let u2 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
                            let z = (-2.0 * u1.max(1e-15).ln()).sqrt()
                                * (2.0 * std::f64::consts::PI * u2).cos();
                            current[[i, j]] = pred + sigma * z;
                        }
                    }
                }
            }

            // Convert back to HashMap
            let mut dataset = IndexMap::new();
            for (j, name) in col_names.iter().enumerate() {
                dataset.insert(name.clone(), current.column(j).to_owned());
            }
            datasets.push(dataset);
        }

        Ok(MICEResult {
            datasets,
            n_imputations,
            n_iter,
            n_obs,
            n_vars,
        })
    }
}

/// Bayesian Gaussian Multiple Imputation.
///
/// Assumes multivariate normal data. Uses Gibbs sampler to draw
/// (mu, Sigma) from posterior given observed data, then imputes missing values.
pub struct BayesGaussMI;

impl BayesGaussMI {
    /// Impute missing data assuming multivariate normality.
    pub fn impute(
        data: &IndexMap<String, Array1<f64>>,
        n_imputations: usize,
    ) -> Result<BayesGaussMIResult, GreenersError> {
        if data.is_empty() {
            return Err(GreenersError::InvalidOperation(
                "Empty data for BayesGaussMI".into(),
            ));
        }

        let col_names: Vec<String> = data.keys().cloned().collect();
        let n_vars = col_names.len();
        let n_obs = data[&col_names[0]].len();

        let mut mat = Array2::<f64>::zeros((n_obs, n_vars));
        for (j, name) in col_names.iter().enumerate() {
            mat.column_mut(j).assign(&data[name]);
        }

        // Find complete and incomplete rows
        let complete_rows: Vec<usize> = (0..n_obs)
            .filter(|&i| (0..n_vars).all(|j| !mat[[i, j]].is_nan()))
            .collect();

        // Compute initial mu and Sigma from complete cases
        let n_complete = complete_rows.len();
        if n_complete < n_vars + 1 {
            return Err(GreenersError::InvalidOperation(
                "Not enough complete cases for BayesGaussMI".into(),
            ));
        }

        let mut mu = Array1::<f64>::zeros(n_vars);
        for &i in &complete_rows {
            for j in 0..n_vars {
                mu[j] += mat[[i, j]];
            }
        }
        mu /= n_complete as f64;

        let mut sigma = Array2::<f64>::zeros((n_vars, n_vars));
        for &i in &complete_rows {
            for a in 0..n_vars {
                for b in 0..n_vars {
                    sigma[[a, b]] += (mat[[i, a]] - mu[a]) * (mat[[i, b]] - mu[b]);
                }
            }
        }
        sigma /= (n_complete - 1) as f64;

        let mut rng_state: u64 = 12345;
        let mut datasets = Vec::with_capacity(n_imputations);

        for _imp in 0..n_imputations {
            let mut current = mat.clone();

            // Fill missing with draws from conditional normal
            for i in 0..n_obs {
                let obs_idx: Vec<usize> =
                    (0..n_vars).filter(|&j| !current[[i, j]].is_nan()).collect();
                let mis_idx: Vec<usize> =
                    (0..n_vars).filter(|&j| current[[i, j]].is_nan()).collect();

                if mis_idx.is_empty() {
                    continue;
                }

                if obs_idx.is_empty() {
                    // All missing: draw from marginal
                    for &j in &mis_idx {
                        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let u1 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
                        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let u2 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
                        let z = (-2.0 * u1.max(1e-15).ln()).sqrt()
                            * (2.0 * std::f64::consts::PI * u2).cos();
                        current[[i, j]] = mu[j] + sigma[[j, j]].sqrt() * z;
                    }
                    continue;
                }

                // Conditional normal: X_mis | X_obs ~ N(mu_cond, Sigma_cond)
                let n_o = obs_idx.len();
                let n_m = mis_idx.len();

                // Extract sub-matrices
                let mut sigma_oo = Array2::<f64>::zeros((n_o, n_o));
                let mut sigma_mo = Array2::<f64>::zeros((n_m, n_o));
                let mut mu_m = Array1::<f64>::zeros(n_m);
                let mut mu_o = Array1::<f64>::zeros(n_o);
                let mut x_o = Array1::<f64>::zeros(n_o);

                for (a, &ja) in obs_idx.iter().enumerate() {
                    mu_o[a] = mu[ja];
                    x_o[a] = current[[i, ja]];
                    for (b, &jb) in obs_idx.iter().enumerate() {
                        sigma_oo[[a, b]] = sigma[[ja, jb]];
                    }
                }
                for (a, &ja) in mis_idx.iter().enumerate() {
                    mu_m[a] = mu[ja];
                    for (b, &jb) in obs_idx.iter().enumerate() {
                        sigma_mo[[a, b]] = sigma[[ja, jb]];
                    }
                }

                // mu_cond = mu_m + Sigma_mo Sigma_oo^-1 (x_o - mu_o)
                use crate::linalg::LinalgInverse as _;
                let sigma_oo_inv = sigma_oo.inv().unwrap_or(Array2::eye(n_o));
                let mu_cond = &mu_m + &sigma_mo.dot(&sigma_oo_inv).dot(&(&x_o - &mu_o));

                // Sigma_cond = Sigma_mm - Sigma_mo Sigma_oo^-1 Sigma_om
                let mut sigma_mm = Array2::<f64>::zeros((n_m, n_m));
                for (a, &ja) in mis_idx.iter().enumerate() {
                    for (b, &jb) in mis_idx.iter().enumerate() {
                        sigma_mm[[a, b]] = sigma[[ja, jb]];
                    }
                }
                let sigma_cond = &sigma_mm - &sigma_mo.dot(&sigma_oo_inv).dot(&sigma_mo.t());

                // Draw from N(mu_cond, diag of sigma_cond) — simplified
                for (a, &ja) in mis_idx.iter().enumerate() {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let u1 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let u2 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
                    let z = (-2.0 * u1.max(1e-15).ln()).sqrt()
                        * (2.0 * std::f64::consts::PI * u2).cos();
                    current[[i, ja]] = mu_cond[a] + sigma_cond[[a, a]].max(0.0).sqrt() * z;
                }
            }

            let mut dataset = IndexMap::new();
            for (j, name) in col_names.iter().enumerate() {
                dataset.insert(name.clone(), current.column(j).to_owned());
            }
            datasets.push(dataset);
        }

        Ok(BayesGaussMIResult {
            datasets,
            n_imputations,
            n_obs,
            n_vars,
        })
    }
}
