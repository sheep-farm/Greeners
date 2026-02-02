use crate::error::GreenersError;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use statrs::distribution::ContinuousCDF;
use std::collections::HashMap;
use std::fmt;

/// Result of Mixed Linear Model estimation.
#[derive(Debug)]
pub struct MixedResult {
    pub fixed_effects: Array1<f64>,
    pub fixed_se: Array1<f64>,
    pub z_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub random_effects: HashMap<usize, Array1<f64>>,
    pub var_random: Array2<f64>,
    pub var_resid: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub n_obs: usize,
    pub n_groups: usize,
    pub converged: bool,
    pub n_iter: usize,
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for MixedResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Mixed Linear Model (REML) ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10}", "Groups:", self.n_groups)?;
        writeln!(f, "{:<20} {:>10.4}", "Log-Likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>10.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>10.4}", "BIC:", self.bic)?;
        writeln!(f, "{:<20} {:>10.4}", "Residual var:", self.var_resid)?;

        writeln!(f, "\nFixed Effects:")?;
        writeln!(f, "{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} | {:>10} | {:>10} | {:>8} | {:>8}",
            "Variable", "coef", "std err", "z", "P>|z|"
        )?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.fixed_effects.len() {
            let name = self
                .variable_names
                .as_ref()
                .and_then(|n| n.get(i).cloned())
                .unwrap_or_else(|| format!("x{}", i));
            writeln!(
                f,
                "{:<12} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3}",
                name, self.fixed_effects[i], self.fixed_se[i], self.z_values[i], self.p_values[i]
            )?;
        }

        writeln!(f, "\nRandom Effects Variance:")?;
        for i in 0..self.var_random.nrows() {
            for j in 0..self.var_random.ncols() {
                write!(f, " {:>8.4}", self.var_random[[i, j]])?;
            }
            writeln!(f)?;
        }

        writeln!(f, "{:=^78}", "")
    }
}

/// Mixed Linear Model with REML estimation.
pub struct MixedLM;

impl MixedLM {
    /// Fit a mixed linear model.
    ///
    /// - y: response vector (n)
    /// - x_fixed: fixed effects design matrix (n x p)
    /// - groups: group indicators (n), integer group IDs
    /// - x_random: random effects design matrix (n x q), often just intercept
    pub fn fit(
        y: &Array1<f64>,
        x_fixed: &Array2<f64>,
        groups: &Array1<usize>,
        x_random: &Array2<f64>,
    ) -> Result<MixedResult, GreenersError> {
        Self::fit_with_names(y, x_fixed, groups, x_random, None)
    }

    pub fn fit_with_names(
        y: &Array1<f64>,
        x_fixed: &Array2<f64>,
        groups: &Array1<usize>,
        x_random: &Array2<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<MixedResult, GreenersError> {
        let n = y.len();
        let p = x_fixed.ncols();
        let q = x_random.ncols();

        if n != x_fixed.nrows() || n != groups.len() || n != x_random.nrows() {
            return Err(GreenersError::ShapeMismatch(
                "Dimension mismatch in MixedLM inputs".into(),
            ));
        }

        // Identify groups
        let mut unique_groups: Vec<usize> = groups.iter().cloned().collect();
        unique_groups.sort();
        unique_groups.dedup();
        let g = unique_groups.len();

        // Group indices
        let group_indices: Vec<Vec<usize>> = unique_groups
            .iter()
            .map(|&grp| (0..n).filter(|&i| groups[i] == grp).collect())
            .collect();

        // Initialize variance components
        // D = random effects covariance (q x q)
        // sigma2 = residual variance
        let mut d_mat = Array2::<f64>::eye(q);
        let mut sigma2 = 1.0;

        let max_iter = 200;
        let tol = 1e-6;
        let mut converged = false;
        let mut n_iter = 0;
        let mut beta = Array1::<f64>::zeros(p);
        let mut blups: HashMap<usize, Array1<f64>> = HashMap::new();

        for iter in 0..max_iter {
            n_iter = iter + 1;

            // E-step: compute V_i = Z_i D Z_i' + sigma2 I for each group
            // Then solve for beta via GLS: (sum X_i' V_i^-1 X_i)^-1 sum X_i' V_i^-1 y_i

            let mut xtvinvx = Array2::<f64>::zeros((p, p));
            let mut xtvinvy = Array1::<f64>::zeros(p);

            for idx in &group_indices {
                let ni = idx.len();
                let zi = stack_rows(x_random, idx);
                let xi = stack_rows(x_fixed, idx);
                let yi: Array1<f64> = idx.iter().map(|&i| y[i]).collect::<Vec<_>>().into();

                // V_i = Z_i D Z_i' + sigma2 I
                let zdzt = zi.dot(&d_mat).dot(&zi.t());
                let mut vi = zdzt;
                for j in 0..ni {
                    vi[[j, j]] += sigma2;
                }

                let vi_inv = match vi.inv() {
                    Ok(inv) => inv,
                    Err(_) => Array2::eye(ni) / sigma2,
                };

                xtvinvx = &xtvinvx + &xi.t().dot(&vi_inv).dot(&xi);
                xtvinvy = &xtvinvy + &xi.t().dot(&vi_inv).dot(&yi);
            }

            let new_beta = match xtvinvx.inv() {
                Ok(inv) => inv.dot(&xtvinvy),
                Err(_) => beta.clone(),
            };

            // M-step: update variance components
            let mut sum_d = Array2::<f64>::zeros((q, q));
            let mut sum_sigma2 = 0.0;
            let mut total_obs = 0;

            for (gi_idx, idx) in group_indices.iter().enumerate() {
                let ni = idx.len();
                let zi = stack_rows(x_random, idx);
                let xi = stack_rows(x_fixed, idx);
                let yi: Array1<f64> = idx.iter().map(|&i| y[i]).collect::<Vec<_>>().into();

                let ri = &yi - &xi.dot(&new_beta);

                let zdzt = zi.dot(&d_mat).dot(&zi.t());
                let mut vi = zdzt;
                for j in 0..ni {
                    vi[[j, j]] += sigma2;
                }

                let vi_inv = match vi.inv() {
                    Ok(inv) => inv,
                    Err(_) => Array2::eye(ni) / sigma2,
                };

                // BLUP: u_i = D Z_i' V_i^-1 r_i
                let u_i = d_mat.dot(&zi.t()).dot(&vi_inv).dot(&ri);
                blups.insert(unique_groups[gi_idx], u_i.clone());

                // Conditional covariance: D - D Z' V^-1 Z D
                let cond_cov = &d_mat - &d_mat.dot(&zi.t()).dot(&vi_inv).dot(&zi).dot(&d_mat);

                // Update D contribution
                let outer_u = outer_product(&u_i, &u_i);
                sum_d = &sum_d + &(&outer_u + &cond_cov);

                // Update sigma2 contribution
                let fitted_ri = &ri - &zi.dot(&u_i);
                sum_sigma2 += fitted_ri.dot(&fitted_ri);

                // Trace correction
                let trace_corr = {
                    let m = vi_inv.dot(&zi).dot(&d_mat);
                    let diag_sum: f64 = m.diag().iter().sum();
                    let tr = sigma2 * (ni.min(q) as f64) * (1.0 - diag_sum / ni as f64);
                    tr.max(0.0)
                };
                sum_sigma2 += trace_corr;

                total_obs += ni;
            }

            let new_d = &sum_d / g as f64;
            let new_sigma2 = (sum_sigma2 / total_obs as f64).max(1e-10);

            // Check convergence
            let diff_beta = (&new_beta - &beta)
                .iter()
                .map(|d| d.abs())
                .fold(0.0_f64, f64::max);
            let diff_sigma = (new_sigma2 - sigma2).abs();

            beta = new_beta;
            d_mat = new_d;
            sigma2 = new_sigma2;

            if diff_beta < tol && diff_sigma < tol {
                converged = true;
                break;
            }
        }

        // Compute fixed effects standard errors
        let mut xtvinvx = Array2::<f64>::zeros((p, p));
        for idx in &group_indices {
            let ni = idx.len();
            let zi = stack_rows(x_random, idx);
            let xi = stack_rows(x_fixed, idx);

            let zdzt = zi.dot(&d_mat).dot(&zi.t());
            let mut vi = zdzt;
            for j in 0..ni {
                vi[[j, j]] += sigma2;
            }

            let vi_inv = match vi.inv() {
                Ok(inv) => inv,
                Err(_) => Array2::eye(ni) / sigma2,
            };

            xtvinvx = &xtvinvx + &xi.t().dot(&vi_inv).dot(&xi);
        }

        let cov_beta = xtvinvx.inv()?;
        let fixed_se: Array1<f64> = (0..p)
            .map(|j| cov_beta[[j, j]].abs().sqrt())
            .collect::<Vec<_>>()
            .into();

        let z_values = &beta / &fixed_se;
        let normal = statrs::distribution::Normal::new(0.0, 1.0)
            .map_err(|_| GreenersError::OptimizationFailed)?;
        let p_values = z_values.mapv(|z| 2.0 * (1.0 - normal.cdf(z.abs())));

        // Log-likelihood (approximate REML)
        let mut ll = -0.5 * n as f64 * (2.0 * std::f64::consts::PI * sigma2).ln();
        for idx in &group_indices {
            let ni = idx.len();
            let zi = stack_rows(x_random, idx);
            let xi = stack_rows(x_fixed, idx);
            let yi: Array1<f64> = idx.iter().map(|&i| y[i]).collect::<Vec<_>>().into();
            let ri = &yi - &xi.dot(&beta);

            let zdzt = zi.dot(&d_mat).dot(&zi.t());
            let mut vi = zdzt;
            for j in 0..ni {
                vi[[j, j]] += sigma2;
            }

            let vi_inv = match vi.inv() {
                Ok(inv) => inv,
                Err(_) => continue,
            };

            ll -= 0.5 * ri.dot(&vi_inv.dot(&ri));
        }

        let n_var_params = q * (q + 1) / 2 + 1; // D lower triangle + sigma2
        let total_params = p + n_var_params;
        let aic = -2.0 * ll + 2.0 * total_params as f64;
        let bic = -2.0 * ll + (total_params as f64) * (n as f64).ln();

        Ok(MixedResult {
            fixed_effects: beta,
            fixed_se,
            z_values,
            p_values,
            random_effects: blups,
            var_random: d_mat,
            var_resid: sigma2,
            log_likelihood: ll,
            aic,
            bic,
            n_obs: n,
            n_groups: g,
            converged,
            n_iter,
            variable_names,
        })
    }
}

fn stack_rows(mat: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let k = mat.ncols();
    let mut result = Array2::<f64>::zeros((indices.len(), k));
    for (i, &idx) in indices.iter().enumerate() {
        result.row_mut(i).assign(&mat.row(idx));
    }
    result
}

fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    let mut result = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}
