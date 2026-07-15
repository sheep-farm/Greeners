//! Factor-augmented panel regression.
//!
//! Giannone, Lenza & Primiceri (2013) style. Combines a large
//! set of macroeconomic indicators into common factors, then
//! uses these factors as additional regressors in a panel
//! regression with fixed effects:
//!
//! y_it = alpha_i + beta' * x_it + gamma' * f_t + eps_it
//!
//! where f_t are common factors extracted from a large panel of
//! auxiliary indicators via PCA, and x_it are the traditional
//! regressors.
//!
//! Two-step estimation:
//! 1. Extract factors via PCA from the auxiliary panel
//! 2. Estimate the augmented panel regression via FE (within transformation)

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of factor-augmented panel estimation.
#[derive(Debug)]
pub struct FaPanelResult {
    /// Coefficients on traditional regressors (beta)
    pub beta: Array1<f64>,
    /// Coefficients on factors (gamma)
    pub gamma: Array1<f64>,
    /// SE of beta
    pub beta_se: Array1<f64>,
    /// SE of gamma
    pub gamma_se: Array1<f64>,
    /// t-values of beta
    pub beta_t: Array1<f64>,
    /// t-values of gamma
    pub gamma_t: Array1<f64>,
    /// p-values of beta
    pub beta_p: Array1<f64>,
    /// p-values of gamma
    pub gamma_p: Array1<f64>,
    /// Extracted factors (T x n_factors)
    pub factors: Array2<f64>,
    /// R-squared
    pub r_squared: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of entities
    pub n_entities: usize,
    /// Number of traditional regressors
    pub n_regressors: usize,
    /// Number of factors
    pub n_factors: usize,
    /// Regressor names
    pub regressor_names: Vec<String>,
    /// Factor names
    pub factor_names: Vec<String>,
}

impl fmt::Display for FaPanelResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Factor-Augmented Panel ")?;
        writeln!(f, "PCA factors + FE panel regression")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Entities:", self.n_entities)?;
        writeln!(f, "{:<20} {:>12}", "Regressors:", self.n_regressors)?;
        writeln!(f, "{:<20} {:>12}", "Factors:", self.n_factors)?;
        writeln!(f, "{:<20} {:>12.6}", "R-squared:", self.r_squared)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<14} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coef.", "Std.Err.", "t", "P>|t|"
        )?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.n_regressors {
            let name = self
                .regressor_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("x{}", i));
            writeln!(
                f,
                "{:<14} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
                name, self.beta[i], self.beta_se[i], self.beta_t[i], self.beta_p[i]
            )?;
        }

        for i in 0..self.n_factors {
            let name = self
                .factor_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("F{}", i + 1));
            writeln!(
                f,
                "{:<14} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
                name, self.gamma[i], self.gamma_se[i], self.gamma_t[i], self.gamma_p[i]
            )?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct FAPanel;

impl FAPanel {
    /// Estimate factor-augmented panel.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (n)
    /// * `x` - Traditional regressors (n x k)
    /// * `aux` - Auxiliary panel for factor extraction (T x n_aux), one row per period
    /// * `entity_ids` - Entity identifier (n)
    /// * `period_ids` - Period identifier (n), must match aux rows
    /// * `n_factors` - Number of factors to extract
    /// * `regressor_names` - Optional names for x
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        aux: &Array2<f64>,
        entity_ids: &[i64],
        period_ids: &[i64],
        n_factors: usize,
        regressor_names: Option<Vec<String>>,
    ) -> Result<FaPanelResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        let t_aux = aux.nrows();
        let n_aux = aux.ncols();

        if x.nrows() != n || entity_ids.len() != n || period_ids.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "FAPanel: dimension mismatch".into(),
            ));
        }
        if n_factors == 0 || n_factors >= n_aux {
            return Err(GreenersError::InvalidOperation(
                "FAPanel: invalid n_factors".into(),
            ));
        }

        let names = regressor_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());

        // Step 1: Extract factors via PCA from auxiliary panel
        let factors = Self::pca_factors(aux, n_factors, t_aux)?;

        // Step 2: Build augmented design matrix
        // For each observation, match factor row by period_id
        let mut unique_periods: Vec<i64> = period_ids
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        unique_periods.sort();

        if unique_periods.len() != t_aux {
            return Err(GreenersError::InvalidOperation(
                "FAPanel: period count must match aux rows".into(),
            ));
        }

        let period_to_factor: std::collections::HashMap<i64, Array1<f64>> = unique_periods
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, factors.row(i).to_owned()))
            .collect();

        // Within transformation (demean by entity)
        let mut entity_sums: std::collections::HashMap<i64, (f64, usize)> =
            std::collections::HashMap::new();
        for i in 0..n {
            let entry = entity_sums.entry(entity_ids[i]).or_insert((0.0, 0));
            entry.0 += y[i];
            entry.1 += 1;
        }
        let entity_means: std::collections::HashMap<i64, f64> = entity_sums
            .iter()
            .map(|(&k, &(s, c))| (k, s / c as f64))
            .collect();

        let mut y_dm = Array1::zeros(n);
        for i in 0..n {
            y_dm[i] = y[i] - entity_means[&entity_ids[i]];
        }

        // Demean x and factors by entity
        let n_total_reg = k + n_factors;
        let mut z_dm = Array2::zeros((n, n_total_reg));

        for j in 0..k {
            let mut x_sums: std::collections::HashMap<i64, (f64, usize)> =
                std::collections::HashMap::new();
            for i in 0..n {
                let xe = x_sums.entry(entity_ids[i]).or_insert((0.0, 0));
                xe.0 += x[(i, j)];
                xe.1 += 1;
            }
            for i in 0..n {
                let xm = x_sums[&entity_ids[i]].0 / x_sums[&entity_ids[i]].1 as f64;
                z_dm[(i, j)] = x[(i, j)] - xm;
            }
        }

        // Factor columns
        for j in 0..n_factors {
            let mut f_sums: std::collections::HashMap<i64, (f64, usize)> =
                std::collections::HashMap::new();
            for i in 0..n {
                let f_val = period_to_factor[&period_ids[i]][j];
                let fe = f_sums.entry(entity_ids[i]).or_insert((0.0, 0));
                fe.0 += f_val;
                fe.1 += 1;
            }
            for i in 0..n {
                let f_val = period_to_factor[&period_ids[i]][j];
                let fm = f_sums[&entity_ids[i]].0 / f_sums[&entity_ids[i]].1 as f64;
                z_dm[(i, k + j)] = f_val - fm;
            }
        }

        // OLS on demeaned data
        let zt = z_dm.t();
        let ztz = zt.dot(&z_dm);
        let ztz_inv = (&ztz + Array2::<f64>::eye(n_total_reg) * 1e-8).inv()?;
        let zty = zt.dot(&y_dm);
        let beta_full: Array1<f64> = ztz_inv.dot(&zty);

        let residuals = &y_dm - z_dm.dot(&beta_full);
        let sse = residuals.dot(&residuals);
        let n_entities = entity_sums.len();
        let sigma2 = sse / (n - n_entities - n_total_reg) as f64;
        let cov = &ztz_inv * sigma2;
        let se = cov.diag().mapv(|v| v.sqrt());
        let tv = &beta_full / &se;
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let pv = tv.mapv(|t| 2.0 * (1.0 - normal.cdf(t.abs())));

        let beta = beta_full.slice(ndarray::s![0..k]).to_owned();
        let gamma = beta_full.slice(ndarray::s![k..n_total_reg]).to_owned();
        let beta_se = se.slice(ndarray::s![0..k]).to_owned();
        let gamma_se = se.slice(ndarray::s![k..n_total_reg]).to_owned();
        let beta_t = tv.slice(ndarray::s![0..k]).to_owned();
        let gamma_t = tv.slice(ndarray::s![k..n_total_reg]).to_owned();
        let beta_p = pv.slice(ndarray::s![0..k]).to_owned();
        let gamma_p = pv.slice(ndarray::s![k..n_total_reg]).to_owned();

        // R-squared
        let y_mean = y_dm.mean().unwrap_or(0.0);
        let tss = y_dm.mapv(|v| (v - y_mean).powi(2)).sum();
        let r_squared = if tss > 1e-15 { 1.0 - sse / tss } else { 0.0 };

        let factor_names: Vec<String> = (0..n_factors).map(|i| format!("F{}", i + 1)).collect();

        Ok(FaPanelResult {
            beta,
            gamma,
            beta_se,
            gamma_se,
            beta_t,
            gamma_t,
            beta_p,
            gamma_p,
            factors,
            r_squared,
            n_obs: n,
            n_entities,
            n_regressors: k,
            n_factors,
            regressor_names: names,
            factor_names,
        })
    }

    /// Extract factors via PCA (power iteration with deflation).
    fn pca_factors(
        aux: &Array2<f64>,
        n_factors: usize,
        t: usize,
    ) -> Result<Array2<f64>, GreenersError> {
        let n_aux = aux.ncols();

        // Standardize aux
        let mut aux_std = Array2::zeros((t, n_aux));
        for j in 0..n_aux {
            let mean: f64 = aux.column(j).mean().unwrap_or(0.0);
            let std_val: f64 = aux.column(j).std(0.0).max(1e-10);
            for i in 0..t {
                aux_std[(i, j)] = (aux[(i, j)] - mean) / std_val;
            }
        }

        // Covariance (n_aux x n_aux)
        let aux_t = aux_std.t();
        let cov = aux_t.dot(&aux_std) / t as f64;

        let mut factors = Array2::zeros((t, n_factors));
        let mut remaining = cov.clone();

        for f in 0..n_factors {
            let mut v = Array1::ones(n_aux) / (n_aux as f64).sqrt();
            for _ in 0..100 {
                let v_new = remaining.dot(&v);
                let norm = v_new.mapv(|x| x * x).sum().sqrt().max(1e-10);
                v = v_new / norm;
            }
            let lambda = v.dot(&remaining.dot(&v));
            // Factor scores
            for i in 0..t {
                factors[(i, f)] = aux_std.row(i).dot(&v);
            }
            // Deflate
            for a in 0..n_aux {
                for b in 0..n_aux {
                    remaining[(a, b)] -= lambda * v[a] * v[b];
                }
            }
        }

        Ok(factors)
    }
}
