use crate::error::GreenersError;
use crate::{CovarianceType, OLS};
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use std::fmt;

/// Influence diagnostics for regression models (DFBetas, DFFITS, CUSUM).
pub struct Influence;

/// Result of influence diagnostics.
#[derive(Debug)]
pub struct InfluenceResult {
    /// DFBetas: change in each coefficient when observation i is deleted (n x k)
    pub dfbetas: Array2<f64>,
    /// DFFITS: change in fitted value when observation i is deleted (n)
    pub dffits: Array1<f64>,
    /// Leverage values (hat matrix diagonal)
    pub leverage: Array1<f64>,
    /// Internally studentized residuals
    pub student_resid: Array1<f64>,
    /// Externally studentized residuals
    pub student_resid_external: Array1<f64>,
    pub n_obs: usize,
    pub n_params: usize,
}

impl InfluenceResult {
    /// DFBetas threshold: 2/sqrt(n)
    pub fn dfbetas_threshold(&self) -> f64 {
        2.0 / (self.n_obs as f64).sqrt()
    }

    /// DFFITS threshold: 2*sqrt(k/n)
    pub fn dffits_threshold(&self) -> f64 {
        2.0 * (self.n_params as f64 / self.n_obs as f64).sqrt()
    }

    /// Indices of observations exceeding DFFITS threshold
    pub fn influential_dffits(&self) -> Vec<usize> {
        let thresh = self.dffits_threshold();
        (0..self.n_obs)
            .filter(|&i| self.dffits[i].abs() > thresh)
            .collect()
    }
}

impl fmt::Display for InfluenceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^60}", " Influence Diagnostics ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10}", "Parameters:", self.n_params)?;
        writeln!(
            f,
            "{:<20} {:>10.4}",
            "DFBetas threshold:",
            self.dfbetas_threshold()
        )?;
        writeln!(
            f,
            "{:<20} {:>10.4}",
            "DFFITS threshold:",
            self.dffits_threshold()
        )?;

        let influential = self.influential_dffits();
        if influential.is_empty() {
            writeln!(f, "\nNo influential observations detected by DFFITS.")?;
        } else {
            writeln!(
                f,
                "\nInfluential observations (DFFITS): {:?}",
                &influential[..influential.len().min(20)]
            )?;
        }
        writeln!(f, "{:=^60}", "")
    }
}

impl Influence {
    /// Compute full influence diagnostics.
    ///
    /// - residuals: OLS residuals
    /// - x: design matrix (n x k)
    /// - mse: mean squared error (sigma^2)
    pub fn compute(
        residuals: &Array1<f64>,
        x: &Array2<f64>,
        mse: f64,
    ) -> Result<InfluenceResult, GreenersError> {
        let n = residuals.len();
        let k = x.ncols();

        if n != x.nrows() {
            return Err(GreenersError::ShapeMismatch(
                "residuals and x row count mismatch".into(),
            ));
        }

        // Hat matrix diagonal
        let xtx_inv = x.t().dot(x).inv()?;
        let mut leverage = Array1::<f64>::zeros(n);
        for i in 0..n {
            let xi = x.row(i);
            let temp = xtx_inv.dot(&xi);
            leverage[i] = xi.dot(&temp);
        }

        // Internally studentized residuals: r_i = e_i / (s * sqrt(1 - h_i))
        let s = mse.sqrt();
        let student_resid: Array1<f64> = (0..n)
            .map(|i| {
                let denom = s * (1.0 - leverage[i]).max(1e-15).sqrt();
                residuals[i] / denom
            })
            .collect::<Vec<_>>()
            .into();

        // Externally studentized: use leave-one-out MSE
        // s_{(i)}^2 = ((n-k)*s^2 - e_i^2/(1-h_i)) / (n-k-1)
        let nk = (n - k) as f64;
        let student_resid_external: Array1<f64> = (0..n)
            .map(|i| {
                let s_i_sq = (nk * mse - residuals[i].powi(2) / (1.0 - leverage[i]).max(1e-15))
                    / (nk - 1.0).max(1.0);
                let denom = s_i_sq.max(1e-15).sqrt() * (1.0 - leverage[i]).max(1e-15).sqrt();
                residuals[i] / denom
            })
            .collect::<Vec<_>>()
            .into();

        // DFFITS: r*_i * sqrt(h_i / (1 - h_i))
        let dffits: Array1<f64> = (0..n)
            .map(|i| {
                student_resid_external[i] * (leverage[i] / (1.0 - leverage[i]).max(1e-15)).sqrt()
            })
            .collect::<Vec<_>>()
            .into();

        // DFBetas: change in each beta_j when observation i is deleted
        // DFBeta_{ij} = (X'X)^{-1} x_i e_i / ((1 - h_i) * s_{(i)})
        let mut dfbetas = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let xi = x.row(i);
            let one_minus_h = (1.0 - leverage[i]).max(1e-15);

            let s_i_sq = (nk * mse - residuals[i].powi(2) / one_minus_h) / (nk - 1.0).max(1.0);
            let s_i = s_i_sq.max(1e-15).sqrt();

            let xtx_inv_xi = xtx_inv.dot(&xi);
            for j in 0..k {
                dfbetas[[i, j]] = xtx_inv_xi[j] * residuals[i]
                    / (one_minus_h * s_i * xtx_inv[[j, j]].abs().sqrt().max(1e-15));
            }
        }

        Ok(InfluenceResult {
            dfbetas,
            dffits,
            leverage,
            student_resid,
            student_resid_external,
            n_obs: n,
            n_params: k,
        })
    }
}

/// CUSUM test for structural stability.
pub struct CUSUMTest;

/// Result of CUSUM test.
#[derive(Debug)]
pub struct CUSUMResult {
    /// Cumulative sum of recursive residuals
    pub cusum: Array1<f64>,
    /// Upper 5% significance bound
    pub upper_bound: Array1<f64>,
    /// Lower 5% significance bound
    pub lower_bound: Array1<f64>,
    /// Whether the CUSUM stays within bounds (stable)
    pub is_stable: bool,
    pub n_obs: usize,
}

impl fmt::Display for CUSUMResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^60}", " CUSUM Test for Structural Stability ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(
            f,
            "{:<20} {:>10}",
            "Stable:",
            if self.is_stable { "Yes" } else { "No" }
        )?;
        let max_cusum = self.cusum.iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
        writeln!(f, "{:<20} {:>10.4}", "Max |CUSUM|:", max_cusum)?;
        writeln!(f, "{:=^60}", "")
    }
}

impl CUSUMTest {
    /// Compute CUSUM test from OLS residuals.
    ///
    /// Uses recursive residuals and checks if cumulative sum
    /// stays within significance bounds.
    pub fn test(y: &Array1<f64>, x: &Array2<f64>) -> Result<CUSUMResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();

        if n <= k + 1 {
            return Err(GreenersError::InvalidOperation(
                "Not enough observations for CUSUM test".into(),
            ));
        }

        // Compute recursive residuals using expanding window OLS
        let mut recursive_resid = Vec::new();

        for t in k..n {
            // Fit OLS on [0..t]
            let x_sub = x.slice(ndarray::s![..t, ..]).to_owned();
            let y_sub = y.slice(ndarray::s![..t]).to_owned();

            if let Ok(ols) = OLS::fit(&y_sub, &x_sub, CovarianceType::NonRobust) {
                // One-step-ahead prediction error
                let x_t = x.row(t);
                let y_pred = x_t.dot(&ols.params);
                let e_t = y[t] - y_pred;

                // Standardize by forecast error variance
                let xtx_inv = match x_sub.t().dot(&x_sub).inv() {
                    Ok(inv) => inv,
                    Err(_) => continue,
                };
                let f_t = 1.0 + x_t.dot(&xtx_inv.dot(&x_t));
                let sigma = ols.sigma;
                let w_t = e_t / (sigma * f_t.sqrt()).max(1e-15);
                recursive_resid.push(w_t);
            }
        }

        let m = recursive_resid.len();
        if m < 2 {
            return Err(GreenersError::InvalidOperation(
                "Not enough recursive residuals".into(),
            ));
        }

        // Standardize recursive residuals
        let ss: f64 = recursive_resid.iter().map(|r| r * r).sum();
        let sigma_w = (ss / m as f64).sqrt().max(1e-15);

        let standardized: Vec<f64> = recursive_resid.iter().map(|r| r / sigma_w).collect();

        // Cumulative sum
        let mut cusum = Array1::<f64>::zeros(m);
        let mut cumsum = 0.0;
        for i in 0..m {
            cumsum += standardized[i];
            cusum[i] = cumsum / (m as f64).sqrt();
        }

        // Significance bounds: ±(a + 2*a*t/T) where a = 0.948 for 5%
        let a = 0.948;
        let upper_bound: Array1<f64> = (0..m)
            .map(|i| a + 2.0 * a * (i as f64) / (m as f64))
            .collect::<Vec<_>>()
            .into();
        let lower_bound = upper_bound.mapv(|v| -v);

        let is_stable = (0..m).all(|i| cusum[i] >= lower_bound[i] && cusum[i] <= upper_bound[i]);

        Ok(CUSUMResult {
            cusum,
            upper_bound,
            lower_bound,
            is_stable,
            n_obs: n,
        })
    }
}
