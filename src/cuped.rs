//! CUPED — Controlled-Experiment Using Pre-Experiment Data.
//!
//! Deng, Xu, Kohavi & Walker (2013). Improves precision of
//! A/B tests by adjusting the outcome using a pre-treatment
//! covariate:
//!
//!   Y_adjusted = Y - theta * (X - E[X])
//!
//! where theta = Cov(Y, X) / Var(X), and X is a pre-treatment
//! outcome (or any covariate correlated with Y).
//!
//! The adjusted outcome has the same expected value as Y but
//! reduced variance, leading to tighter confidence intervals.
//!
//! The treatment effect is then estimated as the difference in
//! means of Y_adjusted between treatment and control groups.

use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of CUPED estimation.
#[derive(Debug)]
pub struct CupedResult {
    /// CUPED-adjusted treatment effect
    pub treatment_effect: f64,
    /// Standard error of the treatment effect
    pub se: f64,
    /// t-statistic
    pub t_stat: f64,
    /// p-value (two-sided)
    pub p_value: f64,
    /// 95% confidence interval [lower, upper]
    pub ci: [f64; 2],
    /// Theta (covariance adjustment coefficient)
    pub theta: f64,
    /// Unadjusted treatment effect (naive DiD)
    pub unadjusted_effect: f64,
    /// Unadjusted standard error
    pub unadjusted_se: f64,
    /// Variance reduction ratio (unadjusted_var / adjusted_var)
    pub variance_reduction: f64,
    /// Adjusted outcome variance
    pub adjusted_variance: f64,
    /// Unadjusted outcome variance
    pub unadjusted_variance: f64,
    /// Number of treatment observations
    pub n_treatment: usize,
    /// Number of control observations
    pub n_control: usize,
    /// Total observations
    pub n_obs: usize,
}

impl fmt::Display for CupedResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " CUPED ")?;
        writeln!(f, "Deng-Xu-Kohavi-Walker (2013)")?;
        writeln!(f, "Controlled-experiment using pre-experiment data")?;
        writeln!(f, "{:<20} {:>12}", "Treatment obs:", self.n_treatment)?;
        writeln!(f, "{:<20} {:>12}", "Control obs:", self.n_control)?;
        writeln!(f, "{:<20} {:>12}", "Total obs:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12.6}", "Theta (adj. coef.):", self.theta)?;
        writeln!(
            f,
            "{:<20} {:>12.6}",
            "Variance reduction:", self.variance_reduction
        )?;
        writeln!(
            f,
            "{:<20} {:>12.4}%",
            "  (relative)",
            self.variance_reduction * 100.0
        )?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "  {:<22} {:>12} {:>12} {:>10} {:>10}",
            "", "Effect", "SE", "t", "P>|t|"
        )?;
        writeln!(f, "{:-^78}", "")?;
        writeln!(
            f,
            "  {:<22} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
            "Unadjusted (DiD):",
            self.unadjusted_effect,
            self.unadjusted_se,
            if self.unadjusted_se > 1e-10 {
                self.unadjusted_effect / self.unadjusted_se
            } else {
                0.0
            },
            if self.unadjusted_se > 1e-10 {
                2.0 * (1.0
                    - Normal::new(0.0, 1.0)
                        .unwrap()
                        .cdf((self.unadjusted_effect / self.unadjusted_se).abs()))
            } else {
                0.0
            }
        )?;
        writeln!(
            f,
            "  {:<22} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
            "CUPED adjusted:", self.treatment_effect, self.se, self.t_stat, self.p_value
        )?;

        writeln!(f, "\n  95% CI: [{:.6}, {:.6}]", self.ci[0], self.ci[1])?;

        write!(f, "{:=^78}", "")
    }
}

pub struct CUPED;

impl CUPED {
    /// Estimate CUPED-adjusted treatment effect.
    ///
    /// # Arguments
    /// * `y` - Post-treatment outcome (n)
    /// * `x` - Pre-treatment covariate (n), correlated with y
    /// * `treated` - Treatment indicator (n), true if treated
    pub fn fit(
        y: &Array1<f64>,
        x: &Array1<f64>,
        treated: &[bool],
    ) -> Result<CupedResult, GreenersError> {
        let n = y.len();
        if x.len() != n || treated.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "CUPED: y, x, treated must have same length".into(),
            ));
        }
        if n < 4 {
            return Err(GreenersError::InvalidOperation(
                "CUPED: need at least 4 observations".into(),
            ));
        }

        let n_treatment = treated.iter().filter(|&&t| t).count();
        let n_control = n - n_treatment;
        if n_treatment == 0 || n_control == 0 {
            return Err(GreenersError::InvalidOperation(
                "CUPED: need both treatment and control groups".into(),
            ));
        }

        // Step 1: Compute theta = Cov(Y, X) / Var(X)
        let y_mean = y.mean().unwrap_or(0.0);
        let x_mean = x.mean().unwrap_or(0.0);

        let cov_yx: f64 = y
            .iter()
            .zip(x.iter())
            .map(|(&yi, &xi)| (yi - y_mean) * (xi - x_mean))
            .sum::<f64>()
            / n as f64;

        let var_x: f64 = x.iter().map(|xi| (xi - x_mean).powi(2)).sum::<f64>() / n as f64;

        if var_x < 1e-15 {
            return Err(GreenersError::InvalidOperation(
                "CUPED: pre-treatment covariate has zero variance".into(),
            ));
        }

        let theta = cov_yx / var_x;

        // Step 2: Compute adjusted outcome Y_adj = Y - theta * (X - X_bar)
        let y_adj: Array1<f64> = y
            .iter()
            .zip(x.iter())
            .map(|(&yi, &xi)| yi - theta * (xi - x_mean))
            .collect();

        // Step 3: Compute treatment effect (difference in means of Y_adj)
        let y_adj_treated: f64 = (0..n)
            .filter(|&i| treated[i])
            .map(|i| y_adj[i])
            .sum::<f64>()
            / n_treatment as f64;
        let y_adj_control: f64 = (0..n)
            .filter(|&i| !treated[i])
            .map(|i| y_adj[i])
            .sum::<f64>()
            / n_control as f64;

        let treatment_effect = y_adj_treated - y_adj_control;

        // Step 4: Standard error (Welch's)
        let y_adj_treated_var: f64 = {
            let vals: Vec<f64> = (0..n).filter(|&i| treated[i]).map(|i| y_adj[i]).collect();
            let m = vals.iter().sum::<f64>() / n_treatment as f64;
            vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n_treatment - 1).max(1) as f64
        };
        let y_adj_control_var: f64 = {
            let vals: Vec<f64> = (0..n).filter(|&i| !treated[i]).map(|i| y_adj[i]).collect();
            let m = vals.iter().sum::<f64>() / n_control as f64;
            vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n_control - 1).max(1) as f64
        };

        let se =
            (y_adj_treated_var / n_treatment as f64 + y_adj_control_var / n_control as f64).sqrt();

        // Unadjusted (naive DiD)
        let y_treated: f64 =
            (0..n).filter(|&i| treated[i]).map(|i| y[i]).sum::<f64>() / n_treatment as f64;
        let y_control: f64 =
            (0..n).filter(|&i| !treated[i]).map(|i| y[i]).sum::<f64>() / n_control as f64;
        let unadjusted_effect = y_treated - y_control;

        let y_treated_var: f64 = {
            let vals: Vec<f64> = (0..n).filter(|&i| treated[i]).map(|i| y[i]).collect();
            let m = vals.iter().sum::<f64>() / n_treatment as f64;
            vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n_treatment - 1).max(1) as f64
        };
        let y_control_var: f64 = {
            let vals: Vec<f64> = (0..n).filter(|&i| !treated[i]).map(|i| y[i]).collect();
            let m = vals.iter().sum::<f64>() / n_control as f64;
            vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n_control - 1).max(1) as f64
        };
        let unadjusted_se =
            (y_treated_var / n_treatment as f64 + y_control_var / n_control as f64).sqrt();

        // Variance reduction
        let adjusted_variance =
            y_adj_treated_var / n_treatment as f64 + y_adj_control_var / n_control as f64;
        let unadjusted_variance =
            y_treated_var / n_treatment as f64 + y_control_var / n_control as f64;
        let variance_reduction = if unadjusted_variance > 1e-15 {
            1.0 - adjusted_variance / unadjusted_variance
        } else {
            0.0
        };

        // t-stat, p-value, CI
        let t_stat = if se > 1e-10 {
            treatment_effect / se
        } else {
            0.0
        };
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_value = 2.0 * (1.0 - normal.cdf(t_stat.abs()));
        let z_crit = 1.959964; // 97.5th percentile of standard normal
        let ci = [
            treatment_effect - z_crit * se,
            treatment_effect + z_crit * se,
        ];

        Ok(CupedResult {
            treatment_effect,
            se,
            t_stat,
            p_value,
            ci,
            theta,
            unadjusted_effect,
            unadjusted_se,
            variance_reduction,
            adjusted_variance,
            unadjusted_variance,
            n_treatment,
            n_control,
            n_obs: n,
        })
    }

    /// Estimate CUPED with a matrix of pre-treatment covariates.
    /// Uses OLS regression of Y on X to compute the adjustment.
    ///
    /// # Arguments
    /// * `y` - Post-treatment outcome (n)
    /// * `x` - Pre-treatment covariates (n x k)
    /// * `treated` - Treatment indicator (n)
    pub fn fit_multivariate(
        y: &Array1<f64>,
        x: &Array2<f64>,
        treated: &[bool],
    ) -> Result<CupedResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if x.nrows() != n || treated.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "CUPED: dimension mismatch".into(),
            ));
        }

        // OLS: Y = X * beta + eps
        // Add intercept
        let mut x_full = Array2::zeros((n, k + 1));
        for i in 0..n {
            x_full[(i, 0)] = 1.0;
            for j in 0..k {
                x_full[(i, j + 1)] = x[(i, j)];
            }
        }

        use crate::linalg::LinalgInverse as _;
        let xt = x_full.t();
        let xtx = xt.dot(&x_full);
        let xtx_inv = (&xtx + Array2::<f64>::eye(k + 1) * 1e-8).inv()?;
        let xty = xt.dot(y);
        let beta: Array1<f64> = xtx_inv.dot(&xty);

        // Adjusted outcome: Y_adj = Y - X * beta (residuals + intercept)
        let y_pred = x_full.dot(&beta);
        let y_adj = y - &y_pred;

        // The treatment effect is the difference in means of Y_adj
        // (since X is pre-treatment, E[X|treated] = E[X|control] in expectation)
        let n_treatment = treated.iter().filter(|&&t| t).count();
        let n_control = n - n_treatment;
        if n_treatment == 0 || n_control == 0 {
            return Err(GreenersError::InvalidOperation(
                "CUPED: need both treatment and control groups".into(),
            ));
        }

        let y_adj_treated: f64 = (0..n)
            .filter(|&i| treated[i])
            .map(|i| y_adj[i])
            .sum::<f64>()
            / n_treatment as f64;
        let y_adj_control: f64 = (0..n)
            .filter(|&i| !treated[i])
            .map(|i| y_adj[i])
            .sum::<f64>()
            / n_control as f64;
        let treatment_effect = y_adj_treated - y_adj_control;

        // SE
        let y_adj_treated_var: f64 = {
            let vals: Vec<f64> = (0..n).filter(|&i| treated[i]).map(|i| y_adj[i]).collect();
            let m = vals.iter().sum::<f64>() / n_treatment as f64;
            vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n_treatment - 1).max(1) as f64
        };
        let y_adj_control_var: f64 = {
            let vals: Vec<f64> = (0..n).filter(|&i| !treated[i]).map(|i| y_adj[i]).collect();
            let m = vals.iter().sum::<f64>() / n_control as f64;
            vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n_control - 1).max(1) as f64
        };
        let se =
            (y_adj_treated_var / n_treatment as f64 + y_adj_control_var / n_control as f64).sqrt();

        // Unadjusted
        let y_treated: f64 =
            (0..n).filter(|&i| treated[i]).map(|i| y[i]).sum::<f64>() / n_treatment as f64;
        let y_control: f64 =
            (0..n).filter(|&i| !treated[i]).map(|i| y[i]).sum::<f64>() / n_control as f64;
        let unadjusted_effect = y_treated - y_control;
        let y_treated_var: f64 = {
            let vals: Vec<f64> = (0..n).filter(|&i| treated[i]).map(|i| y[i]).collect();
            let m = vals.iter().sum::<f64>() / n_treatment as f64;
            vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n_treatment - 1).max(1) as f64
        };
        let y_control_var: f64 = {
            let vals: Vec<f64> = (0..n).filter(|&i| !treated[i]).map(|i| y[i]).collect();
            let m = vals.iter().sum::<f64>() / n_control as f64;
            vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n_control - 1).max(1) as f64
        };
        let unadjusted_se =
            (y_treated_var / n_treatment as f64 + y_control_var / n_control as f64).sqrt();

        let adjusted_variance =
            y_adj_treated_var / n_treatment as f64 + y_adj_control_var / n_control as f64;
        let unadjusted_variance =
            y_treated_var / n_treatment as f64 + y_control_var / n_control as f64;
        let variance_reduction = if unadjusted_variance > 1e-15 {
            1.0 - adjusted_variance / unadjusted_variance
        } else {
            0.0
        };

        // theta = first non-intercept beta (for reporting)
        let theta = beta[1];

        let t_stat = if se > 1e-10 {
            treatment_effect / se
        } else {
            0.0
        };
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_value = 2.0 * (1.0 - normal.cdf(t_stat.abs()));
        let z_crit = 1.959964;
        let ci = [
            treatment_effect - z_crit * se,
            treatment_effect + z_crit * se,
        ];

        Ok(CupedResult {
            treatment_effect,
            se,
            t_stat,
            p_value,
            ci,
            theta,
            unadjusted_effect,
            unadjusted_se,
            variance_reduction,
            adjusted_variance,
            unadjusted_variance,
            n_treatment,
            n_control,
            n_obs: n,
        })
    }
}
