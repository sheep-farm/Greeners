//! Quantile VAR (QVAR) — VAR estimation via quantile regression.
//!
//! Each equation of the VAR is estimated separately via quantile
//! regression (Koenker & Bassett 1978), allowing for asymmetric
//! dynamics across the conditional distribution.
//!
//! y_{j,t} = alpha_j + sum_p A_{j,p} * y_{t-p} + eps_{j,t}
//!
//! where the tau-quantile of eps_{j,t} is zero. This captures
//! quantile-specific dynamics (e.g., recession vs expansion).
//!
//! Estimation: equation-by-equation quantile regression with
//! bootstrap standard errors.

use crate::error::GreenersError;
use crate::quantile::QuantileReg;
use ndarray::{Array1, Array2, Array3};
use std::fmt;

/// Result of Quantile VAR estimation.
#[derive(Debug)]
pub struct QuantileVarResult {
    /// Quantile level
    pub tau: f64,
    /// Coefficients for each equation, shape (k, 1 + k*lags)
    /// Row j = [alpha_j, A_{j,1,1}, ..., A_{j,1,k}, A_{j,2,1}, ...]
    pub coeffs: Array2<f64>,
    /// Standard errors, shape (k, 1 + k*lags)
    pub std_errors: Array2<f64>,
    /// t-values, shape (k, 1 + k*lags)
    pub t_values: Array2<f64>,
    /// p-values, shape (k, 1 + k*lags)
    pub p_values: Array2<f64>,
    /// Pseudo R-squared per equation
    pub pseudo_r2: Array1<f64>,
    /// Number of observations (after lags)
    pub n_obs: usize,
    /// Number of variables
    pub n_vars: usize,
    /// VAR lag order
    pub lags: usize,
    /// Variable names
    pub var_names: Vec<String>,
}

impl fmt::Display for QuantileVarResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            format!(" Quantile VAR (tau={:.2}) ", self.tau)
        )?;
        writeln!(f, "Equation-by-equation quantile regression")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Variables:", self.n_vars)?;
        writeln!(f, "{:<20} {:>12}", "Lags:", self.lags)?;

        for eq in 0..self.n_vars {
            let eq_name = self
                .var_names
                .get(eq)
                .cloned()
                .unwrap_or_else(|| format!("y{}", eq));
            writeln!(f, "\n{:-^78}", format!(" Equation: {} ", eq_name))?;
            writeln!(
                f,
                "{:<14} {:>12} {:>12} {:>10} {:>10}",
                "Variable", "Coef.", "Std.Err.", "t", "P>|t|"
            )?;
            writeln!(f, "{:-^78}", "")?;

            // Intercept
            writeln!(
                f,
                "{:<14} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
                "const",
                self.coeffs[(eq, 0)],
                self.std_errors[(eq, 0)],
                self.t_values[(eq, 0)],
                self.p_values[(eq, 0)]
            )?;

            // AR terms
            for p in 0..self.lags {
                for j in 0..self.n_vars {
                    let var_name = self
                        .var_names
                        .get(j)
                        .cloned()
                        .unwrap_or_else(|| format!("y{}", j));
                    let col = 1 + p * self.n_vars + j;
                    let label = format!("L{}.{}", p + 1, var_name);
                    writeln!(
                        f,
                        "{:<14} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
                        label,
                        self.coeffs[(eq, col)],
                        self.std_errors[(eq, col)],
                        self.t_values[(eq, col)],
                        self.p_values[(eq, col)]
                    )?;
                }
            }
            writeln!(f, "  Pseudo R-squared: {:.6}", self.pseudo_r2[eq])?;
        }
        write!(f, "{:=^78}", "")
    }
}

pub struct QuantileVAR;

impl QuantileVAR {
    /// Estimate Quantile VAR via equation-by-equation quantile regression.
    ///
    /// # Arguments
    /// * `y` - Data matrix (T x k)
    /// * `lags` - VAR lag order p
    /// * `tau` - Quantile level (0-1)
    /// * `n_boot` - Number of bootstrap replications for SE
    /// * `var_names` - Optional variable names
    pub fn fit(
        y: &Array2<f64>,
        lags: usize,
        tau: f64,
        n_boot: usize,
        var_names: Option<Vec<String>>,
    ) -> Result<QuantileVarResult, GreenersError> {
        let t = y.nrows();
        let k = y.ncols();
        if t < (lags + 1) * 3 {
            return Err(GreenersError::InvalidOperation(
                "QVAR: too few observations".into(),
            ));
        }
        if lags == 0 {
            return Err(GreenersError::InvalidOperation(
                "QVAR: lags must be >= 1".into(),
            ));
        }
        if tau <= 0.0 || tau >= 1.0 {
            return Err(GreenersError::InvalidOperation(
                "QVAR: tau must be in (0, 1)".into(),
            ));
        }

        let names = var_names.unwrap_or_else(|| (0..k).map(|i| format!("y{}", i)).collect());
        let n_eff = t - lags;
        let n_reg = 1 + k * lags;

        // Build design matrix Z (n_eff x n_reg) — same for all equations
        let mut z = Array2::zeros((n_eff, n_reg));
        for i in 0..n_eff {
            let t_i = lags + i;
            z[(i, 0)] = 1.0;
            for p in 0..lags {
                for j in 0..k {
                    z[(i, 1 + p * k + j)] = y[(t_i - 1 - p, j)];
                }
            }
        }

        // Estimate each equation via quantile regression
        let mut coeffs = Array2::zeros((k, n_reg));
        let mut std_errors = Array2::zeros((k, n_reg));
        let mut t_values = Array2::zeros((k, n_reg));
        let mut p_values = Array2::zeros((k, n_reg));
        let mut pseudo_r2 = Array1::zeros(k);

        for eq in 0..k {
            // Dependent variable for this equation
            let mut y_eq = Array1::zeros(n_eff);
            for i in 0..n_eff {
                let t_i = lags + i;
                y_eq[i] = y[(t_i, eq)];
            }

            let result = QuantileReg::fit(&y_eq, &z, tau, n_boot)
                .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;

            for col in 0..n_reg {
                coeffs[(eq, col)] = result.params[col];
                std_errors[(eq, col)] = result.std_errors[col];
                t_values[(eq, col)] = result.t_values[col];
                p_values[(eq, col)] = result.p_values[col];
            }
            pseudo_r2[eq] = result.r_squared;
        }

        Ok(QuantileVarResult {
            tau,
            coeffs,
            std_errors,
            t_values,
            p_values,
            pseudo_r2,
            n_obs: n_eff,
            n_vars: k,
            lags,
            var_names: names,
        })
    }

    /// Compute impulse response function from QVAR coefficients.
    /// Returns Array3 of shape (steps x k x k).
    pub fn irf(result: &QuantileVarResult, steps: usize) -> Array3<f64> {
        let k = result.n_vars;
        let p = result.lags;

        // Build companion matrix
        let mut phi = Array2::zeros((k * p, k * p));
        for eq in 0..k {
            for lag in 0..p {
                for j in 0..k {
                    phi[(eq, lag * k + j)] = result.coeffs[(eq, 1 + lag * k + j)];
                }
            }
        }
        if p > 1 {
            for lag in 1..p {
                for j in 0..k {
                    phi[(lag * k + j, (lag - 1) * k + j)] = 1.0;
                }
            }
        }

        let mut irf = Array3::zeros((steps, k, k));
        let mut phi_power = Array2::eye(k * p);
        for h in 0..steps {
            let phi_h = phi_power.slice(ndarray::s![0..k, 0..k]).to_owned();
            for i in 0..k {
                for j in 0..k {
                    irf[(h, i, j)] = phi_h[(i, j)];
                }
            }
            phi_power = phi.dot(&phi_power);
        }

        irf
    }
}
