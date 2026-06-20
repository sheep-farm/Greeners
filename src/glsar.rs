use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use crate::{CovarianceType, DataFrame, Formula, OLS};
use ndarray::{Array1, Array2};
use std::fmt;

/// Result of GLSAR estimation.
#[derive(Debug)]
pub struct GlsarResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub r_squared: f64,
    pub rho: Array1<f64>,
    pub n_iter: usize,
    pub converged: bool,
    pub n_obs: usize,
    pub df_resid: usize,
    pub variable_names: Option<Vec<String>>,
}

impl GlsarResult {
    pub fn predict(&self, x_new: &Array2<f64>) -> Array1<f64> {
        x_new.dot(&self.params)
    }

    pub fn residuals(&self, y: &Array1<f64>, x: &Array2<f64>) -> Array1<f64> {
        y - &x.dot(&self.params)
    }

    pub fn fitted_values(&self, x: &Array2<f64>) -> Array1<f64> {
        x.dot(&self.params)
    }
}

impl fmt::Display for GlsarResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", format!(" GLSAR (AR({})) ", self.rho.len()))?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10.4}", "R-squared:", self.r_squared)?;
        writeln!(
            f,
            "{:<20} {:>10}",
            "Converged:",
            if self.converged { "Yes" } else { "No" }
        )?;
        writeln!(f, "{:<20} {:>10}", "Iterations:", self.n_iter)?;
        write!(f, "{:<20}", "AR coefficients:")?;
        for (i, &r) in self.rho.iter().enumerate() {
            write!(f, " rho[{}]={:.4}", i + 1, r)?;
        }
        writeln!(f)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} | {:>10} | {:>10} | {:>8} | {:>8}",
            "Variable", "coef", "std err", "t", "P>|t|"
        )?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.params.len() {
            let name = self
                .variable_names
                .as_ref()
                .and_then(|n| n.get(i).cloned())
                .unwrap_or_else(|| format!("x{}", i));
            writeln!(
                f,
                "{:<12} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3}",
                name, self.params[i], self.std_errors[i], self.t_values[i], self.p_values[i]
            )?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

/// GLS with AR errors (Cochrane-Orcutt generalized to AR(p)).
pub struct GLSAR;

impl GLSAR {
    pub fn from_formula(
        formula: &Formula,
        data: &DataFrame,
        ar_order: usize,
        max_iter: usize,
    ) -> Result<GlsarResult, GreenersError> {
        let (y, x) = data.to_design_matrix(formula)?;
        let var_names = data.formula_var_names(formula)?;
        Self::fit_with_names(&y, &x, ar_order, max_iter, Some(var_names))
    }

    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        ar_order: usize,
        max_iter: usize,
    ) -> Result<GlsarResult, GreenersError> {
        Self::fit_with_names(y, x, ar_order, max_iter, None)
    }

    pub fn fit_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        ar_order: usize,
        max_iter: usize,
        variable_names: Option<Vec<String>>,
    ) -> Result<GlsarResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();

        if n != x.nrows() {
            return Err(GreenersError::ShapeMismatch(
                "y and x row count mismatch".into(),
            ));
        }
        if ar_order == 0 {
            return Err(GreenersError::InvalidOperation(
                "AR order must be >= 1".into(),
            ));
        }
        if n <= ar_order + k {
            return Err(GreenersError::InvalidOperation(
                "Not enough observations for AR order".into(),
            ));
        }

        let tol = 1e-8;

        // 1. Initial OLS
        let initial_ols = OLS::fit(y, x, CovarianceType::NonRobust)?;
        let mut params = initial_ols.params.clone();
        let mut rho = Array1::<f64>::zeros(ar_order);
        let mut converged = false;
        let mut n_iter = 0;

        for iter in 0..max_iter {
            n_iter = iter + 1;

            // 2. Residuals from current params
            let resid = y - &x.dot(&params);

            // 3. Estimate AR coefficients from residuals via OLS
            let ar_n = n - ar_order;
            let mut ar_y = Array1::<f64>::zeros(ar_n);
            let mut ar_x = Array2::<f64>::zeros((ar_n, ar_order));
            for t in ar_order..n {
                ar_y[t - ar_order] = resid[t];
                for j in 0..ar_order {
                    ar_x[[t - ar_order, j]] = resid[t - 1 - j];
                }
            }

            // Simple OLS for AR coefficients (no intercept needed, but OLS adds one if x has it)
            // Direct: rho = (ar_x' ar_x)^-1 ar_x' ar_y
            let ata = ar_x.t().dot(&ar_x);
            let aty = ar_x.t().dot(&ar_y);
            let new_rho = match ata.inv() {
                Ok(inv) => {
                    let candidate = inv.dot(&aty);
                    if candidate.iter().all(|v| v.is_finite()) {
                        candidate
                    } else {
                        rho.clone()
                    }
                }
                Err(_) => rho.clone(),
            };

            // 4. Transform y and X using AR filter
            let start = ar_order;
            let tn = n - start;
            let mut y_star = Array1::<f64>::zeros(tn);
            let mut x_star = Array2::<f64>::zeros((tn, k));

            for t in start..n {
                let ti = t - start;
                y_star[ti] = y[t];
                x_star.row_mut(ti).assign(&x.row(t).to_owned());
                for j in 0..ar_order {
                    y_star[ti] -= new_rho[j] * y[t - 1 - j];
                    let x_lag = x.row(t - 1 - j).to_owned();
                    for c in 0..k {
                        x_star[[ti, c]] -= new_rho[j] * x_lag[c];
                    }
                }
            }

            // 5. OLS on transformed data
            let ols = OLS::fit(&y_star, &x_star, CovarianceType::NonRobust)?;

            let diff = (&ols.params - &params)
                .iter()
                .map(|d| d.abs())
                .fold(0.0_f64, f64::max);
            let param_scale = params.iter().map(|p| p.abs()).fold(1.0_f64, f64::max);

            params = ols.params;
            rho = new_rho;

            if diff / param_scale < tol {
                converged = true;
                break;
            }
        }

        // Final OLS on transformed data for standard errors
        let start = ar_order;
        let tn = n - start;
        let mut y_star = Array1::<f64>::zeros(tn);
        let mut x_star = Array2::<f64>::zeros((tn, k));
        for t in start..n {
            let ti = t - start;
            y_star[ti] = y[t];
            x_star.row_mut(ti).assign(&x.row(t).to_owned());
            for j in 0..ar_order {
                y_star[ti] -= rho[j] * y[t - 1 - j];
                let x_lag = x.row(t - 1 - j).to_owned();
                for c in 0..k {
                    x_star[[ti, c]] -= rho[j] * x_lag[c];
                }
            }
        }

        let final_ols = OLS::fit(&y_star, &x_star, CovarianceType::NonRobust)?;
        let df_resid = tn.saturating_sub(k);

        Ok(GlsarResult {
            params: final_ols.params,
            std_errors: final_ols.std_errors,
            t_values: final_ols.t_values,
            p_values: final_ols.p_values,
            r_squared: final_ols.r_squared,
            rho,
            n_iter,
            converged,
            n_obs: n,
            df_resid,
            variable_names,
        })
    }
}
