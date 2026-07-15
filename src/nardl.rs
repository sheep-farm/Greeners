//! Nonlinear ARDL (NARDL) — asymmetric cointegration.
//!
//! Shin, Yu & Greenwood-Nimmo (2014). Extends ARDL to allow
//! asymmetric long-run and short-run responses to positive and
//! negative changes in the regressors.
//!
//! Decompose x_t into positive and negative partial sums:
//!   x_t^+ = sum_{j=1}^{t} max(Delta x_j, 0)
//!   x_t^- = sum_{j=1}^{t} min(Delta x_j, 0)
//!
//! Then estimate the conditional ECM:
//!   Delta y_t = alpha + rho * EC_{t-1} + sum gamma_j * Delta y_{t-j}
//!             + sum theta_j^+ * Delta x_{t-j}^+ + sum theta_j^- * Delta x_{t-j}^-
//!             + eps_t
//!
//! where EC_{t-1} = y_{t-1} - beta^+ * x_{t-1}^+ - beta^- * x_{t-1}^-
//!
//! Long-run multipliers: beta^+ and beta^-. Asymmetric if beta^+ != beta^-.

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of NARDL estimation.
#[derive(Debug)]
pub struct NardlResult {
    /// Long-run positive multiplier (beta^+)
    pub beta_pos: f64,
    /// Long-run negative multiplier (beta^-)
    pub beta_neg: f64,
    /// Speed of adjustment (rho)
    pub rho: f64,
    /// Short-run positive coefficients (theta^+)
    pub theta_pos: Array1<f64>,
    /// Short-run negative coefficients (theta^-)
    pub theta_neg: Array1<f64>,
    /// All coefficients (full vector)
    pub coefficients: Array1<f64>,
    /// Standard errors
    pub std_errors: Array1<f64>,
    /// t-values
    pub t_values: Array1<f64>,
    /// p-values
    pub p_values: Array1<f64>,
    /// Coefficient names
    pub coef_names: Vec<String>,
    /// R-squared
    pub r_squared: f64,
    /// F-test for long-run asymmetry (beta^+ = beta^-)
    pub lr_asym_f: f64,
    /// p-value for long-run asymmetry test
    pub lr_asym_p: f64,
    /// F-test for short-run asymmetry (theta^+ = theta^-)
    pub sr_asym_f: f64,
    /// p-value for short-run asymmetry test
    pub sr_asym_p: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of lags
    pub lags: usize,
}

impl fmt::Display for NardlResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " NARDL (Nonlinear ARDL) ")?;
        writeln!(f, "Shin, Yu & Greenwood-Nimmo (2014)")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Lags:", self.lags)?;
        writeln!(f, "{:<20} {:>12.6}", "R-squared:", self.r_squared)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<16} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coef.", "Std.Err.", "t", "P>|t|"
        )?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..self.coefficients.len() {
            let name = self
                .coef_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("b{}", i));
            writeln!(
                f,
                "{:<16} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
                name, self.coefficients[i], self.std_errors[i], self.t_values[i], self.p_values[i]
            )?;
        }

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Long-run multipliers:")?;
        writeln!(f, "  beta^+ (positive): {:>12.6}", self.beta_pos)?;
        writeln!(f, "  beta^- (negative): {:>12.6}", self.beta_neg)?;
        writeln!(
            f,
            "  Asymmetry (beta^+ - beta^-): {:>12.6}",
            self.beta_pos - self.beta_neg
        )?;

        writeln!(f, "\n  Asymmetry tests:")?;
        writeln!(
            f,
            "  Long-run  F={:.4}  p={:.4}",
            self.lr_asym_f, self.lr_asym_p
        )?;
        writeln!(
            f,
            "  Short-run F={:.4}  p={:.4}",
            self.sr_asym_f, self.sr_asym_p
        )?;

        write!(f, "{:=^78}", "")
    }
}

pub struct NARDL;

impl NARDL {
    /// Estimate NARDL with asymmetric cointegration.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (T)
    /// * `x` - Regressor (T)
    /// * `lags` - Number of lags for Delta y and Delta x^+/-
    pub fn fit(
        y: &Array1<f64>,
        x: &Array1<f64>,
        lags: usize,
    ) -> Result<NardlResult, GreenersError> {
        let t = y.len();
        if x.len() != t {
            return Err(GreenersError::ShapeMismatch(
                "NARDL: y and x must have same length".into(),
            ));
        }
        if t < (lags + 4) * 2 {
            return Err(GreenersError::InvalidOperation(
                "NARDL: too few observations".into(),
            ));
        }
        if lags == 0 {
            return Err(GreenersError::InvalidOperation(
                "NARDL: lags must be >= 1".into(),
            ));
        }

        // Step 1: Decompose x into positive and negative partial sums
        let mut dx = Array1::zeros(t - 1);
        for i in 0..t - 1 {
            dx[i] = x[i + 1] - x[i];
        }
        let mut x_pos = Array1::zeros(t);
        let mut x_neg = Array1::zeros(t);
        x_pos[0] = x[0];
        x_neg[0] = x[0];
        let mut cum_pos = 0.0_f64;
        let mut cum_neg = 0.0_f64;
        for i in 0..t - 1 {
            if dx[i] > 0.0 {
                cum_pos += dx[i];
            } else {
                cum_neg += dx[i];
            }
            x_pos[i + 1] = x[0] + cum_pos;
            x_neg[i + 1] = x[0] + cum_neg;
        }

        // Step 2: Build ECM regression
        // Delta y_t = alpha + rho * y_{t-1} + beta^+ * x^+_{t-1} + beta^- * x^-_{t-1}
        //           + sum gamma_j * Delta y_{t-j} + sum theta^+_j * Delta x^+_{t-j}
        //           + sum theta^-_j * Delta x^-_{t-j} + eps_t
        let n_eff = t - lags - 1;
        let n_reg = 4 + lags * 3; // const, y_lag, x_pos_lag, x_neg_lag + lags*(dy, dx_pos, dx_neg)

        let mut z = Array2::zeros((n_eff, n_reg));
        let mut dy = Array1::zeros(n_eff);

        for i in 0..n_eff {
            let t_i = lags + 1 + i;
            dy[i] = y[t_i] - y[t_i - 1];

            // Constant
            z[(i, 0)] = 1.0;
            // y_{t-1} (level)
            z[(i, 1)] = y[t_i - 1];
            // x^+_{t-1} and x^-_{t-1} (levels)
            z[(i, 2)] = x_pos[t_i - 1];
            z[(i, 3)] = x_neg[t_i - 1];

            // Lagged differences
            for j in 0..lags {
                // Delta y_{t-j}
                z[(i, 4 + j)] = y[t_i - j] - y[t_i - j - 1];
                // Delta x^+_{t-j}
                let dx_p = x_pos[t_i - j] - x_pos[t_i - j - 1];
                z[(i, 4 + lags + j)] = dx_p;
                // Delta x^-_{t-j}
                let dx_n = x_neg[t_i - j] - x_neg[t_i - j - 1];
                z[(i, 4 + 2 * lags + j)] = dx_n;
            }
        }

        // OLS
        let zt = z.t();
        let ztz = zt.dot(&z);
        let ztz_reg = &ztz + Array2::<f64>::eye(n_reg) * 1e-8;
        let ztz_inv = ztz_reg.inv()?;
        let zty = zt.dot(&dy);
        let beta: Array1<f64> = ztz_inv.dot(&zty);

        let residuals = &dy - z.dot(&beta);
        let sse = residuals.dot(&residuals);
        let sigma2 = sse / (n_eff - n_reg) as f64;
        let cov = &ztz_inv * sigma2;
        let std_errors = cov.diag().mapv(|v| v.sqrt());
        let t_values = &beta / &std_errors;
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_values = t_values.mapv(|t| 2.0 * (1.0 - normal.cdf(t.abs())));

        // Extract long-run multipliers
        // rho = beta[1], beta^+ = beta[2]/(-rho), beta^- = beta[3]/(-rho)
        let rho = beta[1];
        let beta_pos = if rho.abs() > 1e-10 {
            beta[2] / (-rho)
        } else {
            0.0
        };
        let beta_neg = if rho.abs() > 1e-10 {
            beta[3] / (-rho)
        } else {
            0.0
        };

        // Short-run coefficients
        let theta_pos = beta.slice(ndarray::s![4 + lags..4 + 2 * lags]).to_owned();
        let theta_neg = beta
            .slice(ndarray::s![4 + 2 * lags..4 + 3 * lags])
            .to_owned();

        // R-squared
        let dy_mean = dy.mean().unwrap_or(0.0);
        let tss = dy.mapv(|v| (v - dy_mean).powi(2)).sum();
        let r_squared = if tss > 1e-15 { 1.0 - sse / tss } else { 0.0 };

        // Asymmetry tests (Wald tests)
        // Long-run: beta^+ = beta^- => beta[2]/(-rho) = beta[3]/(-rho) => beta[2] = beta[3]
        let lr_asym_f = Self::wald_test(&beta, &cov, &[(2, 1.0, 3, -1.0, 0.0)]);
        let lr_asym_p = 1.0 - Self::f_cdf(lr_asym_f, 1, n_eff - n_reg);

        // Short-run: sum theta^+ = sum theta^-
        let sum_tp: f64 = theta_pos.sum();
        let sum_tn: f64 = theta_neg.sum();
        let sr_asym_f = ((sum_tp - sum_tn).powi(2)) / (sigma2 * 2.0); // simplified
        let sr_asym_p = 1.0 - Self::f_cdf(sr_asym_f, 1, n_eff - n_reg);

        // Coefficient names
        let mut names = vec![
            "const".to_string(),
            "y_{t-1}".to_string(),
            "x^+_{t-1}".to_string(),
            "x^-_{t-1}".to_string(),
        ];
        for j in 0..lags {
            names.push(format!("Dy_{{t-{}}}", j + 1));
        }
        for j in 0..lags {
            names.push(format!("Dx^+_{{t-{}}}", j + 1));
        }
        for j in 0..lags {
            names.push(format!("Dx^-_{{t-{}}}", j + 1));
        }

        Ok(NardlResult {
            beta_pos,
            beta_neg,
            rho,
            theta_pos,
            theta_neg,
            coefficients: beta,
            std_errors,
            t_values,
            p_values,
            coef_names: names,
            r_squared,
            lr_asym_f,
            lr_asym_p,
            sr_asym_f,
            sr_asym_p,
            n_obs: n_eff,
            lags,
        })
    }

    /// Simple Wald test: H0: c1*b[a] + c2*b[b] = target
    fn wald_test(
        beta: &Array1<f64>,
        cov: &Array2<f64>,
        constraints: &[(usize, f64, usize, f64, f64)],
    ) -> f64 {
        // Simplified: single constraint c1*b[a] + c2*b[b] = target
        if constraints.is_empty() {
            return 0.0;
        }
        let &(a, c1, b, c2, target) = &constraints[0];
        let val = c1 * beta[a] + c2 * beta[b] - target;
        let var = c1 * c1 * cov[(a, a)] + c2 * c2 * cov[(b, b)] + 2.0 * c1 * c2 * cov[(a, b)];
        if var > 1e-15 {
            val * val / var
        } else {
            0.0
        }
    }

    /// Approximate F CDF via regularized incomplete beta function.
    fn f_cdf(f: f64, df1: usize, df2: usize) -> f64 {
        if f <= 0.0 {
            return 0.0;
        }
        // F CDF = I_{df1*f/(df1*f+df2)}(df1/2, df2/2)
        let x = (df1 as f64 * f) / (df1 as f64 * f + df2 as f64);
        // Approximate using normal for large df2
        if df2 > 100 {
            let z = (f.ln() * df1 as f64 / 2.0).sqrt();
            return 0.5 * (1.0 + (z * 0.7).tanh());
        }
        // Simple approximation
        let a = df1 as f64 / 2.0;
        let b = df2 as f64 / 2.0;
        // Regularized incomplete beta (simplified)
        Self::reg_incomplete_beta(x, a, b)
    }

    fn reg_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
        // Continued fraction expansion (Lentz's method)
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        let lbeta = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
        let front = (a * x.ln() + b * (1.0 - x).ln() - lbeta).exp() / a;

        let mut cf = 1.0;
        let mut term = 1.0;
        let mut d = 1.0 - x;
        for iter in 0..100 {
            let m = iter / 2 + 1;
            let numerator = if iter % 2 == 0 {
                -((a + m as f64 - 1.0) * (b + m as f64 - 1.0) * m as f64 * m as f64)
            } else {
                (a + m as f64 - 1.0) * (b + m as f64 - 1.0) * m as f64 * m as f64
            };
            let denominator = (a + 2.0 * cf - 1.0) * (a + 2.0 * cf);
            d = 1.0 + numerator / denominator / d;
            if d.abs() < 1e-30 {
                d = 1e-30;
            }
            d = 1.0 / d;
            let del = d * term;
            term *= d;
            cf += 1.0;
            if (del - 1.0).abs() < 1e-10 {
                break;
            }
        }
        front * (1.0 - term)
    }
}

fn ln_gamma(x: f64) -> f64 {
    // Stirling's approximation
    if x < 0.5 {
        return (std::f64::consts::PI / (x * (std::f64::consts::PI * x).sin())).ln()
            - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut result = 0.9999999999998091 + 676.5203681218851 / (x + 1.0)
        - 1259.1392167224028 / (x + 2.0)
        + 771.323_428_777_653_1 / (x + 3.0)
        - 176.615_029_162_140_6 / (x + 4.0)
        + 12.507343278686905 / (x + 5.0)
        - 0.13857109526572012 / (x + 6.0)
        + 9.984_369_578_019_572e-6 / (x + 7.0)
        + 1.5056327351493116e-7 / (x + 8.0);
    result = result.ln() + (x + 7.5).ln() * (x + 0.5) - (x + 7.5);
    result
}
