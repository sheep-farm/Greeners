//! Conformal Prediction (Vovk, Gammerman & Shafer 2005).
//!
//! Provides distribution-free prediction intervals with finite-
//! sample coverage guarantees. Works with any point predictor
//! (e.g., OLS, random forest, gradient boosting).
//!
//! Method: Split conformal prediction
//! 1. Split data into training (n1) and calibration (n2)
//! 2. Fit predictor on training set
//! 3. Compute nonconformity scores on calibration:
//!    s_i = |y_i - y_hat_i|  (absolute residual)
//! 4. For a new x, the prediction interval at level 1-alpha is:
//!    [y_hat(x) - q, y_hat(x) + q]
//!    where q = ceil((n2+1)*(1-alpha)/n2)-th order statistic of s
//!
//! Guarantee: P(y_new in interval) >= 1-alpha (exchangeability).
//!
//! This implementation uses OLS as the base predictor, but the
//! conformal wrapper is model-agnostic.

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// Result of Conformal Prediction.
#[derive(Debug)]
pub struct ConformalResult {
    /// Point predictions (n_test)
    pub predictions: Array1<f64>,
    /// Lower bound of prediction intervals (n_test)
    pub lower: Array1<f64>,
    /// Upper bound of prediction intervals (n_test)
    pub upper: Array1<f64>,
    /// Conformal quantile (half-width)
    pub quantile: f64,
    /// Miscoverage level alpha
    pub alpha: f64,
    /// Coverage level 1-alpha
    pub coverage: f64,
    /// Calibration nonconformity scores
    pub scores: Vec<f64>,
    /// Number of training samples
    pub n_train: usize,
    /// Number of calibration samples
    pub n_calib: usize,
    /// Number of test samples
    pub n_test: usize,
    /// Empirical coverage on calibration set
    pub empirical_coverage: f64,
    /// OLS coefficients
    pub coefficients: Array1<f64>,
    /// Variable names
    pub variable_names: Vec<String>,
}

impl fmt::Display for ConformalResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Conformal Prediction ")?;
        writeln!(f, "Vovk, Gammerman & Shafer (2005)")?;
        writeln!(f, "Split conformal with OLS base predictor")?;
        writeln!(f, "{:<20} {:>12}", "Training samples:", self.n_train)?;
        writeln!(f, "{:<20} {:>12}", "Calibration samples:", self.n_calib)?;
        writeln!(f, "{:<20} {:>12}", "Test samples:", self.n_test)?;
        writeln!(f, "{:<20} {:>12.4}", "Coverage level:", self.coverage)?;
        writeln!(f, "{:<20} {:>12.4}", "Miscoverage (alpha):", self.alpha)?;
        writeln!(f, "{:<20} {:>12.6}", "Conformal quantile:", self.quantile)?;
        writeln!(
            f,
            "{:<20} {:>12.4}",
            "Empirical coverage:", self.empirical_coverage
        )?;

        // Prediction intervals
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Prediction intervals (first 5 test obs):")?;
        writeln!(
            f,
            "  {:<6} {:>12} {:>14} {:>14}",
            "Obs", "Predicted", "Lower", "Upper"
        )?;
        writeln!(f, "{:-^78}", "")?;
        let n_show = 5.min(self.n_test);
        for i in 0..n_show {
            writeln!(
                f,
                "  {:<6} {:>12.4} {:>14.4} {:>14.4}",
                i + 1,
                self.predictions[i],
                self.lower[i],
                self.upper[i]
            )?;
        }

        // Calibration score summary
        writeln!(f, "\n  Calibration nonconformity scores:")?;
        let mut sorted_scores = self.scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted_scores.len();
        if n > 0 {
            writeln!(
                f,
                "  Min: {:>10.4}  Median: {:>10.4}  Max: {:>10.4}",
                sorted_scores[0],
                sorted_scores[n / 2],
                sorted_scores[n - 1]
            )?;
        }

        // Coefficients
        writeln!(f, "\n  OLS coefficients:")?;
        writeln!(f, "  {:<14} {:>12}", "Variable", "Coef")?;
        writeln!(f, "{:-^78}", "")?;
        writeln!(f, "  {:<14} {:>12.6}", "Intercept", self.coefficients[0])?;
        for (j, name) in self.variable_names.iter().enumerate() {
            if j + 1 < self.coefficients.len() {
                writeln!(f, "  {:<14} {:>12.6}", name, self.coefficients[j + 1])?;
            }
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct ConformalPrediction;

impl ConformalPrediction {
    /// Estimate conformal prediction intervals.
    ///
    /// # Arguments
    /// * `y` - Outcome (n)
    /// * `x` - Features (n x k)
    /// * `x_test` - Test features (n_test x k)
    /// * `alpha` - Miscoverage level (default 0.1 for 90% coverage)
    /// * `calib_fraction` - Fraction of data for calibration (default 0.3)
    /// * `variable_names` - Optional feature names
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        x_test: &Array2<f64>,
        alpha: Option<f64>,
        calib_fraction: Option<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<ConformalResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        let n_test = x_test.nrows();
        if x.nrows() != n {
            return Err(GreenersError::ShapeMismatch(
                "ConformalPrediction: x and y dimension mismatch".into(),
            ));
        }
        if x_test.ncols() != k {
            return Err(GreenersError::ShapeMismatch(
                "ConformalPrediction: x_test must have same ncols as x".into(),
            ));
        }
        if n < 10 {
            return Err(GreenersError::InvalidOperation(
                "ConformalPrediction: need at least 10 observations".into(),
            ));
        }

        let alpha_val = alpha.unwrap_or(0.1).clamp(0.01, 0.5);
        let coverage = 1.0 - alpha_val;
        let calib_frac = calib_fraction.unwrap_or(0.3).clamp(0.1, 0.5);
        let names = variable_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());

        // Split into training and calibration
        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..n {
            let j = i + Self::rand_int(n - i);
            indices.swap(i, j);
        }
        let n_calib = (n as f64 * calib_frac).round() as usize;
        let n_calib = n_calib.max(5).min(n - 5);
        let n_train = n - n_calib;

        let train_idx = &indices[..n_train];
        let calib_idx = &indices[n_train..];

        // Fit OLS on training set
        let beta = Self::ols_fit_subset(y, x, train_idx, k)?;

        // Compute nonconformity scores on calibration set
        let mut scores: Vec<f64> = Vec::with_capacity(n_calib);
        let mut n_covered = 0;
        for &i in calib_idx {
            let y_pred = Self::predict_ols(&beta, &x.row(i).to_owned(), k);
            let score = (y[i] - y_pred).abs();
            scores.push(score);
            // Empirical coverage check (using current model)
            // Will compute after quantile
        }

        // Compute conformal quantile
        // q = ceil((n_calib + 1) * (1 - alpha) / n_calib)-th smallest score
        let mut sorted_scores = scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q_rank = (((n_calib + 1) as f64 * coverage / n_calib as f64).ceil()) as usize;
        let q_rank = q_rank.min(n_calib).max(1);
        let quantile = sorted_scores[q_rank - 1];

        // Empirical coverage on calibration set
        for &i in calib_idx {
            let y_pred = Self::predict_ols(&beta, &x.row(i).to_owned(), k);
            if (y[i] - y_pred).abs() <= quantile {
                n_covered += 1;
            }
        }
        let empirical_coverage = n_covered as f64 / n_calib as f64;

        // Predict on test set with intervals
        let mut predictions = Array1::zeros(n_test);
        let mut lower = Array1::zeros(n_test);
        let mut upper = Array1::zeros(n_test);
        for i in 0..n_test {
            let pred = Self::predict_ols(&beta, &x_test.row(i).to_owned(), k);
            predictions[i] = pred;
            lower[i] = pred - quantile;
            upper[i] = pred + quantile;
        }

        Ok(ConformalResult {
            predictions,
            lower,
            upper,
            quantile,
            alpha: alpha_val,
            coverage,
            scores,
            n_train,
            n_calib,
            n_test,
            empirical_coverage,
            coefficients: beta,
            variable_names: names,
        })
    }

    fn ols_fit_subset(
        y: &Array1<f64>,
        x: &Array2<f64>,
        indices: &[usize],
        k: usize,
    ) -> Result<Array1<f64>, GreenersError> {
        let n = indices.len();
        let mut x_full = Array2::zeros((n, k + 1));
        let mut y_sub = Array1::zeros(n);
        for (i, &idx) in indices.iter().enumerate() {
            x_full[(i, 0)] = 1.0;
            for j in 0..k {
                x_full[(i, j + 1)] = x[(idx, j)];
            }
            y_sub[i] = y[idx];
        }
        let xt = x_full.t();
        let xtx = xt.dot(&x_full);
        let xtx_inv = (&xtx + Array2::<f64>::eye(k + 1) * 1e-8).inv()?;
        let xty = xt.dot(&y_sub);
        Ok(xtx_inv.dot(&xty))
    }

    fn predict_ols(beta: &Array1<f64>, x: &Array1<f64>, k: usize) -> f64 {
        let mut pred = beta[0];
        for j in 0..k {
            pred += beta[j + 1] * x[j];
        }
        pred
    }

    fn rand_int(n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (Self::rand_uniform() * n as f64) as usize
    }

    fn rand_uniform() -> f64 {
        use std::cell::Cell;
        thread_local! {
            static STATE: Cell<u64> = const { Cell::new(1732050807) };
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
