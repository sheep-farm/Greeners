//! Double/debiased machine learning (Chernozhukov et al. 2018).
//!
//! Partially linear model: Y = θ·D + g(X) + ε
//! where D is the treatment of interest, X are controls,
//! and g(X) is an unknown nuisance function.
//!
//! Cross-fitting (K-fold) avoids overfitting bias:
//!   1. Split data into K folds.
//!   2. For each fold k, train nuisance models (ĝ, m̂) on the other K-1 folds.
//!   3. Predict on fold k: ĝ(X), m̂(D|X).
//!   4. Compute residualized outcome: Ỹ = Y - ĝ(X)
//!      and residualized treatment: D̃ = D - m̂(X).
//!   5. Estimate θ via OLS of Ỹ on D̃.
//!
//! Nuisance models use OLS with polynomial expansion (degree 2 or 3)
//! as a simple ML proxy. This avoids adding a full ML framework dependency
//! while still demonstrating the cross-fitting orthogonalization.

use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of double/debiased ML estimation.
#[derive(Debug)]
pub struct DoubleMLResult {
    /// Estimated treatment effect (theta)
    pub theta: f64,
    /// Standard error of theta
    pub std_error: f64,
    /// t-statistic
    pub t_value: f64,
    /// p-value (two-sided)
    pub p_value: f64,
    /// 95% confidence interval
    pub ci_low: f64,
    pub ci_high: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of folds used
    pub n_folds: usize,
    /// Residualized outcome (Y - g_hat(X))
    pub y_tilde: Array1<f64>,
    /// Residualized treatment (D - m_hat(X))
    pub d_tilde: Array1<f64>,
}

impl fmt::Display for DoubleMLResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Double/Debiased ML ")?;
        writeln!(f, "Model: Y = theta*D + g(X) + epsilon")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Folds (cross-fitting):", self.n_folds)?;
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<20} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
            "theta (treatment)", self.theta, self.std_error, self.t_value, self.p_value
        )?;
        writeln!(f, "\n95% CI: [{:.6}, {:.6}]", self.ci_low, self.ci_high)?;
        write!(f, "{:=^78}", "")
    }
}

pub struct DoubleML;

impl DoubleML {
    /// Estimate the partially linear model Y = θ·D + g(X) + ε
    /// using double/debiased ML with K-fold cross-fitting.
    ///
    /// # Arguments
    /// * `y` - Outcome variable (n)
    /// * `d` - Treatment variable (n)
    /// * `x` - Controls matrix (n × p)
    /// * `n_folds` - Number of folds for cross-fitting (default 5)
    /// * `poly_degree` - Degree of polynomial expansion for nuisance models
    pub fn fit_plr(
        y: &Array1<f64>,
        d: &Array1<f64>,
        x: &Array2<f64>,
        n_folds: usize,
        poly_degree: usize,
    ) -> Result<DoubleMLResult, GreenersError> {
        let n = y.len();
        if n != d.len() || n != x.nrows() {
            return Err(GreenersError::ShapeMismatch(
                "DoubleML: y, d, x must have same number of rows".into(),
            ));
        }
        if n_folds < 2 {
            return Err(GreenersError::InvalidOperation(
                "DoubleML: n_folds must be >= 2".into(),
            ));
        }

        let k_folds = n_folds.min(n);
        let fold_size = n / k_folds;

        // Build expanded X (polynomial features + intercept)
        let x_expanded = Self::poly_expand(x, poly_degree);

        let mut y_tilde = Array1::zeros(n);
        let mut d_tilde = Array1::zeros(n);

        // K-fold cross-fitting
        for fold in 0..k_folds {
            let start = fold * fold_size;
            let end = if fold == k_folds - 1 {
                n
            } else {
                (fold + 1) * fold_size
            };

            // Training indices: all except [start, end)
            let train_idx: Vec<usize> = (0..start).chain(end..n).collect();
            let test_idx: Vec<usize> = (start..end).collect();

            // Train ĝ(X): OLS of Y on X_expanded (training)
            let g_hat = Self::ols_predict(y, &x_expanded, &train_idx, &test_idx)?;

            // Train m̂(X): OLS of D on X_expanded (training)
            let m_hat = Self::ols_predict(d, &x_expanded, &train_idx, &test_idx)?;

            // Residualize on test fold
            for &i in &test_idx {
                y_tilde[i] = y[i] - g_hat[i];
                d_tilde[i] = d[i] - m_hat[i];
            }
        }

        // Final stage: OLS of Ỹ on D̃ (no intercept — orthogonalized)
        // theta = (D̃'D̃)^{-1} D̃'Ỹ
        let dtd = d_tilde.dot(&d_tilde);
        let dty = d_tilde.dot(&y_tilde);
        if dtd.abs() < 1e-15 {
            return Err(GreenersError::InvalidOperation(
                "DoubleML: residualized treatment has zero variance".into(),
            ));
        }
        let theta = dty / dtd;

        // Residuals and sigma2
        let residuals = &y_tilde - theta * &d_tilde;
        let rss = residuals.dot(&residuals);
        let sigma2 = rss / (n - 1) as f64;
        let se = (sigma2 / dtd).sqrt();

        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let t_value = theta / se;
        let p_value = 2.0 * (1.0 - normal.cdf(t_value.abs()));
        let z = 1.96;
        let ci_low = theta - z * se;
        let ci_high = theta + z * se;

        Ok(DoubleMLResult {
            theta,
            std_error: se,
            t_value,
            p_value,
            ci_low,
            ci_high,
            n_obs: n,
            n_folds: k_folds,
            y_tilde,
            d_tilde,
        })
    }

    /// Polynomial expansion of X: adds intercept, squares, and cross-products.
    fn poly_expand(x: &Array2<f64>, degree: usize) -> Array2<f64> {
        let n = x.nrows();
        let p = x.ncols();

        if degree == 0 {
            return Array2::ones((n, 1));
        }

        // Columns: intercept + linear + squares + cross-products (degree 2)
        let mut cols: Vec<Array1<f64>> = vec![Array1::ones(n)];

        // Linear terms
        for j in 0..p {
            cols.push(x.column(j).to_owned());
        }

        if degree >= 2 {
            // Squares
            for j in 0..p {
                let sq = x.column(j).mapv(|v| v * v);
                cols.push(sq);
            }
            // Cross-products
            for j in 0..p {
                for k in (j + 1)..p {
                    let cross = &x.column(j) * &x.column(k);
                    cols.push(cross);
                }
            }
        }

        if degree >= 3 {
            // Cubes
            for j in 0..p {
                let cube = x.column(j).mapv(|v| v * v * v);
                cols.push(cube);
            }
        }

        let n_cols = cols.len();
        let mut result = Array2::zeros((n, n_cols));
        for (j, col) in cols.iter().enumerate() {
            for i in 0..n {
                result[(i, j)] = col[i];
            }
        }
        result
    }

    /// OLS regression: train on train_idx, predict on test_idx.
    fn ols_predict(
        target: &Array1<f64>,
        x: &Array2<f64>,
        train_idx: &[usize],
        test_idx: &[usize],
    ) -> Result<Array1<f64>, GreenersError> {
        let n_train = train_idx.len();
        let k = x.ncols();

        let mut x_train = Array2::zeros((n_train, k));
        let mut y_train = Array1::zeros(n_train);
        for (i, &idx) in train_idx.iter().enumerate() {
            for j in 0..k {
                x_train[(i, j)] = x[(idx, j)];
            }
            y_train[i] = target[idx];
        }

        let xt = x_train.t();
        let xtx = xt.dot(&x_train);
        let xtx_inv = xtx.inv()?;
        let xty = xt.dot(&y_train);
        let beta: Array1<f64> = xtx_inv.dot(&xty);

        // Predict on test
        let n_test = test_idx.len();
        let mut predictions = Array1::zeros(n_test);
        let n = target.len();
        for (i, &idx) in test_idx.iter().enumerate() {
            let mut pred = 0.0;
            for j in 0..k {
                pred += beta[j] * x[(idx, j)];
            }
            predictions[i] = pred;
        }
        // Expand to full length (only test positions matter)
        let mut full_pred = Array1::zeros(n);
        for (i, &idx) in test_idx.iter().enumerate() {
            full_pred[idx] = predictions[i];
        }
        Ok(full_pred)
    }
}
