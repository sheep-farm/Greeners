use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use crate::{CovarianceType, OLS};
use ndarray::{s, Array1, Array2};
use std::fmt;

// ─── RecursiveLS ───────────────────────────────────────────────────────────────

/// Result of Recursive Least Squares estimation.
#[derive(Debug)]
pub struct RecursiveLSResult {
    /// Parameter estimates at each time step (T x k)
    pub params_history: Array2<f64>,
    /// One-step-ahead prediction errors
    pub residuals: Array1<f64>,
    /// CUSUM statistic
    pub cusum: Array1<f64>,
    /// CUSUM of squares
    pub cusum_squares: Array1<f64>,
    /// Final parameter estimates
    pub params: Array1<f64>,
    pub n_obs: usize,
}

impl fmt::Display for RecursiveLSResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Recursive Least Squares ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(
            f,
            "{:<20} {:>10}",
            "Parameters:",
            self.params_history.ncols()
        )?;
        writeln!(f, "\nFinal coefficients:")?;
        for (i, &v) in self.params.iter().enumerate() {
            writeln!(f, "  beta[{}] = {:.6}", i, v)?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

/// Recursive Least Squares estimator (Kalman-like sequential update).
pub struct RecursiveLS;

impl RecursiveLS {
    pub fn fit(y: &Array1<f64>, x: &Array2<f64>) -> Result<RecursiveLSResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();

        if n != x.nrows() {
            return Err(GreenersError::ShapeMismatch(
                "y and x row count mismatch".into(),
            ));
        }
        if n <= k {
            return Err(GreenersError::InvalidOperation(
                "Need more observations than parameters".into(),
            ));
        }

        // Initialize with first k observations via OLS
        let x_init = x.slice(s![..k, ..]).to_owned();
        let y_init = y.slice(s![..k]).to_owned();
        let xtx_inv = x_init.t().dot(&x_init).inv()?;
        let mut beta = xtx_inv.dot(&x_init.t().dot(&y_init));
        let mut p_mat = xtx_inv;

        let mut params_history = Array2::<f64>::zeros((n, k));
        let mut residuals = Array1::<f64>::zeros(n);

        // Store initial estimates
        for t in 0..k {
            params_history.row_mut(t).assign(&beta);
        }

        // Recursive update from observation k onwards
        for t in k..n {
            let xt = x.row(t).to_owned();
            let yt = y[t];

            // Prediction error
            let y_pred = xt.dot(&beta);
            let e = yt - y_pred;
            residuals[t] = e;

            // Kalman gain: K = P * x / (1 + x' P x)
            let px = p_mat.dot(&xt);
            let f = 1.0 + xt.dot(&px);
            let gain = &px / f;

            // Update beta
            beta = &beta + &(&gain * e);

            // Update P: P = P - K * x' * P
            let outer = {
                let mut m = Array2::<f64>::zeros((k, k));
                for i in 0..k {
                    for j in 0..k {
                        m[[i, j]] = gain[i] * px[j];
                    }
                }
                m
            };
            p_mat = &p_mat - &outer;

            params_history.row_mut(t).assign(&beta);
        }

        // CUSUM
        let start = k;
        let valid_resid = residuals.slice(s![start..]).to_owned();
        let sigma = {
            let ss: f64 = valid_resid.iter().map(|r| r * r).sum();
            (ss / (n - k) as f64).sqrt()
        };

        let mut cusum = Array1::<f64>::zeros(n);
        let mut cumsum = 0.0;
        for t in start..n {
            cumsum += residuals[t] / sigma.max(1e-15);
            cusum[t] = cumsum;
        }

        // CUSUM of squares
        let mut cusum_sq = Array1::<f64>::zeros(n);
        let total_ss: f64 = (start..n).map(|t| residuals[t].powi(2)).sum();
        let mut partial_ss = 0.0;
        for t in start..n {
            partial_ss += residuals[t].powi(2);
            cusum_sq[t] = partial_ss / total_ss.max(1e-15);
        }

        Ok(RecursiveLSResult {
            params_history,
            residuals,
            cusum,
            cusum_squares: cusum_sq,
            params: beta,
            n_obs: n,
        })
    }
}

// ─── RollingOLS ────────────────────────────────────────────────────────────────

/// Result of Rolling OLS/WLS estimation.
#[derive(Debug)]
pub struct RollingResult {
    /// Parameter estimates at each time step (T x k), NaN before window fills
    pub params_history: Array2<f64>,
    /// R-squared at each time step
    pub r_squared_history: Array1<f64>,
    /// Residuals (one-step-ahead)
    pub residuals: Array1<f64>,
    pub n_obs: usize,
    pub window: usize,
}

impl fmt::Display for RollingResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Rolling Regression ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10}", "Window:", self.window)?;
        writeln!(
            f,
            "{:<20} {:>10}",
            "Parameters:",
            self.params_history.ncols()
        )?;

        // Show last valid parameters
        if self.n_obs > 0 {
            let last = self.params_history.row(self.n_obs - 1);
            writeln!(f, "\nLast window coefficients:")?;
            for (i, &v) in last.iter().enumerate() {
                writeln!(f, "  beta[{}] = {:.6}", i, v)?;
            }
        }
        writeln!(f, "{:=^78}", "")
    }
}

/// Rolling OLS estimator with fixed window size.
pub struct RollingOLS;

impl RollingOLS {
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        window: usize,
    ) -> Result<RollingResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();

        if n != x.nrows() {
            return Err(GreenersError::ShapeMismatch(
                "y and x row count mismatch".into(),
            ));
        }
        if window <= k {
            return Err(GreenersError::InvalidOperation(
                "Window must be larger than number of parameters".into(),
            ));
        }
        if window > n {
            return Err(GreenersError::InvalidOperation(
                "Window larger than sample size".into(),
            ));
        }

        let mut params_history = Array2::<f64>::from_elem((n, k), f64::NAN);
        let mut r_squared_history = Array1::<f64>::from_elem(n, f64::NAN);
        let mut residuals = Array1::<f64>::from_elem(n, f64::NAN);

        for t in (window - 1)..n {
            let start = t + 1 - window;
            let y_win = y.slice(s![start..=t]).to_owned();
            let x_win = x.slice(s![start..=t, ..]).to_owned();

            if let Ok(ols) = OLS::fit(&y_win, &x_win, CovarianceType::NonRobust) {
                params_history.row_mut(t).assign(&ols.params);
                r_squared_history[t] = ols.r_squared;
                let fitted = x_win.dot(&ols.params);
                residuals[t] = y_win[window - 1] - fitted[window - 1];
            }
        }

        Ok(RollingResult {
            params_history,
            r_squared_history,
            residuals,
            n_obs: n,
            window,
        })
    }
}

/// Rolling WLS estimator with fixed window size.
pub struct RollingWLS;

impl RollingWLS {
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        window: usize,
        weights: &Array1<f64>,
    ) -> Result<RollingResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();

        if n != x.nrows() || n != weights.len() {
            return Err(GreenersError::ShapeMismatch(
                "y, x, and weights length mismatch".into(),
            ));
        }
        if window <= k {
            return Err(GreenersError::InvalidOperation(
                "Window must be larger than number of parameters".into(),
            ));
        }
        if window > n {
            return Err(GreenersError::InvalidOperation(
                "Window larger than sample size".into(),
            ));
        }

        let mut params_history = Array2::<f64>::from_elem((n, k), f64::NAN);
        let mut r_squared_history = Array1::<f64>::from_elem(n, f64::NAN);
        let mut residuals = Array1::<f64>::from_elem(n, f64::NAN);

        for t in (window - 1)..n {
            let start = t + 1 - window;
            let y_win = y.slice(s![start..=t]).to_owned();
            let x_win = x.slice(s![start..=t, ..]).to_owned();
            let w_win = weights.slice(s![start..=t]).to_owned();

            // WLS: transform by sqrt(w)
            let sqrt_w = w_win.mapv(f64::sqrt);
            let y_t: Array1<f64> = &y_win * &sqrt_w;
            let mut x_t = x_win.clone();
            for (i, mut row) in x_t.axis_iter_mut(ndarray::Axis(0)).enumerate() {
                row *= sqrt_w[i];
            }

            if let Ok(ols) = OLS::fit(&y_t, &x_t, CovarianceType::NonRobust) {
                params_history.row_mut(t).assign(&ols.params);
                r_squared_history[t] = ols.r_squared;
                let fitted = x_win.dot(&ols.params);
                residuals[t] = y_win[window - 1] - fitted[window - 1];
            }
        }

        Ok(RollingResult {
            params_history,
            r_squared_history,
            residuals,
            n_obs: n,
            window,
        })
    }
}
