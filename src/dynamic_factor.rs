use crate::error::GreenersError;
use crate::statespace::{KalmanFilter, KalmanSmoother, StateSpaceModel};
use ndarray::{s, Array1, Array2, Axis};
use ndarray_linalg::{Eigh, Inverse, UPLO};
use std::fmt;

/// Dynamic Factor Model estimator.
///
/// Represents multiple observed time series as driven by a smaller number of
/// latent factors:
///
/// y_t = Lambda * f_t + eps_t       (observation equation)
/// f_t = A_1 f_{t-1} + ... + A_p f_{t-p} + u_t  (factor transition, VAR(p))
///
/// where y_t is k x 1 observed, f_t is r x 1 latent factors (r < k),
/// Lambda is k x r factor loadings.
pub struct DynamicFactor;

/// Result of a Dynamic Factor Model estimation.
#[derive(Debug)]
pub struct DynamicFactorResult {
    /// Factor loadings matrix Lambda (k x r)
    pub factor_loadings: Array2<f64>,
    /// Estimated factors (T x r)
    pub factors: Array2<f64>,
    /// AR coefficient matrices for factor VAR dynamics
    pub factor_ar_params: Vec<Array2<f64>>,
    /// Observation noise variances (k)
    pub sigma_obs: Array1<f64>,
    /// Factor innovation covariance (r x r)
    pub sigma_factor: Array2<f64>,
    /// Log-likelihood from Kalman filter
    pub log_likelihood: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Number of time observations
    pub n_obs: usize,
    /// Number of observed variables
    pub n_vars: usize,
    /// Number of latent factors
    pub n_factors: usize,
    /// AR order for factor dynamics
    pub factor_order: usize,
}

impl DynamicFactor {
    /// Fit a dynamic factor model.
    ///
    /// # Arguments
    /// * `data` - T x k matrix of observed series (T time periods, k variables)
    /// * `k_factors` - number of latent factors (r), must be < k
    /// * `factor_order` - AR order for factor dynamics (p)
    ///
    /// # Implementation
    /// Two-step estimation:
    /// 1. PCA to extract initial factors
    /// 2. VAR on extracted factors for AR parameters
    /// 3. OLS regression for factor loadings
    /// 4. Kalman smoother for refined factor estimates
    pub fn fit(
        data: &Array2<f64>,
        k_factors: usize,
        factor_order: usize,
    ) -> Result<DynamicFactorResult, GreenersError> {
        let (t, k) = (data.nrows(), data.ncols());

        if k_factors == 0 || k_factors >= k {
            return Err(GreenersError::InvalidOperation(
                "k_factors must be between 1 and n_vars - 1".into(),
            ));
        }
        if factor_order == 0 {
            return Err(GreenersError::InvalidOperation(
                "factor_order must be at least 1".into(),
            ));
        }
        if t <= factor_order + 1 {
            return Err(GreenersError::InvalidOperation(
                "Not enough observations for the given factor_order".into(),
            ));
        }

        let r = k_factors;
        let p = factor_order;

        // Standardize data
        let mut mean = Array1::<f64>::zeros(k);
        let mut std_dev = Array1::<f64>::zeros(k);
        for j in 0..k {
            let col = data.column(j);
            mean[j] = col.mean().unwrap_or(0.0);
            let var =
                col.iter().map(|x| (x - mean[j]).powi(2)).sum::<f64>() / (t - 1) as f64;
            std_dev[j] = var.sqrt().max(1e-15);
        }

        let mut z = data.clone();
        for (j, mut col) in z.axis_iter_mut(Axis(1)).enumerate() {
            col -= mean[j];
            col /= std_dev[j];
        }

        // Step 1: PCA - extract initial factors
        let corr = z.t().dot(&z) / (t - 1) as f64;
        let (eigenvalues, eigenvectors) = corr.eigh(UPLO::Upper)?;

        // Reverse to descending order, take top r eigenvectors
        let evec: Array2<f64> = eigenvectors.slice(s![.., ..;-1]).to_owned();
        let components = evec.slice(s![.., ..r]).to_owned(); // k x r

        // Initial factors: T x r
        let initial_factors = z.dot(&components);

        // Step 2: VAR(p) on extracted factors
        let (ar_params, sigma_u) = fit_var_on_factors(&initial_factors, p)?;

        // Step 3: OLS regression of each observed series on factors -> Lambda
        // z_t = Lambda * f_t + eps_t
        // Lambda = (F'F)^{-1} F' Z  transposed appropriately
        let ftf = initial_factors.t().dot(&initial_factors);
        let ftf_inv = ftf.inv().map_err(|_| GreenersError::SingularMatrix)?;
        let ftz = initial_factors.t().dot(&z); // r x k
        let lambda_t = ftf_inv.dot(&ftz); // r x k
        let lambda = lambda_t.t().to_owned(); // k x r

        // Observation noise variances
        let residuals = &z - &initial_factors.dot(&lambda.t()); // T x k
        let mut sigma_obs = Array1::<f64>::zeros(k);
        for j in 0..k {
            let col = residuals.column(j);
            sigma_obs[j] = col.dot(&col) / t as f64;
            sigma_obs[j] = sigma_obs[j].max(1e-10);
        }

        // Step 4: Kalman smoother for refined factor estimates
        // State: [f_t, f_{t-1}, ..., f_{t-p+1}] of dimension r*p
        let state_dim = r * p;

        // Observation matrix H: k x state_dim
        // y_t = H * state_t, where H = [Lambda, 0, ..., 0]
        let mut h_mat = Array2::<f64>::zeros((k, state_dim));
        h_mat.slice_mut(s![.., ..r]).assign(&lambda);

        // Transition matrix F: state_dim x state_dim (companion form)
        let mut f_mat = Array2::<f64>::zeros((state_dim, state_dim));
        for (lag, ar_mat) in ar_params.iter().enumerate() {
            let col_start = lag * r;
            f_mat.slice_mut(s![..r, col_start..col_start + r])
                .assign(ar_mat);
        }
        // Identity blocks for lagged states
        if p > 1 {
            for i in 0..(p - 1) {
                let row_start = (i + 1) * r;
                let col_start = i * r;
                for j in 0..r {
                    f_mat[[row_start + j, col_start + j]] = 1.0;
                }
            }
        }

        // State noise selection matrix R: state_dim x r
        let mut r_mat = Array2::<f64>::zeros((state_dim, r));
        for i in 0..r {
            r_mat[[i, i]] = 1.0;
        }

        // Observation noise covariance (diagonal)
        let mut r_obs = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            r_obs[[i, i]] = sigma_obs[i];
        }

        // Initial state
        let s0 = Array1::<f64>::zeros(state_dim);
        let p0 = Array2::<f64>::eye(state_dim) * 10.0;

        let ss_model = StateSpaceModel {
            h: h_mat,
            f: f_mat,
            r: r_mat,
            q: sigma_u.clone(),
            r_obs,
            s0,
            p0,
        };

        // Convert data to observations for Kalman filter
        let observations: Vec<Array1<f64>> = (0..t).map(|i| z.row(i).to_owned()).collect();

        let filter_result = KalmanFilter::filter(&ss_model, &observations)?;
        let smooth_result = KalmanSmoother::smooth(&ss_model, &filter_result)?;

        // Extract smoothed factors (first r elements of state)
        let mut factors = Array2::<f64>::zeros((t, r));
        for (i, state) in smooth_result.smoothed_states.iter().enumerate() {
            for j in 0..r {
                factors[[i, j]] = state[j];
            }
        }

        // Compute log-likelihood and information criteria
        let log_lik = filter_result.log_likelihood;
        let n_params = (k * r) + (r * r * p) + k + (r * (r + 1) / 2);
        let aic = -2.0 * log_lik + 2.0 * n_params as f64;
        let bic = -2.0 * log_lik + (n_params as f64) * (t as f64).ln();

        Ok(DynamicFactorResult {
            factor_loadings: lambda,
            factors,
            factor_ar_params: ar_params,
            sigma_obs,
            sigma_factor: sigma_u,
            log_likelihood: log_lik,
            aic,
            bic,
            n_obs: t,
            n_vars: k,
            n_factors: r,
            factor_order: p,
        })
    }
}

/// Fit a VAR(p) model on factor matrix by OLS.
/// Returns (Vec of r x r AR matrices, r x r innovation covariance).
fn fit_var_on_factors(
    factors: &Array2<f64>,
    p: usize,
) -> Result<(Vec<Array2<f64>>, Array2<f64>), GreenersError> {
    let (t, r) = (factors.nrows(), factors.ncols());
    let t_eff = t - p;

    // Build Y: t_eff x r  and X: t_eff x (r*p)
    let mut y = Array2::<f64>::zeros((t_eff, r));
    let mut x = Array2::<f64>::zeros((t_eff, r * p));

    for i in 0..t_eff {
        let row = i + p;
        y.row_mut(i).assign(&factors.row(row));
        for lag in 0..p {
            let src_row = row - lag - 1;
            let col_start = lag * r;
            for j in 0..r {
                x[[i, col_start + j]] = factors[[src_row, j]];
            }
        }
    }

    // OLS: B = (X'X)^{-1} X'Y, B is (r*p) x r
    let xtx = x.t().dot(&x);
    let xtx_inv = xtx.inv().map_err(|_| GreenersError::SingularMatrix)?;
    let xty = x.t().dot(&y);
    let b = xtx_inv.dot(&xty); // (r*p) x r

    // Extract AR matrices
    let mut ar_params = Vec::with_capacity(p);
    for lag in 0..p {
        let row_start = lag * r;
        let ar_mat = b.slice(s![row_start..row_start + r, ..]).t().to_owned(); // r x r
        ar_params.push(ar_mat);
    }

    // Residuals and covariance
    let resid = &y - &x.dot(&b);
    let sigma = resid.t().dot(&resid) / t_eff as f64;

    Ok((ar_params, sigma))
}

impl DynamicFactorResult {
    /// Forecast observed series `steps` ahead.
    ///
    /// Returns a `steps x n_vars` matrix of forecasted values.
    pub fn predict(&self, steps: usize) -> Array2<f64> {
        let r = self.n_factors;
        let p = self.factor_order;
        let k = self.n_vars;
        let t = self.n_obs;

        // Collect recent factor values for VAR forecasting
        let mut recent_factors: Vec<Array1<f64>> = Vec::with_capacity(p);
        for lag in 0..p {
            let row = t - 1 - lag;
            if row < t {
                recent_factors.push(self.factors.row(row).to_owned());
            } else {
                recent_factors.push(Array1::<f64>::zeros(r));
            }
        }

        let mut forecasts = Array2::<f64>::zeros((steps, k));

        for step in 0..steps {
            // f_t = A_1 f_{t-1} + A_2 f_{t-2} + ...
            let mut f_new = Array1::<f64>::zeros(r);
            for (lag, ar_mat) in self.factor_ar_params.iter().enumerate() {
                if lag < recent_factors.len() {
                    f_new = &f_new + &ar_mat.dot(&recent_factors[lag]);
                }
            }

            // y_t = Lambda * f_t
            let y_hat = self.factor_loadings.dot(&f_new);
            forecasts.row_mut(step).assign(&y_hat);

            // Shift recent factors
            recent_factors.insert(0, f_new);
            if recent_factors.len() > p {
                recent_factors.pop();
            }
        }

        forecasts
    }
}

impl fmt::Display for DynamicFactorResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^60}", " Dynamic Factor Model ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10}", "Variables:", self.n_vars)?;
        writeln!(f, "{:<20} {:>10}", "Factors:", self.n_factors)?;
        writeln!(f, "{:<20} {:>10}", "Factor AR order:", self.factor_order)?;
        writeln!(f, "{:<20} {:>10.4}", "Log-likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>10.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>10.4}", "BIC:", self.bic)?;

        writeln!(f, "\nFactor Loadings (Lambda):")?;
        writeln!(f, "{:-^40}", "")?;
        for i in 0..self.n_vars {
            write!(f, "  Var{:<3}", i + 1)?;
            for j in 0..self.n_factors {
                write!(f, " {:>8.4}", self.factor_loadings[[i, j]])?;
            }
            writeln!(f)?;
        }

        writeln!(f, "\nObservation Noise Variances:")?;
        for i in 0..self.n_vars {
            writeln!(f, "  Var{}: {:.4}", i + 1, self.sigma_obs[i])?;
        }

        writeln!(f, "{:=^60}", "")
    }
}
