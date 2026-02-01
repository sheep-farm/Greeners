use crate::GreenersError;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use std::fmt;

/// State space model specification:
/// y_t = H * s_t + e_t,  e_t ~ N(0, R_obs)
/// s_t = F * s_{t-1} + R * u_t,  u_t ~ N(0, Q)
#[derive(Debug, Clone)]
pub struct StateSpaceModel {
    /// Observation matrix (n_obs x n_states)
    pub h: Array2<f64>,
    /// Transition matrix (n_states x n_states)
    pub f: Array2<f64>,
    /// State noise selection matrix (n_states x n_state_shocks)
    pub r: Array2<f64>,
    /// State noise covariance (n_state_shocks x n_state_shocks)
    pub q: Array2<f64>,
    /// Observation noise covariance (n_obs x n_obs)
    pub r_obs: Array2<f64>,
    /// Initial state mean (n_states)
    pub s0: Array1<f64>,
    /// Initial state covariance (n_states x n_states)
    pub p0: Array2<f64>,
}

/// Kalman filter result.
#[derive(Debug, Clone)]
pub struct KalmanResult {
    pub filtered_states: Vec<Array1<f64>>,
    pub filtered_cov: Vec<Array2<f64>>,
    pub predicted_states: Vec<Array1<f64>>,
    pub predicted_cov: Vec<Array2<f64>>,
    pub innovations: Vec<Array1<f64>>,
    pub innovation_cov: Vec<Array2<f64>>,
    pub log_likelihood: f64,
    pub n_obs: usize,
    pub n_states: usize,
}

/// Smoothed state result (Rauch-Tung-Striebel).
#[derive(Debug, Clone)]
pub struct SmoothedResult {
    pub smoothed_states: Vec<Array1<f64>>,
    pub smoothed_cov: Vec<Array2<f64>>,
    pub log_likelihood: f64,
    pub n_obs: usize,
}

/// Full state space estimation result.
#[derive(Debug)]
pub struct StateSpaceResult {
    pub filtered_states: Vec<Array1<f64>>,
    pub smoothed_states: Vec<Array1<f64>>,
    pub filtered_cov: Vec<Array2<f64>>,
    pub smoothed_cov: Vec<Array2<f64>>,
    pub innovations: Vec<Array1<f64>>,
    pub log_likelihood: f64,
    pub n_obs: usize,
    pub n_states: usize,
}

impl StateSpaceResult {
    /// Forecast `steps` ahead from the last filtered state.
    pub fn predict(&self, model: &StateSpaceModel, steps: usize) -> Vec<Array1<f64>> {
        let mut forecasts = Vec::with_capacity(steps);
        let mut s = self
            .filtered_states
            .last()
            .cloned()
            .unwrap_or(model.s0.clone());

        for _ in 0..steps {
            s = model.f.dot(&s);
            let y_hat = model.h.dot(&s);
            forecasts.push(y_hat);
        }

        forecasts
    }
}

impl fmt::Display for StateSpaceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^60}", " State Space Model ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10}", "States:", self.n_states)?;
        writeln!(f, "{:<20} {:>10.4}", "Log-likelihood:", self.log_likelihood)?;
        writeln!(f, "{:=^60}", "")
    }
}

/// Kalman filter.
pub struct KalmanFilter;

impl KalmanFilter {
    /// Run the Kalman filter forward pass.
    ///
    /// * `model` — state space model specification
    /// * `y` — observations, each element is a vector (can handle multivariate)
    pub fn filter(
        model: &StateSpaceModel,
        y: &[Array1<f64>],
    ) -> Result<KalmanResult, GreenersError> {
        let t = y.len();
        let m = model.f.nrows(); // n_states

        let mut filtered_states = Vec::with_capacity(t);
        let mut filtered_cov = Vec::with_capacity(t);
        let mut predicted_states = Vec::with_capacity(t);
        let mut predicted_cov = Vec::with_capacity(t);
        let mut innovations = Vec::with_capacity(t);
        let mut innovation_cov = Vec::with_capacity(t);
        let mut log_lik = 0.0;

        let rqr = model.r.dot(&model.q).dot(&model.r.t());

        let mut s_pred = model.s0.clone();
        let mut p_pred = model.p0.clone();

        for obs in y.iter() {
            // Predict step already done (s_pred, p_pred)
            predicted_states.push(s_pred.clone());
            predicted_cov.push(p_pred.clone());

            // Innovation
            let y_pred = model.h.dot(&s_pred);
            let v = obs - &y_pred;

            // Innovation covariance: F_t = H * P_pred * H' + R_obs
            let f_cov = model.h.dot(&p_pred).dot(&model.h.t()) + &model.r_obs;

            innovations.push(v.clone());
            innovation_cov.push(f_cov.clone());

            // Kalman gain: K = P_pred * H' * F^{-1}
            let f_inv = f_cov.inv().map_err(|_| GreenersError::SingularMatrix)?;
            let k_gain = p_pred.dot(&model.h.t()).dot(&f_inv);

            // Update
            let s_filt = &s_pred + &k_gain.dot(&v);
            let eye = Array2::<f64>::eye(m);
            let p_filt = (&eye - &k_gain.dot(&model.h)).dot(&p_pred);

            filtered_states.push(s_filt.clone());
            filtered_cov.push(p_filt.clone());

            // Log-likelihood contribution
            let n_y = obs.len() as f64;
            let det = det_positive(&f_cov);
            if det > 1e-30 {
                log_lik += -0.5
                    * (n_y * (2.0 * std::f64::consts::PI).ln() + det.ln() + v.dot(&f_inv.dot(&v)));
            }

            // Predict next
            s_pred = model.f.dot(&s_filt);
            p_pred = model.f.dot(&p_filt).dot(&model.f.t()) + &rqr;
        }

        Ok(KalmanResult {
            filtered_states,
            filtered_cov,
            predicted_states,
            predicted_cov,
            innovations,
            innovation_cov,
            log_likelihood: log_lik,
            n_obs: t,
            n_states: m,
        })
    }
}

/// Kalman smoother (Rauch-Tung-Striebel).
pub struct KalmanSmoother;

impl KalmanSmoother {
    /// Run the backward smoothing pass.
    pub fn smooth(
        model: &StateSpaceModel,
        filter_result: &KalmanResult,
    ) -> Result<SmoothedResult, GreenersError> {
        let t = filter_result.n_obs;

        let mut smoothed_states = vec![Array1::<f64>::zeros(0); t];
        let mut smoothed_cov = vec![Array2::<f64>::zeros((0, 0)); t];

        // Initialize with last filtered state
        smoothed_states[t - 1] = filter_result.filtered_states[t - 1].clone();
        smoothed_cov[t - 1] = filter_result.filtered_cov[t - 1].clone();

        // Backward pass
        for i in (0..t - 1).rev() {
            let p_filt = &filter_result.filtered_cov[i];
            let p_pred_next = &filter_result.predicted_cov[i + 1];

            let p_pred_inv = p_pred_next
                .inv()
                .map_err(|_| GreenersError::SingularMatrix)?;

            // Smoother gain: L = P_filt * F' * P_pred_next^{-1}
            let l_gain = p_filt.dot(&model.f.t()).dot(&p_pred_inv);

            // Smoothed state
            let s_diff = &smoothed_states[i + 1] - &filter_result.predicted_states[i + 1];
            smoothed_states[i] = &filter_result.filtered_states[i] + &l_gain.dot(&s_diff);

            // Smoothed covariance
            let p_diff = &smoothed_cov[i + 1] - p_pred_next;
            smoothed_cov[i] = p_filt + &l_gain.dot(&p_diff).dot(&l_gain.t());
        }

        Ok(SmoothedResult {
            smoothed_states,
            smoothed_cov,
            log_likelihood: filter_result.log_likelihood,
            n_obs: t,
        })
    }
}

/// Convenience function: filter + smooth in one call.
pub fn state_space_estimate(
    model: &StateSpaceModel,
    y: &[Array1<f64>],
) -> Result<StateSpaceResult, GreenersError> {
    let filter_result = KalmanFilter::filter(model, y)?;
    let smooth_result = KalmanSmoother::smooth(model, &filter_result)?;

    Ok(StateSpaceResult {
        filtered_states: filter_result.filtered_states,
        smoothed_states: smooth_result.smoothed_states,
        filtered_cov: filter_result.filtered_cov,
        smoothed_cov: smooth_result.smoothed_cov,
        innovations: filter_result.innovations,
        log_likelihood: filter_result.log_likelihood,
        n_obs: filter_result.n_obs,
        n_states: filter_result.n_states,
    })
}

/// Compute determinant, returning positive value or fallback.
fn det_positive(m: &Array2<f64>) -> f64 {
    // Simple determinant for small matrices
    let n = m.nrows();
    if n == 1 {
        return m[[0, 0]].abs().max(1e-30);
    }
    if n == 2 {
        return (m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]])
            .abs()
            .max(1e-30);
    }
    // For larger, use LU
    use ndarray_linalg::Determinant;
    m.det().unwrap_or(1e-30).abs().max(1e-30)
}
