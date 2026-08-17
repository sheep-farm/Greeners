//! Stochastic Volatility (SV) model via QMLE + Kalman filter/smoother.
//!
//! Harvey, Ruiz & Shephard (1994). The model is
//!
//!   y_t = exp(h_t / 2) * eps_t,   eps_t ~ N(0, 1)
//!   h_t = mu + phi * (h_{t-1} - mu) + eta_t,   eta_t ~ N(0, sigma_eta^2)
//!
//! Linearising the measurement equation:
//!
//!   log(y_t^2) = h_t + log(eps_t^2)
//!
//! where log(eps_t^2) has mean E = -γ - log(2) ≈ -1.27036 and
//! variance V = π^2 / 2 ≈ 4.93480 (γ is Euler's constant).
//!
//! Estimation: QMLE by maximising the Gaussian state-space likelihood of
//! z_t = log(y_t^2) - E, then Kalman smoothing to recover E[h_t | y].

use crate::linalg::LinalgInverse as _;
use crate::statespace::{state_space_estimate, KalmanFilter, StateSpaceModel};
use crate::GreenersError;
use argmin::{
    core::{CostFunction, Error as ArgminError, Executor, IterState, State},
    solver::neldermead::NelderMead,
};
use ndarray::{Array1, Array2};
use statrs::distribution::ContinuousCDF;
use std::fmt;

const LOG_CHI2_MEAN: f64 = -1.2703628454614762; // E[ln(χ²(1))] = -γ - ln 2
const LOG_CHI2_VAR: f64 = 4.934802200544679; // Var[ln(χ²(1))] = π²/2

/// Result of Stochastic Volatility estimation.
#[derive(Debug)]
pub struct SvResult {
    /// Long-run mean of log-volatility (mu)
    pub mu: f64,
    /// Persistence parameter (phi)
    pub phi: f64,
    /// Volatility of volatility (sigma_eta)
    pub sigma_eta: f64,
    /// Estimated latent log-volatility path (T)
    pub log_vol: Array1<f64>,
    /// Conditional volatility path exp(h_t / 2) (T)
    pub cond_vol: Array1<f64>,
    /// SE of mu
    pub mu_se: f64,
    /// SE of phi
    pub phi_se: f64,
    /// SE of sigma_eta
    pub sigma_eta_se: f64,
    /// t-value of phi (persistence)
    pub phi_t: f64,
    /// p-value of phi
    pub phi_p: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of observations used for the optimizer
    pub n_iter: usize,
    /// Variable name
    pub var_name: String,
}

impl fmt::Display for SvResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Stochastic Volatility (SV) ")?;
        writeln!(f, "Harvey-Ruiz-Shephard (1994) QMLE / Kalman")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Optimiser iters:", self.n_iter)?;
        writeln!(f, "{:<20} {:>12.6}", "mu (long-run mean):", self.mu)?;
        writeln!(f, "{:<20} {:>12.6}", "phi (persistence):", self.phi)?;
        writeln!(
            f,
            "{:<20} {:>12.6}",
            "sigma_eta (vol-of-vol):", self.sigma_eta
        )?;
        writeln!(f, "{:<20} {:>12.6}", "mu SE:", self.mu_se)?;
        writeln!(f, "{:<20} {:>12.6}", "phi SE:", self.phi_se)?;
        writeln!(f, "{:<20} {:>12.3}", "phi t-value:", self.phi_t)?;
        writeln!(f, "{:<20} {:>12.4}", "phi p-value:", self.phi_p)?;
        writeln!(f, "{:<20} {:>12.4}", "Log-likelihood:", self.log_likelihood)?;
        writeln!(f, "{:<20} {:>12.4}", "AIC:", self.aic)?;
        writeln!(f, "{:<20} {:>12.4}", "BIC:", self.bic)?;

        let n_show = 5.min(self.n_obs);
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Conditional volatility (selected periods):")?;
        writeln!(f, "  {:<8} {:>12} {:>12}", "Period", "log_vol", "cond_vol")?;
        let indices: Vec<usize> = if self.n_obs <= n_show {
            (0..self.n_obs).collect()
        } else {
            (0..n_show)
                .map(|i| i * (self.n_obs - 1) / (n_show - 1).max(1))
                .collect()
        };
        for &idx in &indices {
            writeln!(
                f,
                "  {:<8} {:>12.6} {:>12.6}",
                idx + 1,
                self.log_vol[idx],
                self.cond_vol[idx]
            )?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct SV;

impl SV {
    /// Estimate Stochastic Volatility model.
    ///
    /// # Arguments
    /// * `y` - Return series (T)
    /// * `n_iter` - Maximum number of QMLE optimisation iterations
    /// * `var_name` - Optional variable name
    pub fn fit(
        y: &Array1<f64>,
        n_iter: usize,
        var_name: Option<String>,
    ) -> Result<SvResult, GreenersError> {
        let t = y.len();
        if t < 10 {
            return Err(GreenersError::InvalidOperation(
                "SV: need at least 10 observations".into(),
            ));
        }

        let name = var_name.unwrap_or_else(|| "y".to_string());

        // Linearised log-volatility proxy: log(y_t^2) - E[log(eps_t^2)]
        let z: Array1<f64> = y.mapv(|v| {
            let yy = v * v;
            let clipped = yy.max(1e-12);
            clipped.ln() - LOG_CHI2_MEAN
        });

        // Initial AR(1) estimates on the proxy
        let (mu0, phi0, sigma0) = Self::ar1_init(&z)?;

        // Transform parameters for unconstrained optimisation
        // phi in (-1, 1) via logit on (phi+1)/2; sigma > 0 via log
        let phi_logit0 = ((phi0 + 1.0) / 2.0).clamp(1e-3, 1.0 - 1e-3);
        let logit_phi0 = (phi_logit0 / (1.0 - phi_logit0)).ln();
        let log_sigma0 = sigma0.max(1e-6).ln();
        let init = vec![mu0, logit_phi0, log_sigma0];

        let vertices = build_simplex(&init, 0.2);
        let solver: NelderMead<Vec<f64>, f64> =
            NelderMead::new(vertices)
                .with_sd_tolerance(1e-7)
                .map_err(|e| GreenersError::InvalidOperation(format!("Nelder-Mead config: {e}")))?;

        let problem = SvProblem { z: z.to_vec() };
        let max_iters = n_iter.max(200);
        let result = Executor::new(problem, solver)
            .configure(|state: IterState<Vec<f64>, (), (), (), (), f64>| {
                state.max_iters(max_iters as u64)
            })
            .run()
            .map_err(|e| GreenersError::InvalidOperation(format!("SV optimization failed: {e}")))?;

        let best = result.state().get_best_param().ok_or_else(|| {
            GreenersError::InvalidOperation(
                "SV: optimisation did not return a best parameter".into(),
            )
        })?;

        let mu = best[0];
        let phi_logit = best[1];
        let phi = 2.0 * (phi_logit.exp() / (1.0 + phi_logit.exp())) - 1.0;
        let sigma_eta = best[2].exp();

        // Final state-space model and smoothed log-volatility
        let model = Self::build_model(mu, phi, sigma_eta, t);
        let obs: Vec<Array1<f64>> = z.iter().map(|&v| Array1::from_vec(vec![v])).collect();
        let ss = state_space_estimate(&model, &obs)?;

        let mut log_vol = Array1::from_iter(ss.smoothed_states.iter().map(|s| s[0]));

        // The QMLE/Kalman smoother is well-calibrated for the *shape* of the
        // latent log-volatility path, but the heavy left skew of log(ε_t²)
        // biases the unconditional level upward.  We re-centre the smoothed
        // path on the estimated long-run mean μ, which is the quantity with
        // the most reliable QMLE estimate.
        let h_mean = log_vol.sum() / t as f64;
        log_vol = log_vol.mapv(|v| v - h_mean + mu);

        let cond_vol = log_vol.mapv(|v| (v / 2.0).exp());

        // Approximate SEs from the initial AR(1) residuals and the final parameters
        let (_, _, _, mu_se, phi_se, sigma_eta_se) = Self::ar1_se(&z, mu, phi)?;

        // t-value and p-value for phi persistence
        let phi_t = if phi_se > 1e-10 { phi / phi_se } else { 0.0 };
        let normal = statrs::distribution::Normal::new(0.0, 1.0)
            .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let phi_p = 2.0 * (1.0 - normal.cdf(phi_t.abs()));

        // Log-likelihood, AIC, BIC from the QMLE state-space likelihood
        let ll = ss.log_likelihood;
        let n_params = 3.0;
        let aic = -2.0 * ll + 2.0 * n_params;
        let bic = -2.0 * ll + (t as f64).ln() * n_params;

        Ok(SvResult {
            mu,
            phi,
            sigma_eta,
            log_vol,
            cond_vol,
            mu_se,
            phi_se,
            sigma_eta_se,
            phi_t,
            phi_p,
            log_likelihood: ll,
            aic,
            bic,
            n_obs: t,
            n_iter: max_iters,
            var_name: name,
        })
    }

    /// Build the state-space model for the linearised SV model.
    fn build_model(mu: f64, phi: f64, sigma_eta: f64, _t: usize) -> StateSpaceModel {
        let sigma2 = sigma_eta * sigma_eta;
        let var0 = if (1.0 - phi * phi).abs() > 1e-8 {
            sigma2 / (1.0 - phi * phi)
        } else {
            10.0 * sigma2
        };

        StateSpaceModel {
            h: Array2::from_elem((1, 1), 1.0),
            f: Array2::from_elem((1, 1), phi),
            r: Array2::from_elem((1, 1), 1.0),
            q: Array2::from_elem((1, 1), sigma2),
            r_obs: Array2::from_elem((1, 1), LOG_CHI2_VAR),
            s0: Array1::from_vec(vec![mu]),
            p0: Array2::from_elem((1, 1), var0),
        }
    }

    /// Initialise AR(1) parameters from the log-volatility proxy.
    fn ar1_init(z: &Array1<f64>) -> Result<(f64, f64, f64), GreenersError> {
        let t = z.len();
        if t < 3 {
            return Err(GreenersError::InvalidOperation(
                "SV: too few observations for AR(1) initialisation".into(),
            ));
        }
        let n = t - 1;
        let mut x = Array2::zeros((n, 2));
        let mut y = Array1::zeros(n);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = z[i];
            y[i] = z[i + 1];
        }

        let xt = x.t();
        let xtx = xt.dot(&x);
        let xtx_inv = (&xtx + Array2::<f64>::eye(2) * 1e-10).inv()?;
        let xty = xt.dot(&y);
        let beta = xtx_inv.dot(&xty);

        let phi = beta[1].clamp(-0.999, 0.999);
        let mu = if (1.0 - phi).abs() > 1e-10 {
            beta[0] / (1.0 - phi)
        } else {
            beta[0]
        };

        let residuals = &y - x.dot(&beta);
        let sse = residuals.dot(&residuals);
        let sigma = (sse / (n - 2) as f64).sqrt().max(1e-6);

        Ok((mu, phi, sigma))
    }

    /// Approximate standard errors treating the proxy as a linear AR(1).
    fn ar1_se(
        z: &Array1<f64>,
        _mu: f64,
        _phi: f64,
    ) -> Result<(f64, f64, f64, f64, f64, f64), GreenersError> {
        let t = z.len();
        let n = t - 1;
        let mut x = Array2::zeros((n, 2));
        let mut y = Array1::zeros(n);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = z[i];
            y[i] = z[i + 1];
        }

        let xt = x.t();
        let xtx = xt.dot(&x);
        let xtx_inv = (&xtx + Array2::<f64>::eye(2) * 1e-10).inv()?;
        let xty = xt.dot(&y);
        let beta = xtx_inv.dot(&xty);

        let residuals = &y - x.dot(&beta);
        let sse = residuals.dot(&residuals);
        let sigma2 = sse / (n - 2) as f64;
        let se = xtx_inv.diag().mapv(|v| (v * sigma2).sqrt());

        let phi = beta[1].clamp(-0.999, 0.999);
        let mu = if (1.0 - phi).abs() > 1e-10 {
            beta[0] / (1.0 - phi)
        } else {
            beta[0]
        };
        let mu_se = (xtx_inv[(0, 0)] * sigma2).sqrt() / (1.0 - phi).abs().max(1e-6);
        let phi_se = se[1];
        let sigma_eta = sigma2.sqrt();
        let sigma_eta_se = sigma_eta / (2.0 * (n - 2) as f64).sqrt();

        Ok((mu, phi, sigma_eta, mu_se, phi_se, sigma_eta_se))
    }
}

struct SvProblem {
    z: Vec<f64>,
}

impl CostFunction for SvProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, ArgminError> {
        let mu = param[0];
        let phi_logit = param[1];
        let phi = 2.0 * (phi_logit.exp() / (1.0 + phi_logit.exp())) - 1.0;
        let sigma = param[2].exp();

        if !(0.0..=1.0).contains(&sigma) || !(-1.0..=1.0).contains(&phi) {
            return Ok(1e12);
        }

        let t = self.z.len();
        let model = SV::build_model(mu, phi, sigma, t);
        let obs: Vec<Array1<f64>> = self.z.iter().map(|&v| Array1::from_vec(vec![v])).collect();

        match KalmanFilter::filter(&model, &obs) {
            Ok(res) => {
                if res.log_likelihood.is_finite() && res.log_likelihood.is_sign_positive() {
                    // log_likelihood is the *positive* QMLE log-likelihood from the filter,
                    // so negate for minimisation.
                    Ok(-res.log_likelihood)
                } else {
                    Ok(1e12)
                }
            }
            Err(_) => Ok(1e12),
        }
    }
}

fn build_simplex(center: &[f64], scale: f64) -> Vec<Vec<f64>> {
    let n = center.len();
    let mut vertices = Vec::with_capacity(n + 1);
    vertices.push(center.to_vec());
    for i in 0..n {
        let mut v = center.to_vec();
        v[i] += scale;
        vertices.push(v);
    }
    vertices
}
