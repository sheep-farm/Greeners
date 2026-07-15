//! Targeted Maximum Likelihood Estimation (TMLE)
//! (van der Laan & Rubin 2006).
//!
//! A doubly-robust, semiparametric efficient estimator for
//! causal effects. Combines:
//!   1. Initial estimates of nuisance functions Q(W) = E[Y|T,W]
//!      and g(W) = P(T=1|W) via ML
//!   2. Targeting step: updates Q via a clever covariate
//!      H = (T - g(W)) / (g(W) * (1 - g(W)))
//!      and fluctuation epsilon: Q* = logit^{-1}(logit(Q) + epsilon*H)
//!   3. Final ATE = mean(Q*(T=1, W) - Q*(T=0, W))
//!
//! TMLE is consistent if either Q or g is correctly specified
//! (double robustness), and achieves the semiparametric efficiency
//! bound when both are correct. The targeting step ensures the
//! estimator solves the efficient influence function equation.

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of TMLE estimation.
#[derive(Debug)]
pub struct TmleResult {
    /// TMLE estimate of ATE
    pub ate: f64,
    /// Standard error (efficient influence function-based)
    pub se: f64,
    /// t-statistic
    pub t_stat: f64,
    /// p-value (two-sided)
    pub p_value: f64,
    /// 95% confidence interval
    pub ci: [f64; 2],
    /// Targeting fluctuation parameter epsilon
    pub epsilon: f64,
    /// Initial ATE (before targeting)
    pub initial_ate: f64,
    /// Propensity score g(W) (n)
    pub propensity: Array1<f64>,
    /// Initial outcome regression Q(W) (n)
    pub initial_q: Array1<f64>,
    /// Targeted outcome regression Q*(W) (n)
    pub targeted_q: Array1<f64>,
    /// Clever covariate H (n)
    pub clever_covariate: Array1<f64>,
    /// Efficient influence function values (n)
    pub eif: Array1<f64>,
    /// Number of observations
    pub n_obs: usize,
    /// Number of confounders
    pub n_confounders: usize,
}

impl fmt::Display for TmleResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " TMLE ")?;
        writeln!(f, "van der Laan & Rubin (2006)")?;
        writeln!(f, "Targeted Maximum Likelihood Estimation")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Confounders:", self.n_confounders)?;
        writeln!(f, "{:<20} {:>12.6}", "Initial ATE:", self.initial_ate)?;
        writeln!(f, "{:<20} {:>12.6}", "TMLE ATE:", self.ate)?;
        writeln!(f, "{:<20} {:>12.6}", "Epsilon (fluctuation):", self.epsilon)?;
        writeln!(f, "{:<20} {:>12.6}", "Std. Error:", self.se)?;
        writeln!(f, "{:<20} {:>12.3}", "t-statistic:", self.t_stat)?;
        writeln!(f, "{:<20} {:>12.4}", "p-value:", self.p_value)?;
        writeln!(
            f,
            "{:<20} [{:.4}, {:.4}]",
            "95% CI:", self.ci[0], self.ci[1]
        )?;

        // EIF summary
        let eif_mean = self.eif.mean().unwrap_or(0.0);
        let eif_var = self.eif.mapv(|v| (v - eif_mean).powi(2)).sum() / self.n_obs as f64;
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Efficient Influence Function:")?;
        writeln!(f, "  Mean: {:>12.6}  Var: {:>12.6}", eif_mean, eif_var)?;

        // Propensity summary
        let prop_mean = self.propensity.mean().unwrap_or(0.0);
        let prop_min = self
            .propensity
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let prop_max = self
            .propensity
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        writeln!(f, "\n  Propensity score g(W):")?;
        writeln!(
            f,
            "  Mean: {:>12.6}  Range: [{:.4}, {:.4}]",
            prop_mean, prop_min, prop_max
        )?;

        write!(f, "{:=^78}", "")
    }
}

pub struct TMLE;

impl TMLE {
    /// Estimate ATE via TMLE.
    ///
    /// # Arguments
    /// * `y` - Outcome (n)
    /// * `t` - Treatment indicator (n), true if treated
    /// * `w` - Confounders (n x k)
    pub fn fit(y: &Array1<f64>, t: &[bool], w: &Array2<f64>) -> Result<TmleResult, GreenersError> {
        let n = y.len();
        let k = w.ncols();
        if t.len() != n || w.nrows() != n {
            return Err(GreenersError::ShapeMismatch(
                "TMLE: dimension mismatch".into(),
            ));
        }
        if n < 20 {
            return Err(GreenersError::InvalidOperation(
                "TMLE: need at least 20 observations".into(),
            ));
        }

        let n_treated = t.iter().filter(|&&t| t).count();
        let n_control = n - n_treated;
        if n_treated < 3 || n_control < 3 {
            return Err(GreenersError::InvalidOperation(
                "TMLE: need at least 3 treated and 3 control".into(),
            ));
        }

        // Step 1: Initial estimates of Q(W) and g(W) via OLS
        // Q(W) = E[Y|T,W] — regression of Y on [T, W]
        let t_vec: Array1<f64> = t.iter().map(|&t| if t { 1.0 } else { 0.0 }).collect();

        // Q model: Y ~ T + W (linear)
        let mut x_q = Array2::zeros((n, k + 2)); // intercept + T + W
        for i in 0..n {
            x_q[(i, 0)] = 1.0;
            x_q[(i, 1)] = t_vec[i];
            for j in 0..k {
                x_q[(i, j + 2)] = w[(i, j)];
            }
        }
        let xt_q = x_q.t();
        let xtx_q = xt_q.dot(&x_q);
        let xtx_q_inv = (&xtx_q + Array2::<f64>::eye(k + 2) * 1e-8).inv()?;
        let xty_q = xt_q.dot(y);
        let beta_q: Array1<f64> = xtx_q_inv.dot(&xty_q);

        // Initial Q(W) for each obs: predict with T=1 and T=0
        let mut q1_init = Array1::zeros(n); // Q(T=1, W)
        let mut q0_init = Array1::zeros(n); // Q(T=0, W)
        let mut q_init = Array1::zeros(n); // Q(T, W) with actual T
        for i in 0..n {
            let mut pred1 = beta_q[0] + beta_q[1]; // T=1
            let mut pred0 = beta_q[0]; // T=0
            for j in 0..k {
                pred1 += beta_q[j + 2] * w[(i, j)];
                pred0 += beta_q[j + 2] * w[(i, j)];
            }
            q1_init[i] = pred1;
            q0_init[i] = pred0;
            q_init[i] = if t[i] { pred1 } else { pred0 };
        }

        // g(W) = P(T=1|W) — linear probability model
        let mut x_g = Array2::zeros((n, k + 1));
        for i in 0..n {
            x_g[(i, 0)] = 1.0;
            for j in 0..k {
                x_g[(i, j + 1)] = w[(i, j)];
            }
        }
        let xt_g = x_g.t();
        let xtx_g = xt_g.dot(&x_g);
        let xtx_g_inv = (&xtx_g + Array2::<f64>::eye(k + 1) * 1e-8).inv()?;
        let xty_g = xt_g.dot(&t_vec);
        let beta_g: Array1<f64> = xtx_g_inv.dot(&xty_g);

        let mut g = Array1::zeros(n);
        for i in 0..n {
            let mut pred = beta_g[0];
            for j in 0..k {
                pred += beta_g[j + 1] * w[(i, j)];
            }
            g[i] = pred.clamp(0.01, 0.99);
        }

        // Step 2: Targeting step
        // Bound Q to (0.01, 0.99) for logit transform
        let q_bounded: Array1<f64> = q_init.mapv(|v| v.clamp(0.01, 0.99));

        // Clever covariate: H = T/g - (1-T)/(1-g) = (T - g) / (g * (1-g))
        let h: Array1<f64> = (0..n)
            .map(|i| {
                let ti = if t[i] { 1.0 } else { 0.0 };
                (ti - g[i]) / (g[i] * (1.0 - g[i]))
            })
            .collect();

        // logit(Q) = log(Q / (1 - Q))
        let logit_q: Array1<f64> = q_bounded.mapv(|v| (v / (1.0 - v)).ln());

        // Estimate epsilon: minimize log-likelihood loss
        // For binary Y: loss = -[Y*log(Q*) + (1-Y)*log(1-Q*)]
        // Q* = logit^{-1}(logit(Q) + epsilon * H)
        // epsilon = sum(Y - Q) * H / sum(Q * (1-Q) * H^2)  (one-step Newton)
        let numerator: f64 = (0..n).map(|i| (y[i] - q_bounded[i]) * h[i]).sum();
        let denominator: f64 = (0..n)
            .map(|i| q_bounded[i] * (1.0 - q_bounded[i]) * h[i] * h[i])
            .sum();
        let epsilon = if denominator.abs() > 1e-15 {
            numerator / denominator
        } else {
            0.0
        };

        // Targeted Q*: logit^{-1}(logit(Q) + epsilon * H)
        let logit_q_star: Array1<f64> = (0..n).map(|i| logit_q[i] + epsilon * h[i]).collect();
        let q_star: Array1<f64> = logit_q_star.mapv(|v| {
            let exp_v = v.exp();
            exp_v / (1.0 + exp_v)
        });

        // Targeted Q1* and Q0*: predict with T=1 and T=0
        let mut q1_star = Array1::zeros(n);
        let mut q0_star = Array1::zeros(n);
        for i in 0..n {
            // H with T=1: 1/g
            let h1 = 1.0 / g[i];
            // H with T=0: -1/(1-g)
            let h0 = -1.0 / (1.0 - g[i]);

            let logit_q1 = logit_q[i] + epsilon * h1;
            let logit_q0 = logit_q[i] + epsilon * h0;

            let exp1 = logit_q1.exp();
            let exp0 = logit_q0.exp();
            q1_star[i] = exp1 / (1.0 + exp1);
            q0_star[i] = exp0 / (1.0 + exp0);
        }

        // Step 3: TMLE ATE = mean(Q1* - Q0*)
        let ate: f64 = (0..n).map(|i| q1_star[i] - q0_star[i]).sum::<f64>() / n as f64;
        let initial_ate: f64 = (0..n).map(|i| q1_init[i] - q0_init[i]).sum::<f64>() / n as f64;

        // Step 4: EIF-based SE
        // EIF = H * (Y - Q*) + (Q1* - Q0* - ATE)
        let eif: Array1<f64> = (0..n)
            .map(|i| {
                let ti = if t[i] { 1.0 } else { 0.0 };
                let h_i = (ti - g[i]) / (g[i] * (1.0 - g[i]));
                h_i * (y[i] - q_star[i]) + (q1_star[i] - q0_star[i] - ate)
            })
            .collect();

        let eif_var = eif.mapv(|v| v.powi(2)).sum() / n as f64;
        let se = (eif_var / n as f64).sqrt();

        let t_stat = if se > 1e-10 { ate / se } else { 0.0 };
        let normal =
            Normal::new(0.0, 1.0).map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_value = 2.0 * (1.0 - normal.cdf(t_stat.abs()));
        let z = 1.959964;
        let ci = [ate - z * se, ate + z * se];

        Ok(TmleResult {
            ate,
            se,
            t_stat,
            p_value,
            ci,
            epsilon,
            initial_ate,
            propensity: g,
            initial_q: q_init,
            targeted_q: q_star,
            clever_covariate: h,
            eif,
            n_obs: n,
            n_confounders: k,
        })
    }
}
