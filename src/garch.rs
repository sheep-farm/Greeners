use ndarray::Array1;
use statrs::distribution::{ContinuousCDF, Normal as NormalDist};
use std::f64::consts::PI;
use std::fmt;

/// GARCH model type
#[derive(Debug, Clone, PartialEq)]
pub enum GarchModelType {
    GARCH,
    EGARCH,
    GJRGARCH,
}

/// Error distribution for GARCH models
#[derive(Debug, Clone, PartialEq)]
pub enum GarchDist {
    Normal,
    StudentT,
}

impl fmt::Display for GarchModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GarchModelType::GARCH => write!(f, "GARCH"),
            GarchModelType::EGARCH => write!(f, "EGARCH"),
            GarchModelType::GJRGARCH => write!(f, "GJR-GARCH"),
        }
    }
}

impl fmt::Display for GarchDist {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GarchDist::Normal => write!(f, "Normal"),
            GarchDist::StudentT => write!(f, "Student-t"),
        }
    }
}

/// Result from a GARCH-family model estimation
#[derive(Debug, Clone)]
pub struct GarchResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub z_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub conf_lower: Array1<f64>,
    pub conf_upper: Array1<f64>,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub n_iter: usize,
    pub converged: bool,
    pub residuals: Array1<f64>,
    pub conditional_variance: Array1<f64>,
    pub standardized_residuals: Array1<f64>,
    pub n_obs: usize,
    pub p: usize,
    pub q: usize,
    pub model_type: GarchModelType,
    pub dist: GarchDist,
    pub variable_names: Vec<String>,
}

impl GarchResult {
    /// Forecast conditional variance `steps` periods ahead
    pub fn forecast(&self, steps: usize) -> Array1<f64> {
        let n = self.n_obs;
        match self.model_type {
            GarchModelType::GARCH | GarchModelType::GJRGARCH => {
                // params: [mu, omega, alpha_1..alpha_q, beta_1..beta_p, (gamma_1..gamma_q), (nu)]
                let omega = self.params[1];
                let alphas: Vec<f64> = (0..self.q).map(|i| self.params[2 + i]).collect();
                let betas: Vec<f64> = (0..self.p).map(|i| self.params[2 + self.q + i]).collect();

                // For GJR, gammas follow betas
                let gammas: Vec<f64> = if self.model_type == GarchModelType::GJRGARCH {
                    (0..self.q)
                        .map(|i| self.params[2 + self.q + self.p + i])
                        .collect()
                } else {
                    vec![0.0; self.q]
                };

                let mut forecasts: Array1<f64> = Array1::zeros(steps);
                // We need recent squared residuals and variances
                let eps2: Vec<f64> = self.residuals.iter().map(|e| e * e).collect();
                let h: Vec<f64> = self.conditional_variance.to_vec();

                for s in 0..steps {
                    let mut val = omega;
                    for i in 0..self.q {
                        let e2 = if s == 0 {
                            if (n as isize - 1 - i as isize) >= 0 {
                                eps2[n - 1 - i]
                            } else {
                                0.0
                            }
                        } else {
                            // For multi-step, E[eps^2_{t+s-i}] = h_{t+s-i}
                            if s > i {
                                forecasts[s - 1 - i]
                            } else if (n as isize - 1 - (i - s) as isize) >= 0 {
                                eps2[n - 1 - (i - s)]
                            } else {
                                0.0
                            }
                        };
                        // For GJR: E[gamma * I(eps<0) * eps^2] = gamma * 0.5 * h (under normality)
                        val += alphas[i] * e2 + gammas[i] * 0.5 * e2;
                    }
                    for j in 0..self.p {
                        let h_val = if s == 0 {
                            if (n as isize - 1 - j as isize) >= 0 {
                                h[n - 1 - j]
                            } else {
                                0.0
                            }
                        } else if s > j {
                            forecasts[s - 1 - j]
                        } else if (n as isize - 1 - (j - s) as isize) >= 0 {
                            h[n - 1 - (j - s)]
                        } else {
                            0.0
                        };
                        val += betas[j] * h_val;
                    }
                    forecasts[s] = val.max(1e-10);
                }
                forecasts
            }
            GarchModelType::EGARCH => {
                // params: [mu, omega, alpha_1..alpha_q, gamma_1..gamma_q, beta_1..beta_p, (nu)]
                let omega = self.params[1];
                let alphas: Vec<f64> = (0..self.q).map(|i| self.params[2 + i]).collect();
                let gammas: Vec<f64> = (0..self.q).map(|i| self.params[2 + self.q + i]).collect();
                let betas: Vec<f64> = (0..self.p)
                    .map(|i| self.params[2 + 2 * self.q + i])
                    .collect();

                let log_h: Vec<f64> = self.conditional_variance.iter().map(|v| v.ln()).collect();
                let z: Vec<f64> = self.standardized_residuals.to_vec();
                let e_abs_z = (2.0_f64 / PI).sqrt();

                let mut forecasts: Array1<f64> = Array1::zeros(steps);
                for s in 0..steps {
                    let mut log_val = omega;
                    for i in 0..self.q {
                        if s == 0 && (n as isize - 1 - i as isize) >= 0 {
                            let zi = z[n - 1 - i];
                            log_val += alphas[i] * (zi.abs() - e_abs_z) + gammas[i] * zi;
                        }
                        // For multi-step, E[alpha*(|z|-sqrt(2/pi)) + gamma*z] = 0 under normality
                    }
                    for j in 0..self.p {
                        let lh = if s == 0 {
                            if (n as isize - 1 - j as isize) >= 0 {
                                log_h[n - 1 - j]
                            } else {
                                0.0
                            }
                        } else if s > j {
                            forecasts[s - 1 - j].ln()
                        } else if (n as isize - 1 - (j - s) as isize) >= 0 {
                            log_h[n - 1 - (j - s)]
                        } else {
                            0.0
                        };
                        log_val += betas[j] * lh;
                    }
                    forecasts[s] = log_val.exp().max(1e-10);
                }
                forecasts
            }
        }
    }

    /// Forecast volatility (square root of conditional variance)
    pub fn forecast_volatility(&self, steps: usize) -> Array1<f64> {
        self.forecast(steps).mapv(|v| v.sqrt())
    }
}

impl fmt::Display for GarchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let title = format!(
            " {}({},{}) - {} Results ",
            self.model_type, self.p, self.q, self.dist
        );
        writeln!(f, "\n{:=^78}", title)?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15}",
            "Model:",
            format!("{}({},{})", self.model_type, self.p, self.q),
            "No. Observations:",
            self.n_obs
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Distribution:", self.dist, "Log-Likelihood:", self.log_likelihood
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Method:", "MLE", "AIC:", self.aic
        )?;
        writeln!(
            f,
            "{:<20} {:>15} || {:<20} {:>15.4}",
            "Converged:",
            if self.converged { "Yes" } else { "No" },
            "BIC:",
            self.bic
        )?;
        writeln!(f, "{:<20} {:>15} ||", "Iterations:", self.n_iter)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} | {:>10} | {:>10} | {:>8} | {:>8} | {:>8} | {:>8}",
            "Variable", "coef", "std err", "z", "P>|z|", "[0.025", "0.975]"
        )?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.params.len() {
            let name = if i < self.variable_names.len() {
                self.variable_names[i].clone()
            } else {
                format!("param{}", i)
            };
            writeln!(
                f,
                "{:<12} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3} | {:>8.3} | {:>8.3}",
                name,
                self.params[i],
                self.std_errors[i],
                self.z_values[i],
                self.p_values[i],
                self.conf_lower[i],
                self.conf_upper[i]
            )?;
        }
        writeln!(f, "{:=^78}", "")?;
        Ok(())
    }
}

// ─── Helper functions ───────────────────────────────────────────────────────

fn normal_log_pdf(x: f64, _mean: f64, var: f64) -> f64 {
    -0.5 * (2.0 * PI).ln() - 0.5 * var.ln() - 0.5 * x * x / var
}

fn student_t_log_pdf(x: f64, var: f64, nu: f64) -> f64 {
    use statrs::function::gamma::ln_gamma;
    let s = var.sqrt();
    ln_gamma((nu + 1.0) / 2.0)
        - ln_gamma(nu / 2.0)
        - 0.5 * ((nu - 2.0) * PI).ln()
        - s.ln()
        - (nu + 1.0) / 2.0 * (1.0 + x * x / (var * (nu - 2.0))).ln()
}

fn numerical_gradient<F: Fn(&[f64]) -> f64>(f: &F, params: &[f64], eps: f64) -> Vec<f64> {
    let n = params.len();
    let mut grad = vec![0.0; n];
    let mut p = params.to_vec();
    for i in 0..n {
        let orig = p[i];
        p[i] = orig + eps;
        let f_plus = f(&p);
        p[i] = orig - eps;
        let f_minus = f(&p);
        p[i] = orig;
        grad[i] = (f_plus - f_minus) / (2.0 * eps);
    }
    grad
}

fn numerical_hessian<F: Fn(&[f64]) -> f64 + ?Sized>(
    f: &F,
    params: &[f64],
    eps: f64,
) -> Vec<Vec<f64>> {
    let n = params.len();
    let mut hess = vec![vec![0.0; n]; n];
    let mut p = params.to_vec();
    for i in 0..n {
        for j in i..n {
            let orig_i = p[i];
            let orig_j = p[j];

            p[i] = orig_i + eps;
            p[j] = orig_j + eps;
            let f_pp = f(&p);

            p[i] = orig_i + eps;
            p[j] = orig_j - eps;
            let f_pm = f(&p);

            p[i] = orig_i - eps;
            p[j] = orig_j + eps;
            let f_mp = f(&p);

            p[i] = orig_i - eps;
            p[j] = orig_j - eps;
            let f_mm = f(&p);

            p[i] = orig_i;
            p[j] = orig_j;

            hess[i][j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps);
            hess[j][i] = hess[i][j];
        }
    }
    hess
}

/// Invert a square matrix (small, for Hessian)
fn invert_matrix(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = m[i][j];
        }
        aug[i][n + i] = 1.0;
    }
    for col in 0..n {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for (row, aug_row) in aug.iter().enumerate().skip(col + 1) {
            if aug_row[col].abs() > max_val {
                max_val = aug_row[col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return None;
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        for val in &mut aug[col] {
            *val /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                let col_row: Vec<f64> = aug[col].clone();
                for (val, c) in aug[row].iter_mut().zip(col_row.iter()) {
                    *val -= factor * c;
                }
            }
        }
    }
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    Some(inv)
}

fn compute_inference(
    params: &Array1<f64>,
    std_errors: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let normal = NormalDist::new(0.0, 1.0).unwrap();
    let n = params.len();
    let mut z_values = Array1::zeros(n);
    let mut p_values = Array1::zeros(n);
    let mut conf_lower = Array1::zeros(n);
    let mut conf_upper = Array1::zeros(n);

    for i in 0..n {
        if std_errors[i] > 0.0 {
            z_values[i] = params[i] / std_errors[i];
            p_values[i] = 2.0 * (1.0 - normal.cdf(z_values[i].abs()));
        } else {
            z_values[i] = f64::NAN;
            p_values[i] = f64::NAN;
        }
        conf_lower[i] = params[i] - 1.96 * std_errors[i];
        conf_upper[i] = params[i] + 1.96 * std_errors[i];
    }
    (z_values, p_values, conf_lower)
}

fn compute_std_errors_from_hessian(neg_ll: &dyn Fn(&[f64]) -> f64, params: &[f64]) -> Array1<f64> {
    let hess = numerical_hessian(neg_ll, params, 1e-5);
    let n = params.len();
    if let Some(inv) = invert_matrix(&hess) {
        Array1::from_vec((0..n).map(|i| inv[i][i].abs().sqrt()).collect())
    } else {
        Array1::from_vec(vec![f64::NAN; n])
    }
}

struct BuildResultArgs {
    residuals: Array1<f64>,
    cond_var: Array1<f64>,
    p: usize,
    q: usize,
    model_type: GarchModelType,
    dist: GarchDist,
    variable_names: Vec<String>,
    log_likelihood: f64,
    n_iter: usize,
    converged: bool,
}

fn build_result(
    params_vec: &[f64],
    y: &Array1<f64>,
    args: BuildResultArgs,
    neg_ll: &dyn Fn(&[f64]) -> f64,
) -> GarchResult {
    let BuildResultArgs {
        residuals,
        cond_var,
        p,
        q,
        model_type,
        dist,
        variable_names,
        log_likelihood,
        n_iter,
        converged,
    } = args;
    let n_obs = y.len();
    let k = params_vec.len() as f64;
    let n_f = n_obs as f64;

    let params = Array1::from_vec(params_vec.to_vec());
    let std_errors = compute_std_errors_from_hessian(neg_ll, params_vec);
    let (z_values, p_values, conf_lower) = compute_inference(&params, &std_errors);
    let conf_upper = &params + &(&std_errors * 1.96);

    let standardized_residuals = Array1::from_vec(
        residuals
            .iter()
            .zip(cond_var.iter())
            .map(|(e, h)| e / h.sqrt().max(1e-10))
            .collect(),
    );

    GarchResult {
        params,
        std_errors,
        z_values,
        p_values,
        conf_lower,
        conf_upper,
        log_likelihood,
        aic: -2.0 * log_likelihood + 2.0 * k,
        bic: -2.0 * log_likelihood + k * n_f.ln(),
        n_iter,
        converged,
        residuals,
        conditional_variance: cond_var,
        standardized_residuals,
        n_obs,
        p,
        q,
        model_type,
        dist,
        variable_names,
    }
}

// ─── GARCH optimization core ────────────────────────────────────────────────

fn optimize(
    neg_ll: impl Fn(&[f64]) -> f64,
    init: &[f64],
    max_iter: usize,
    constrain: impl Fn(&mut [f64]),
) -> (Vec<f64>, usize, bool) {
    let mut params = init.to_vec();
    constrain(&mut params);
    let mut best_ll = neg_ll(&params);
    let mut best_params = params.clone();

    for iter in 0..max_iter {
        let grad = numerical_gradient(&neg_ll, &params, 1e-5);
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < 1e-6 {
            return (params, iter + 1, true);
        }

        // Steepest descent with line search
        let mut step = 1.0;
        let mut improved = false;
        for _ in 0..30 {
            let mut candidate: Vec<f64> = params
                .iter()
                .zip(grad.iter())
                .map(|(p, g)| p - step * g)
                .collect();
            constrain(&mut candidate);
            let ll = neg_ll(&candidate);
            if ll < best_ll && ll.is_finite() {
                best_ll = ll;
                best_params = candidate.clone();
                params = candidate;
                improved = true;
                break;
            }
            step *= 0.5;
        }

        if !improved {
            // Try smaller step anyway
            let mut candidate: Vec<f64> = params
                .iter()
                .zip(grad.iter())
                .map(|(p, g)| p - 1e-6 * g)
                .collect();
            constrain(&mut candidate);
            let ll = neg_ll(&candidate);
            if ll < best_ll && ll.is_finite() {
                best_ll = ll;
                best_params = candidate.clone();
                params = candidate;
            } else {
                // Check convergence by param change
                return (best_params, iter + 1, true);
            }
        }

        // Check param change convergence
        let param_change: f64 = params
            .iter()
            .zip(best_params.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();
        if param_change < 1e-8 && iter > 10 {
            return (params, iter + 1, true);
        }
    }
    (best_params, max_iter, false)
}

// ─── Conditional variance computation ────────────────────────────────────────

fn garch_conditional_variance(
    eps: &[f64],
    omega: f64,
    alphas: &[f64],
    betas: &[f64],
    var_init: f64,
) -> Vec<f64> {
    let n = eps.len();
    let q = alphas.len();
    let p = betas.len();
    let mut h = vec![var_init; n];
    for t in 1..n {
        let mut val = omega;
        for i in 0..q {
            if t > i {
                val += alphas[i] * eps[t - 1 - i] * eps[t - 1 - i];
            }
        }
        for j in 0..p {
            if t > j {
                val += betas[j] * h[t - 1 - j];
            }
        }
        h[t] = val.max(1e-10);
    }
    h
}

fn egarch_conditional_variance(
    eps: &[f64],
    omega: f64,
    alphas: &[f64],
    gammas: &[f64],
    betas: &[f64],
    var_init: f64,
) -> Vec<f64> {
    let n = eps.len();
    let q = alphas.len();
    let p = betas.len();
    let e_abs_z = (2.0_f64 / PI).sqrt();
    let mut h = vec![var_init; n];
    let mut log_h = vec![var_init.ln(); n];
    for t in 1..n {
        let mut log_val = omega;
        for i in 0..q {
            if t > i {
                let z = eps[t - 1 - i] / h[t - 1 - i].sqrt().max(1e-10);
                log_val += alphas[i] * (z.abs() - e_abs_z) + gammas[i] * z;
            }
        }
        for j in 0..p {
            if t > j {
                log_val += betas[j] * log_h[t - 1 - j];
            }
        }
        log_h[t] = log_val;
        h[t] = log_val.exp().clamp(1e-10, 1e10);
    }
    h
}

fn gjrgarch_conditional_variance(
    eps: &[f64],
    omega: f64,
    alphas: &[f64],
    betas: &[f64],
    gammas: &[f64],
    var_init: f64,
) -> Vec<f64> {
    let n = eps.len();
    let q = alphas.len();
    let p = betas.len();
    let mut h = vec![var_init; n];
    for t in 1..n {
        let mut val = omega;
        for i in 0..q {
            if t > i {
                let e = eps[t - 1 - i];
                let e2 = e * e;
                val += alphas[i] * e2;
                if e < 0.0 {
                    val += gammas[i] * e2;
                }
            }
        }
        for j in 0..p {
            if t > j {
                val += betas[j] * h[t - 1 - j];
            }
        }
        h[t] = val.max(1e-10);
    }
    h
}

// ─── GARCH struct ────────────────────────────────────────────────────────────

/// Standard GARCH(p,q) model. ARCH(q) is GARCH(0,q).
pub struct GARCH;

impl GARCH {
    /// Fit GARCH(p,q) with Normal errors
    pub fn fit(y: &Array1<f64>, p: usize, q: usize) -> Result<GarchResult, String> {
        if y.len() < 10 {
            return Err("Need at least 10 observations".to_string());
        }
        if q == 0 {
            return Err("q must be >= 1 for GARCH".to_string());
        }

        let n = y.len();
        let mean_y = y.iter().sum::<f64>() / n as f64;
        let var_y = y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / n as f64;

        // n_params: mu + omega + q alphas + p betas
        let n_params = 1 + 1 + q + p;
        let mut init = vec![0.0; n_params];
        init[0] = mean_y; // mu
        init[1] = 0.1 * var_y; // omega
        for i in 0..q {
            init[2 + i] = 0.05; // alphas
        }
        for j in 0..p {
            init[2 + q + j] = 0.85 / p.max(1) as f64; // betas
        }

        let y_clone = y.clone();
        let neg_ll = move |params: &[f64]| -> f64 {
            let mu = params[0];
            let omega = params[1];
            let alphas: Vec<f64> = (0..q).map(|i| params[2 + i]).collect();
            let betas: Vec<f64> = (0..p).map(|i| params[2 + q + i]).collect();
            let eps: Vec<f64> = y_clone.iter().map(|v| v - mu).collect();
            let var_init = eps.iter().map(|e| e * e).sum::<f64>() / eps.len() as f64;
            let h = garch_conditional_variance(&eps, omega, &alphas, &betas, var_init);
            let mut ll = 0.0;
            for t in 0..eps.len() {
                ll += normal_log_pdf(eps[t], 0.0, h[t]);
            }
            if ll.is_finite() {
                -ll
            } else {
                1e18
            }
        };

        let constrain = move |params: &mut [f64]| {
            params[1] = params[1].max(1e-10); // omega > 0
            for i in 0..q {
                params[2 + i] = params[2 + i].max(0.0); // alpha >= 0
            }
            for j in 0..p {
                params[2 + q + j] = params[2 + q + j].max(0.0); // beta >= 0
            }
            // Stationarity: sum(alpha) + sum(beta) < 1
            let sum_ab: f64 = (0..q).map(|i| params[2 + i]).sum::<f64>()
                + (0..p).map(|j| params[2 + q + j]).sum::<f64>();
            if sum_ab >= 0.9999 {
                let scale = 0.999 / sum_ab;
                for i in 0..q {
                    params[2 + i] *= scale;
                }
                for j in 0..p {
                    params[2 + q + j] *= scale;
                }
            }
        };

        let (opt_params, n_iter, converged) = optimize(&neg_ll, &init, 500, constrain);

        let mu = opt_params[0];
        let omega = opt_params[1];
        let alphas: Vec<f64> = (0..q).map(|i| opt_params[2 + i]).collect();
        let betas: Vec<f64> = (0..p).map(|i| opt_params[2 + q + i]).collect();
        let final_eps: Vec<f64> = y.iter().map(|v| v - mu).collect();
        let var_init = final_eps.iter().map(|e| e * e).sum::<f64>() / final_eps.len() as f64;
        let h = garch_conditional_variance(&final_eps, omega, &alphas, &betas, var_init);
        let log_likelihood = -neg_ll(&opt_params);

        let residuals = Array1::from_vec(final_eps);
        let cond_var = Array1::from_vec(h);

        let mut names = vec!["mu".to_string(), "omega".to_string()];
        for i in 0..q {
            names.push(format!("alpha[{}]", i + 1));
        }
        for j in 0..p {
            names.push(format!("beta[{}]", j + 1));
        }

        Ok(build_result(
            &opt_params,
            y,
            BuildResultArgs {
                residuals,
                cond_var,
                p,
                q,
                model_type: GarchModelType::GARCH,
                dist: GarchDist::Normal,
                variable_names: names,
                log_likelihood,
                n_iter,
                converged,
            },
            &neg_ll,
        ))
    }

    /// Fit GARCH(p,q) with Student-t errors (df estimated)
    pub fn fit_t(y: &Array1<f64>, p: usize, q: usize) -> Result<GarchResult, String> {
        if y.len() < 10 {
            return Err("Need at least 10 observations".to_string());
        }
        if q == 0 {
            return Err("q must be >= 1 for GARCH".to_string());
        }

        let n = y.len();
        let mean_y = y.iter().sum::<f64>() / n as f64;
        let var_y = y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / n as f64;

        // n_params: mu + omega + q alphas + p betas + nu
        let n_params = 1 + 1 + q + p + 1;
        let mut init = vec![0.0; n_params];
        init[0] = mean_y;
        init[1] = 0.1 * var_y;
        for i in 0..q {
            init[2 + i] = 0.05;
        }
        for j in 0..p {
            init[2 + q + j] = 0.85 / p.max(1) as f64;
        }
        init[n_params - 1] = 8.0; // nu

        let y_clone = y.clone();
        let neg_ll = move |params: &[f64]| -> f64 {
            let mu = params[0];
            let omega = params[1];
            let alphas: Vec<f64> = (0..q).map(|i| params[2 + i]).collect();
            let betas: Vec<f64> = (0..p).map(|i| params[2 + q + i]).collect();
            let nu = params[n_params - 1];
            if nu <= 2.0 {
                return 1e18;
            }
            let eps: Vec<f64> = y_clone.iter().map(|v| v - mu).collect();
            let var_init = eps.iter().map(|e| e * e).sum::<f64>() / eps.len() as f64;
            let h = garch_conditional_variance(&eps, omega, &alphas, &betas, var_init);
            let mut ll = 0.0;
            for t in 0..eps.len() {
                ll += student_t_log_pdf(eps[t], h[t], nu);
            }
            if ll.is_finite() {
                -ll
            } else {
                1e18
            }
        };

        let n_p = n_params;
        let constrain = move |params: &mut [f64]| {
            params[1] = params[1].max(1e-10);
            for i in 0..q {
                params[2 + i] = params[2 + i].max(0.0);
            }
            for j in 0..p {
                params[2 + q + j] = params[2 + q + j].max(0.0);
            }
            let sum_ab: f64 = (0..q).map(|i| params[2 + i]).sum::<f64>()
                + (0..p).map(|j| params[2 + q + j]).sum::<f64>();
            if sum_ab >= 0.9999 {
                let scale = 0.999 / sum_ab;
                for i in 0..q {
                    params[2 + i] *= scale;
                }
                for j in 0..p {
                    params[2 + q + j] *= scale;
                }
            }
            params[n_p - 1] = params[n_p - 1].clamp(2.1, 100.0);
        };

        let (opt_params, n_iter, converged) = optimize(&neg_ll, &init, 500, constrain);

        let mu = opt_params[0];
        let omega = opt_params[1];
        let alphas: Vec<f64> = (0..q).map(|i| opt_params[2 + i]).collect();
        let betas: Vec<f64> = (0..p).map(|i| opt_params[2 + q + i]).collect();
        let final_eps: Vec<f64> = y.iter().map(|v| v - mu).collect();
        let var_init = final_eps.iter().map(|e| e * e).sum::<f64>() / final_eps.len() as f64;
        let h = garch_conditional_variance(&final_eps, omega, &alphas, &betas, var_init);
        let log_likelihood = -neg_ll(&opt_params);

        let residuals = Array1::from_vec(final_eps);
        let cond_var = Array1::from_vec(h);

        let mut names = vec!["mu".to_string(), "omega".to_string()];
        for i in 0..q {
            names.push(format!("alpha[{}]", i + 1));
        }
        for j in 0..p {
            names.push(format!("beta[{}]", j + 1));
        }
        names.push("nu".to_string());

        Ok(build_result(
            &opt_params,
            y,
            BuildResultArgs {
                residuals,
                cond_var,
                p,
                q,
                model_type: GarchModelType::GARCH,
                dist: GarchDist::StudentT,
                variable_names: names,
                log_likelihood,
                n_iter,
                converged,
            },
            &neg_ll,
        ))
    }
}

// ─── EGARCH struct ───────────────────────────────────────────────────────────

/// EGARCH(p,q) model — log-variance specification, no positivity constraints
pub struct EGARCH;

impl EGARCH {
    /// Fit EGARCH(p,q) with Normal errors
    pub fn fit(y: &Array1<f64>, p: usize, q: usize) -> Result<GarchResult, String> {
        if y.len() < 10 {
            return Err("Need at least 10 observations".to_string());
        }
        if q == 0 {
            return Err("q must be >= 1".to_string());
        }

        let n = y.len();
        let mean_y = y.iter().sum::<f64>() / n as f64;
        let var_y = y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / n as f64;

        // params: mu, omega, alpha_1..q, gamma_1..q, beta_1..p
        let n_params = 1 + 1 + q + q + p;
        let mut init = vec![0.0; n_params];
        init[0] = mean_y;
        init[1] = var_y.ln() * 0.1; // omega (log scale)
        for i in 0..q {
            init[2 + i] = 0.1; // alpha
            init[2 + q + i] = -0.05; // gamma (leverage, typically negative)
        }
        for j in 0..p {
            init[2 + 2 * q + j] = 0.9 / p.max(1) as f64; // beta
        }

        let y_clone = y.clone();
        let neg_ll = move |params: &[f64]| -> f64 {
            let mu = params[0];
            let omega = params[1];
            let alphas: Vec<f64> = (0..q).map(|i| params[2 + i]).collect();
            let gammas: Vec<f64> = (0..q).map(|i| params[2 + q + i]).collect();
            let betas: Vec<f64> = (0..p).map(|i| params[2 + 2 * q + i]).collect();
            let eps: Vec<f64> = y_clone.iter().map(|v| v - mu).collect();
            let var_init = eps.iter().map(|e| e * e).sum::<f64>() / eps.len() as f64;
            let h = egarch_conditional_variance(&eps, omega, &alphas, &gammas, &betas, var_init);
            let mut ll = 0.0;
            for t in 0..eps.len() {
                ll += normal_log_pdf(eps[t], 0.0, h[t]);
            }
            if ll.is_finite() {
                -ll
            } else {
                1e18
            }
        };

        let constrain = move |params: &mut [f64]| {
            // EGARCH has no positivity constraints on alpha/gamma
            // But beta should have |sum| < 1 for stationarity
            let sum_b: f64 = (0..p).map(|j| params[2 + 2 * q + j].abs()).sum::<f64>();
            if sum_b >= 0.9999 {
                let scale = 0.999 / sum_b;
                for j in 0..p {
                    params[2 + 2 * q + j] *= scale;
                }
            }
        };

        let (opt_params, n_iter, converged) = optimize(&neg_ll, &init, 500, constrain);

        let mu = opt_params[0];
        let omega = opt_params[1];
        let alphas: Vec<f64> = (0..q).map(|i| opt_params[2 + i]).collect();
        let gammas: Vec<f64> = (0..q).map(|i| opt_params[2 + q + i]).collect();
        let betas: Vec<f64> = (0..p).map(|i| opt_params[2 + 2 * q + i]).collect();
        let final_eps: Vec<f64> = y.iter().map(|v| v - mu).collect();
        let var_init = final_eps.iter().map(|e| e * e).sum::<f64>() / final_eps.len() as f64;
        let h = egarch_conditional_variance(&final_eps, omega, &alphas, &gammas, &betas, var_init);
        let log_likelihood = -neg_ll(&opt_params);

        let residuals = Array1::from_vec(final_eps);
        let cond_var = Array1::from_vec(h);

        let mut names = vec!["mu".to_string(), "omega".to_string()];
        for i in 0..q {
            names.push(format!("alpha[{}]", i + 1));
        }
        for i in 0..q {
            names.push(format!("gamma[{}]", i + 1));
        }
        for j in 0..p {
            names.push(format!("beta[{}]", j + 1));
        }

        Ok(build_result(
            &opt_params,
            y,
            BuildResultArgs {
                residuals,
                cond_var,
                p,
                q,
                model_type: GarchModelType::EGARCH,
                dist: GarchDist::Normal,
                variable_names: names,
                log_likelihood,
                n_iter,
                converged,
            },
            &neg_ll,
        ))
    }
}

// ─── GJR-GARCH struct ────────────────────────────────────────────────────────

/// GJR-GARCH(p,q) model — asymmetric GARCH with leverage effect
pub struct GJRGARCH;

impl GJRGARCH {
    /// Fit GJR-GARCH(p,q) with Normal errors
    pub fn fit(y: &Array1<f64>, p: usize, q: usize) -> Result<GarchResult, String> {
        if y.len() < 10 {
            return Err("Need at least 10 observations".to_string());
        }
        if q == 0 {
            return Err("q must be >= 1".to_string());
        }

        let n = y.len();
        let mean_y = y.iter().sum::<f64>() / n as f64;
        let var_y = y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / n as f64;

        // params: mu, omega, alpha_1..q, beta_1..p, gamma_1..q
        let n_params = 1 + 1 + q + p + q;
        let mut init = vec![0.0; n_params];
        init[0] = mean_y;
        init[1] = 0.1 * var_y;
        for i in 0..q {
            init[2 + i] = 0.05; // alpha
        }
        for j in 0..p {
            init[2 + q + j] = 0.85 / p.max(1) as f64; // beta
        }
        for i in 0..q {
            init[2 + q + p + i] = 0.05; // gamma
        }

        let y_clone = y.clone();
        let neg_ll = move |params: &[f64]| -> f64 {
            let mu = params[0];
            let omega = params[1];
            let alphas: Vec<f64> = (0..q).map(|i| params[2 + i]).collect();
            let betas: Vec<f64> = (0..p).map(|i| params[2 + q + i]).collect();
            let gammas: Vec<f64> = (0..q).map(|i| params[2 + q + p + i]).collect();
            let eps: Vec<f64> = y_clone.iter().map(|v| v - mu).collect();
            let var_init = eps.iter().map(|e| e * e).sum::<f64>() / eps.len() as f64;
            let h = gjrgarch_conditional_variance(&eps, omega, &alphas, &betas, &gammas, var_init);
            let mut ll = 0.0;
            for t in 0..eps.len() {
                ll += normal_log_pdf(eps[t], 0.0, h[t]);
            }
            if ll.is_finite() {
                -ll
            } else {
                1e18
            }
        };

        let constrain = move |params: &mut [f64]| {
            params[1] = params[1].max(1e-10); // omega > 0
            for i in 0..q {
                params[2 + i] = params[2 + i].max(0.0); // alpha >= 0
            }
            for j in 0..p {
                params[2 + q + j] = params[2 + q + j].max(0.0); // beta >= 0
            }
            for i in 0..q {
                params[2 + q + p + i] = params[2 + q + p + i].max(0.0); // gamma >= 0
            }
            // Stationarity: sum(alpha) + sum(beta) + 0.5*sum(gamma) < 1
            let sum_abg: f64 = (0..q).map(|i| params[2 + i]).sum::<f64>()
                + (0..p).map(|j| params[2 + q + j]).sum::<f64>()
                + 0.5 * (0..q).map(|i| params[2 + q + p + i]).sum::<f64>();
            if sum_abg >= 0.9999 {
                let scale = 0.999 / sum_abg;
                for i in 0..q {
                    params[2 + i] *= scale;
                }
                for j in 0..p {
                    params[2 + q + j] *= scale;
                }
                for i in 0..q {
                    params[2 + q + p + i] *= scale;
                }
            }
        };

        let (opt_params, n_iter, converged) = optimize(&neg_ll, &init, 500, constrain);

        let mu = opt_params[0];
        let omega = opt_params[1];
        let alphas: Vec<f64> = (0..q).map(|i| opt_params[2 + i]).collect();
        let betas: Vec<f64> = (0..p).map(|i| opt_params[2 + q + i]).collect();
        let gammas: Vec<f64> = (0..q).map(|i| opt_params[2 + q + p + i]).collect();
        let final_eps: Vec<f64> = y.iter().map(|v| v - mu).collect();
        let var_init = final_eps.iter().map(|e| e * e).sum::<f64>() / final_eps.len() as f64;
        let h =
            gjrgarch_conditional_variance(&final_eps, omega, &alphas, &betas, &gammas, var_init);
        let log_likelihood = -neg_ll(&opt_params);

        let residuals = Array1::from_vec(final_eps);
        let cond_var = Array1::from_vec(h);

        let mut names = vec!["mu".to_string(), "omega".to_string()];
        for i in 0..q {
            names.push(format!("alpha[{}]", i + 1));
        }
        for j in 0..p {
            names.push(format!("beta[{}]", j + 1));
        }
        for i in 0..q {
            names.push(format!("gamma[{}]", i + 1));
        }

        Ok(build_result(
            &opt_params,
            y,
            BuildResultArgs {
                residuals,
                cond_var,
                p,
                q,
                model_type: GarchModelType::GJRGARCH,
                dist: GarchDist::Normal,
                variable_names: names,
                log_likelihood,
                n_iter,
                converged,
            },
            &neg_ll,
        ))
    }
}
