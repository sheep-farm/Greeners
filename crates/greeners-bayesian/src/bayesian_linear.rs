//! Bayesian Linear Regression with conjugate Normal-Inverse-Gamma prior.
//!
//! Model:
//!   y = X * beta + epsilon,  epsilon ~ N(0, sigma^2)
//!
//! Prior (conjugate):
//!   beta | sigma^2 ~ N(beta_0, sigma^2 * V_0)
//!   sigma^2 ~ Inverse-Gamma(a_0, b_0)
//!
//! Posterior (closed-form):
//!   V_n = (V_0^{-1} + X'X)^{-1}
//!   beta_n = V_n * (V_0^{-1} * beta_0 + X'y)
//!   a_n = a_0 + n/2
//!   b_n = b_0 + 0.5 * (y'y + beta_0' * V_0^{-1} * beta_0 - beta_n' * V_n^{-1} * beta_n)
//!
//! Default prior: diffuse proper: beta_0 = 0, V_0 = 1000*I,
//! a_0 = 0.001, b_0 = 0.001.

use faer::linalg::solvers::{Llt, Qr, SolveLstsq};
use faer::linalg::triangular_solve::{
    solve_lower_triangular_in_place, solve_upper_triangular_in_place,
};
use faer::{Mat, Par, Side};
use greeners_core::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, StudentsT};
use statrs::function::gamma::ln_gamma;
use std::fmt;

/// Result of Bayesian linear regression.
#[derive(Debug)]
pub struct BayesianLinearResult {
    /// Posterior mean of coefficients
    pub beta: Array1<f64>,
    /// Posterior covariance of coefficients
    pub beta_cov: Array2<f64>,
    /// Posterior mean of sigma^2
    pub sigma2: f64,
    /// Posterior shape (a_n)
    pub sigma2_shape: f64,
    /// Posterior scale (b_n)
    pub sigma2_scale: f64,
    /// Prior mean of coefficients
    pub beta_prior: Array1<f64>,
    /// Conditional prior covariance multiplier: Var(beta | sigma²) = sigma² * v_prior
    pub v_prior: Array2<f64>,
    /// Prior shape (a_0)
    pub a_prior: f64,
    /// Prior scale (b_0)
    pub b_prior: f64,
    /// 95% credible intervals for coefficients
    pub beta_ci: Array2<f64>,
    /// Posterior probability that each coefficient > 0
    pub p_positive: Array1<f64>,
    /// Number of observations
    pub n_obs: usize,
    /// Number of predictors (including intercept)
    pub n_pred: usize,
    /// Marginal likelihood
    pub log_marginal: f64,
    /// In-sample R-squared
    pub r_squared: f64,
    /// Fitted values
    pub fitted: Array1<f64>,
    /// Variable names
    pub variable_names: Vec<String>,
}

impl fmt::Display for BayesianLinearResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Bayesian Linear Regression ")?;
        writeln!(f, "Conjugate Normal-Inverse-Gamma prior")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Predictors:", self.n_pred)?;
        writeln!(f, "{:<20} {:>12.6}", "sigma² (posterior):", self.sigma2)?;
        writeln!(f, "{:<20} {:>12.6}", "R²:", self.r_squared)?;
        writeln!(
            f,
            "{:<20} {:>12.4}",
            "Log marginal lik.:", self.log_marginal
        )?;

        // Coefficients
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "  {:<14} {:>10} {:>10} {:>10} {:>10}",
            "Variable", "Post. mean", "SD", "2.5%", "97.5%"
        )?;
        writeln!(f, "{:-^78}", "")?;
        for (j, name) in self.variable_names.iter().enumerate() {
            let sd = self.beta_cov[(j, j)].sqrt();
            writeln!(
                f,
                "  {:<14} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                name,
                self.beta[j],
                sd,
                self.beta_ci[(j, 0)],
                self.beta_ci[(j, 1)]
            )?;
        }

        // Posterior probabilities
        writeln!(f, "\n  P(coef > 0):")?;
        for (j, name) in self.variable_names.iter().enumerate() {
            writeln!(f, "  {:<14} {:>10.4}", name, self.p_positive[j])?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct BayesianLinear;

impl BayesianLinear {
    /// Estimate Bayesian linear regression with conjugate prior.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (n)
    /// * `x` - Design matrix (n x k), WITHOUT intercept (added internally)
    /// * `variable_names` - Optional variable names
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<BayesianLinearResult, GreenersError> {
        Self::fit_with_prior(y, x, None, None, None, None, variable_names)
    }

    /// Estimate with a proper normal-inverse-gamma prior, including the intercept.
    ///
    /// The returned covariance is marginal over sigma²; credible intervals use
    /// the distinct Student-t scale. Invalid priors, non-finite calculations or
    /// nonexistent requested posterior moments return an error. The existing
    /// requirement n >= k + 2 is retained even though a proper prior can identify
    /// some smaller samples.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_with_prior(
        y: &Array1<f64>,
        x: &Array2<f64>,
        beta_prior: Option<&Array1<f64>>,
        v_prior: Option<&Array2<f64>>,
        a_prior: Option<f64>,
        b_prior: Option<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<BayesianLinearResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if x.nrows() != n {
            return Err(GreenersError::ShapeMismatch(
                "BayesianLinear: y and x must have same n".into(),
            ));
        }
        if n < k + 2 {
            return Err(GreenersError::InvalidOperation(
                "BayesianLinear: need more observations than predictors".into(),
            ));
        }

        if y.iter().chain(x.iter()).any(|v| !v.is_finite()) {
            return Err(GreenersError::InvalidOperation(
                "BayesianLinear: data must be finite".into(),
            ));
        }
        if variable_names
            .as_ref()
            .is_some_and(|names| names.len() != k)
        {
            return Err(GreenersError::ShapeMismatch(
                "BayesianLinear: need one name per predictor, excluding the intercept".into(),
            ));
        }

        // Add intercept column
        let p = k + 1; // intercept + k predictors
        let mut x_full = Array2::zeros((n, p));
        for i in 0..n {
            x_full[(i, 0)] = 1.0;
            for j in 0..k {
                x_full[(i, j + 1)] = x[(i, j)];
            }
        }

        let mut names =
            variable_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());
        names.insert(0, "Intercept".to_string());

        // Default proper prior.
        let beta_0 = beta_prior.cloned().unwrap_or_else(|| Array1::zeros(p));
        let v_0 = v_prior
            .cloned()
            .unwrap_or_else(|| Array2::<f64>::eye(p) * 1000.0);
        let a_0 = a_prior.unwrap_or(0.001);
        let b_0 = b_prior.unwrap_or(0.001);

        if beta_0.len() != p || v_0.nrows() != p || v_0.ncols() != p {
            return Err(GreenersError::ShapeMismatch(
                "BayesianLinear: prior dimensions don't match".into(),
            ));
        }

        if beta_0.iter().any(|v| !v.is_finite())
            || !a_0.is_finite()
            || a_0 <= 0.0
            || !b_0.is_finite()
            || b_0 <= 0.0
        {
            return Err(GreenersError::InvalidOperation(
                "BayesianLinear: finite prior mean and positive finite inverse-gamma parameters required".into(),
            ));
        }

        // V0 = L L', so W = L^-1 supplies the prior rows of the augmented
        // least-squares problem [X; W] beta = [y; W beta0]. The posterior mean
        // uses this direct response, without normal equations or response centring.
        let prior = factor_spd(&v_0, "prior scale")?;
        let mut whitener = Mat::identity(p, p);
        solve_lower_triangular_in_place(prior.L(), whitener.as_mut(), Par::Seq);
        let whitener = Array2::from_shape_fn((p, p), |(i, j)| whitener[(i, j)]);

        // Project each augmented predictor column off the augmented intercept.
        // This respects both likelihood and prior geometry: a concentrated
        // prior must not acquire a poorly conditioned transform from X alone.
        // Normalise before inner products to avoid squaring large prior weights.
        // beta = T gamma,
        // where T has unit determinant and beta[0] = gamma[0] - centre' gamma.
        // Transform the prior rows by the SAME T, preserving the declared prior.
        let mut centres = Array1::zeros(p);
        let intercept_norm = whitener
            .column(0)
            .iter()
            .fold((n as f64).sqrt(), |norm, &v| norm.hypot(v));
        if !intercept_norm.is_finite() {
            return Err(GreenersError::InvalidOperation(
                "BayesianLinear: augmented intercept is not representable".into(),
            ));
        }
        for j in 1..p {
            centres[j] = (0..p)
                .map(|i| (whitener[(i, 0)] / intercept_norm) * (whitener[(i, j)] / intercept_norm))
                .sum::<f64>()
                + x_full
                    .column(j)
                    .iter()
                    .map(|v| (v / intercept_norm) / intercept_norm)
                    .sum::<f64>();
        }
        let centred_design = Array2::from_shape_fn((n, p), |(i, j)| x_full[(i, j)] - centres[j]);
        let prior_design = Array2::from_shape_fn((p, p), |(i, j)| {
            whitener[(i, j)] - whitener[(i, 0)] * centres[j]
        });
        let prior_rhs = Array1::from_shape_fn(p, |i| {
            compensated_sum(0.0, (0..p).map(|j| (whitener[(i, j)], beta_0[j])))
        });
        // Prior rows first also preserve small likelihood contributions when
        // the proper prior is highly concentrated.
        let mut augmented = Mat::from_fn(n + p, p, |i, j| {
            if i < p {
                prior_design[(i, j)]
            } else {
                centred_design[(i - p, j)]
            }
        });
        // Solve separately for the correction delta = beta - beta0 using the
        // SAME factorisation. W delta computed this way does not amplify ulps
        // in the returned, rounded beta under a concentrated prior. It also
        // supplies means when prior-plus-correction addition does not cancel;
        // the direct RHS preserves y under a distant diffuse prior.
        let correction_response = Array1::from_shape_fn(n, |i| {
            compensated_sum(y[i], (0..p).map(|j| (-x_full[(i, j)], beta_0[j])))
        });
        let rhs = Mat::from_fn(n + p, 2, |i, j| match (i < p, j) {
            (true, 0) => prior_rhs[i],
            (true, _) => 0.0,
            (false, 0) => y[i - p],
            (false, _) => correction_response[i - p],
        });
        let mut column_scales = Vec::with_capacity(p);
        for j in 0..p {
            let scale = (0..n + p).fold(0.0_f64, |norm, i| norm.hypot(augmented[(i, j)]));
            if !scale.is_finite() || scale <= 0.0 {
                return Err(GreenersError::InvalidOperation(
                    "BayesianLinear: augmented posterior design is not representable".into(),
                ));
            }
            column_scales.push(scale);
            for i in 0..n + p {
                augmented[(i, j)] /= scale;
            }
        }
        if (0..n + p).any(|i| (0..2).any(|j| !rhs[(i, j)].is_finite())) {
            return Err(GreenersError::InvalidOperation(
                "BayesianLinear: weighted prior or response is not representable".into(),
            ));
        }
        let posterior = Qr::new(augmented.as_ref());
        let r = posterior.thin_R();
        let rank_tolerance = f64::EPSILON * (n + p) as f64;
        if (0..p).any(|j| !r[(j, j)].is_finite() || r[(j, j)].abs() <= rank_tolerance) {
            return Err(GreenersError::InvalidOperation(
                "BayesianLinear: augmented posterior is numerically unresolved".into(),
            ));
        }
        let scaled_mean = posterior.solve_lstsq(rhs);
        let gamma = Array1::from_shape_fn(p, |j| scaled_mean[(j, 0)] / column_scales[j]);
        let delta_gamma = Array1::from_shape_fn(p, |j| scaled_mean[(j, 1)] / column_scales[j]);
        let mut beta_n = gamma.clone();
        beta_n[0] = compensated_sum(gamma[0], (1..p).map(|j| (-centres[j], gamma[j])));
        let mut delta = delta_gamma.clone();
        delta[0] = compensated_sum(
            delta_gamma[0],
            (1..p).map(|j| (-centres[j], delta_gamma[j])),
        );
        for j in 0..p {
            // Equivalent posterior representations have different rounding
            // errors. Adding a small correction preserves a concentrated prior;
            // the direct solve avoids cancellation for a distant diffuse prior.
            if non_cancelling_addition(beta_0[j], delta[j]) {
                beta_n[j] = beta_0[j] + delta[j];
            }
        }
        let mut fitted = Array1::zeros(n);
        let mut residuals = Array1::zeros(n);
        for i in 0..n {
            let prior_prediction =
                compensated_sum(0.0, (0..p).map(|j| (x_full[(i, j)], beta_0[j])));
            let change = compensated_sum(
                0.0,
                (0..p).map(|j| (centred_design[(i, j)], delta_gamma[j])),
            );
            if non_cancelling_addition(prior_prediction, change) {
                fitted[i] = compensated_sum(
                    0.0,
                    (0..p)
                        .map(|j| (x_full[(i, j)], beta_0[j]))
                        .chain((0..p).map(|j| (centred_design[(i, j)], delta_gamma[j]))),
                );
                residuals[i] = compensated_sum(
                    y[i],
                    (0..p)
                        .map(|j| (-x_full[(i, j)], beta_0[j]))
                        .chain((0..p).map(|j| (-centred_design[(i, j)], delta_gamma[j]))),
                );
            } else {
                fitted[i] =
                    compensated_sum(0.0, (0..p).map(|j| (centred_design[(i, j)], gamma[j])));
                residuals[i] =
                    compensated_sum(y[i], (0..p).map(|j| (-centred_design[(i, j)], gamma[j])));
            }
        }
        let prior_residual = prior_design.dot(&delta_gamma);
        let prior_norm = prior_residual
            .iter()
            .fold(0.0_f64, |norm, &v| norm.hypot(v));
        let prior_quadratic = prior_norm * prior_norm;
        let sse = residuals.dot(&residuals);
        let a_n = a_0 + n as f64 / 2.0;
        let b_n = b_0 + 0.5 * (sse + prior_quadratic);
        if !a_n.is_finite()
            || a_n <= 1.0
            || !b_n.is_finite()
            || b_n <= 0.0
            || !prior_quadratic.is_finite()
            || prior_quadratic < 0.0
        {
            return Err(GreenersError::InvalidOperation(
                "BayesianLinear: posterior scale or requested posterior moments are invalid".into(),
            ));
        }
        let sigma2_post = b_n / (a_n - 1.0);
        // Cov(beta | sigma2) = sigma2 T D^-1 R^-1 R^-T D^-1 T'.
        let mut inverse_r = Mat::identity(p, p);
        solve_upper_triangular_in_place(r, inverse_r.as_mut(), Par::Seq);
        let mut covariance_factor =
            Array2::from_shape_fn((p, p), |(i, j)| inverse_r[(i, j)] / column_scales[i]);
        for j in 0..p {
            covariance_factor[(0, j)] -= centres.dot(&covariance_factor.column(j));
        }
        let v_n = covariance_factor.dot(&covariance_factor.t());
        let beta_cov = &v_n * sigma2_post;
        let t_dist = StudentsT::new(0.0, 1.0, 2.0 * a_n)
            .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let t_975 = t_dist.inverse_cdf(0.975);
        let mut beta_ci = Array2::zeros((p, 2));
        let mut p_positive = Array1::zeros(p);
        for j in 0..p {
            // This is the marginal Student-t scale, not its standard deviation.
            let scale = v_n[(j, j)].sqrt() * (b_n / a_n).sqrt();
            if !scale.is_finite() || scale <= 0.0 || beta_cov[(j, j)] <= 0.0 {
                return Err(GreenersError::InvalidOperation(
                    "BayesianLinear: posterior coefficient scale or marginal variance is not representable".into(),
                ));
            }
            beta_ci[(j, 0)] = beta_n[j] - t_975 * scale;
            beta_ci[(j, 1)] = beta_n[j] + t_975 * scale;
            p_positive[j] = t_dist.cdf(beta_n[j] / scale);
        }
        let log_det_prior: f64 = (0..p).map(|j| 2.0 * prior.L()[(j, j)].ln()).sum();
        let log_det_precision: f64 = (0..p)
            .map(|j| 2.0 * (r[(j, j)].abs().ln() + column_scales[j].ln()))
            .sum();
        let log_marginal = ln_gamma(a_n)
            - ln_gamma(a_0)
            - (n as f64 / 2.0) * (2.0 * std::f64::consts::PI).ln()
            - 0.5 * (log_det_precision + log_det_prior)
            + a_0 * b_0.ln()
            - a_n * b_n.ln();
        let y_mean = y.mean().unwrap_or(0.0);
        let tss = y.mapv(|v| (v - y_mean).powi(2)).sum();
        let r_squared = if tss > 0.0 { 1.0 - sse / tss } else { 0.0 };
        if !sigma2_post.is_finite()
            || !log_marginal.is_finite()
            || !r_squared.is_finite()
            || !tss.is_finite()
            || beta_n
                .iter()
                .chain(beta_cov.iter())
                .chain(beta_ci.iter())
                .chain(p_positive.iter())
                .chain(fitted.iter())
                .any(|v| !v.is_finite())
        {
            return Err(GreenersError::InvalidOperation(
                "BayesianLinear: posterior calculation exceeded numerical precision".into(),
            ));
        }

        Ok(BayesianLinearResult {
            beta: beta_n,
            beta_cov,
            sigma2: sigma2_post,
            sigma2_shape: a_n,
            sigma2_scale: b_n,
            beta_prior: beta_0,
            v_prior: v_0,
            a_prior: a_0,
            b_prior: b_0,
            beta_ci,
            p_positive,
            n_obs: n,
            n_pred: p,
            log_marginal,
            r_squared,
            fitted,
            variable_names: names,
        })
    }
}

// The condition number of these additions is at most three. Opposite terms
// of similar size instead use the direct-response solution. This is a choice
// of numerical representation, without changing or snapping the posterior.
fn non_cancelling_addition(base: f64, change: f64) -> bool {
    base.signum() == change.signum()
        || change.abs() <= 0.5 * base.abs()
        || base.abs() <= 0.5 * change.abs()
}

// Compensate both rounded products (using FMA) and their sum. In particular,
// compute y - X beta as one accumulation: subtracting an already rounded
// fitted value can discard a representable residual such as 1e16-(1e16+1).
fn compensated_sum(initial: f64, terms: impl Iterator<Item = (f64, f64)>) -> f64 {
    let mut sum = initial;
    let mut correction = 0.0;
    for (a, b) in terms {
        let product = a * b;
        let next = sum + product;
        correction += if sum.abs() >= product.abs() {
            (sum - next) + product
        } else {
            (product - next) + sum
        };
        correction += a.mul_add(b, -product);
        sum = next;
    }
    sum + correction
}

/// Validate both triangles before using the Cholesky factor's lower triangle.
fn factor_spd(matrix: &Array2<f64>, label: &str) -> Result<Llt<f64>, GreenersError> {
    let n = matrix.nrows();
    if n != matrix.ncols() || matrix.iter().any(|v| !v.is_finite()) {
        return Err(GreenersError::InvalidOperation(format!(
            "BayesianLinear: {label} must be finite and square"
        )));
    }
    if (0..n).any(|i| matrix[(i, i)] <= 0.0) {
        return Err(GreenersError::InvalidOperation(format!(
            "BayesianLinear: {label} must be symmetric positive definite"
        )));
    }
    for i in 0..n {
        for j in 0..i {
            // Local covariance units prevent a large, unrelated diagonal from
            // hiding material asymmetry in another coefficient block.
            let scale = (matrix[(i, i)].sqrt() * matrix[(j, j)].sqrt())
                .max(matrix[(i, j)].abs())
                .max(matrix[(j, i)].abs());
            let tolerance = 32.0 * f64::EPSILON * n as f64 * scale;
            if (matrix[(i, j)] - matrix[(j, i)]).abs() > tolerance {
                return Err(GreenersError::InvalidOperation(format!(
                    "BayesianLinear: {label} must be symmetric positive definite"
                )));
            }
        }
    }
    let symmetric = Mat::from_fn(n, n, |i, j| matrix[(i, j)] * 0.5 + matrix[(j, i)] * 0.5);
    Llt::new(symmetric.as_ref(), Side::Lower).map_err(|_| {
        GreenersError::InvalidOperation(format!(
            "BayesianLinear: {label} must be symmetric positive definite"
        ))
    })
}
