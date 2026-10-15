//! Phillips-Hansen FMOLS for one cointegrating time-series regression.
//!
//! Uses all observations for initial OLS and rows 2..N for the fully modified
//! regression, Bartlett long-run covariance and the one-sided bias correction.
//! Cointegration and weak-dependence assumptions are not tested by this estimator.

use greeners_core::linalg::{LinalgQR as _, LinalgSVD as _};
use greeners_core::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Result of FMOLS estimation.
#[derive(Debug)]
pub struct FmolsResult {
    /// Intercept
    pub alpha: f64,
    /// Slope coefficients
    pub beta: Array1<f64>,
    /// SE of alpha
    pub alpha_se: f64,
    /// SE of beta
    pub beta_se: Array1<f64>,
    /// Normal test statistics of beta (legacy field name)
    pub beta_t: Array1<f64>,
    /// Two-sided asymptotic Normal p-values of beta
    pub beta_p: Array1<f64>,
    /// Bartlett long-run covariance of [initial all-N OLS residual, diff(x)]'
    /// on rows 2..N, in retained slope order and original response/predictor units.
    pub omega: Array2<f64>,
    /// Initial all-observation OLS R-squared
    pub r_squared: f64,
    /// Fully modified regression observations (input length minus one)
    pub n_obs: usize,
    /// Number of regressors
    pub n_regressors: usize,
    /// Bandwidth used for kernel
    pub bandwidth: usize,
    /// Variable names
    pub variable_names: Vec<String>,
}

impl fmt::Display for FmolsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Fully Modified OLS (FMOLS) ")?;
        writeln!(f, "Phillips & Hansen (1990) nonparametric correction")?;
        writeln!(
            f,
            "Asymptotic Normal inference; finite-sample coverage is not guaranteed"
        )?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Regressors:", self.n_regressors)?;
        writeln!(f, "{:<20} {:>12}", "Bandwidth:", self.bandwidth)?;
        writeln!(f, "{:<20} {:>12.6}", "R-squared:", self.r_squared)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coef.", "Std.Err.", "z", "P>|z|"
        )?;
        writeln!(f, "{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
            "alpha",
            self.alpha,
            self.alpha_se,
            self.alpha / self.alpha_se,
            2.0 * Normal::standard().sf((self.alpha / self.alpha_se).abs())
        )?;
        for i in 0..self.beta.len() {
            let name = self
                .variable_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("x{}", i));
            writeln!(
                f,
                "{:<12} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
                name, self.beta[i], self.beta_se[i], self.beta_t[i], self.beta_p[i]
            )?;
        }
        write!(f, "{:=^78}", "")
    }
}

pub struct FMOLS;

impl FMOLS {
    /// Estimate an intercept and one or more I(1) slopes, assuming cointegration.
    ///
    /// An exact unit column anywhere in `x` supplies the intercept; other constant
    /// columns, deficient ranks and unresolved inference return errors. Names must
    /// describe all supplied columns, or just slopes when a unit column exists.
    /// Inputs must be finite and N >= k+5 for k retained slopes. Initial OLS uses N;
    /// FMOLS uses N-1, Bartlett bandwidth max(1,floor(4*(N/100)^(2/9))), no covariance
    /// demeaning/df adjustment, conditional long-run variance and Normal inference.
    /// This intentionally changes the old ridge-based coefficients and uncertainty.
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<FmolsResult, GreenersError> {
        let total = y.len();
        if x.nrows() != total {
            return Err(GreenersError::ShapeMismatch(
                "FMOLS: y and x must have same length".into(),
            ));
        }
        if total == 0
            || total as u128 > (1_u128 << 53)
            || x.iter().chain(y.iter()).any(|v| !v.is_finite())
        {
            return Err(invalid("empty, non-finite or unrepresentable input"));
        }
        let constants: Vec<_> = (0..x.ncols())
            .filter(|&j| x.column(j).iter().all(|&v| v == 1.0))
            .collect();
        if constants.len() > 1 {
            return Err(invalid("multiple unit intercept columns"));
        }
        let intercept = constants.first().copied();
        let columns: Vec<_> = (0..x.ncols()).filter(|j| Some(*j) != intercept).collect();
        let k = columns.len();
        if k == 0 || total.saturating_sub(k) < 5 {
            return Err(invalid("at least one slope and N >= k+5 required"));
        }
        let names = match variable_names {
            None => (0..k).map(|j| format!("x{j}")).collect(),
            Some(names) if names.len() == x.ncols() => {
                columns.iter().map(|&j| names[j].clone()).collect()
            }
            Some(names) if intercept.is_some() && names.len() == k => names,
            Some(_) => {
                return Err(invalid(
                    "names do not match supplied columns or retained slopes",
                ))
            }
        };
        if names.iter().any(|s| s.trim().is_empty())
            || names
                .iter()
                .enumerate()
                .any(|(j, s)| names[..j].contains(s))
        {
            return Err(invalid("slope names must be nonempty and unique"));
        }
        let mut z = Array2::ones((total, k + 1));
        let offsets: Vec<_> = columns.iter().map(|&j| x[[0, j]]).collect();
        let mut scales = Vec::with_capacity(k);
        for (j, &column) in columns.iter().enumerate() {
            let centered = x.column(column).mapv(|v| v - offsets[j]);
            let scale = norm(centered.iter().copied());
            if !scale.is_finite() || scale <= 0.0 {
                return Err(invalid("constant or unrepresentable predictor"));
            }
            z.column_mut(j + 1).assign(&(&centered / scale));
            scales.push(scale);
        }
        let y_offset = y[0];
        let centered_y = y.mapv(|v| v - y_offset);
        let y_scale = norm(centered_y.iter().copied());
        if !y_scale.is_finite() || y_scale <= 0.0 {
            return Err(invalid("constant or unrepresentable response"));
        }
        let scaled_y = &centered_y / y_scale;
        let (q, r) = checked_qr(&z)?;
        let ols = back_solve(&r, &q.t().dot(&scaled_y));
        let fitted = z.dot(&ols);
        let residuals = &scaled_y - &fitted;
        resolved_residual(&residuals, &scaled_y, &fitted, total)?;
        let n = total - 1;
        let bandwidth = ((4.0 * (total as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize).max(1);
        if bandwidth >= n {
            return Err(invalid("bandwidth exceeds aligned sample"));
        }
        let eta = Array2::from_shape_fn((n, k + 1), |(i, j)| {
            if j == 0 {
                residuals[i + 1]
            } else {
                z[[i + 1, j]] - z[[i, j]]
            }
        });
        let (scores, lambda) = bartlett(&eta, bandwidth);
        let score_x = scores.slice(ndarray::s![.., 1..]).to_owned();
        let score_u = scores.column(0).to_owned();
        let (score_q, score_r) = checked_qr(&score_x)?;
        let correction = back_solve(&score_r, &score_q.t().dot(&score_u));
        let fitted_score = score_x.dot(&correction);
        let conditional_residual = &score_u - &fitted_score;
        resolved_residual(
            &conditional_residual,
            &score_u,
            &fitted_score,
            scores.nrows(),
        )?;
        let conditional_sd = norm(conditional_residual.iter().copied());
        let bias = Array1::from_shape_fn(k + 1, |j| {
            if j == 0 {
                0.0
            } else {
                lambda[[0, j]]
                    - (0..k)
                        .map(|a| correction[a] * lambda[[a + 1, j]])
                        .sum::<f64>()
            }
        });
        let final_z = z.slice(ndarray::s![1.., ..]).to_owned();
        let y_plus = scaled_y.slice(ndarray::s![1..]).to_owned()
            - eta.slice(ndarray::s![.., 1..]).dot(&correction);
        let (final_q, final_r) = checked_qr(&final_z)?;
        let mut inverse_r = Array2::zeros((k + 1, k + 1));
        for j in 0..k + 1 {
            let mut e = Array1::zeros(k + 1);
            e[j] = 1.0;
            inverse_r.column_mut(j).assign(&back_solve(&final_r, &e));
        }
        let fm = back_solve(&final_r, &final_q.t().dot(&y_plus))
            - inverse_r.dot(&inverse_r.t().dot(&(bias * n as f64)));
        // Row norms propagate uncertainty without subtracting nearly equal covariances.
        let mut transform = Array2::zeros((k + 1, k + 1));
        transform[[0, 0]] = y_scale;
        for j in 0..k {
            transform[[j + 1, j + 1]] = y_scale / scales[j];
            transform[[0, j + 1]] = -offsets[j] * (y_scale / scales[j]);
        }
        let mut params = transform.dot(&fm);
        params[0] += y_offset;
        let se_factor = transform.dot(&inverse_r);
        let standard_errors = Array1::from_iter(
            se_factor
                .rows()
                .into_iter()
                .map(|row| conditional_sd * norm(row.iter().copied())),
        );
        if params.iter().any(|v| !v.is_finite())
            || standard_errors.iter().any(|&v| !v.is_finite() || v <= 0.0)
        {
            return Err(invalid(
                "coefficients or positive standard errors are not representable",
            ));
        }
        let statistics = &params / &standard_errors;
        if statistics.iter().any(|v| !v.is_finite()) {
            return Err(invalid("test statistics are not representable"));
        }
        let probabilities = statistics.mapv(|v| 2.0 * Normal::standard().sf(v.abs()));
        let score_scales: Vec<_> = std::iter::once(y_scale)
            .chain(scales.iter().copied())
            .collect();
        let scaled_omega = scores.t().dot(&scores);
        let omega = Array2::from_shape_fn((k + 1, k + 1), |(i, j)| {
            scaled_omega[[i, j]] * score_scales[i] * score_scales[j]
        });
        if omega.iter().any(|v| !v.is_finite()) || omega.diag().iter().any(|&v| v <= 0.0) {
            return Err(invalid("long-run covariance is not representable"));
        }
        let mean = scaled_y.sum() / total as f64;
        let tss_norm = norm(scaled_y.iter().map(|&v| v - mean));
        let residual_norm = norm(residuals.iter().copied());
        let r_squared = 1.0 - (residual_norm / tss_norm).powi(2);
        if !r_squared.is_finite() {
            return Err(invalid("initial OLS R-squared is unresolved"));
        }
        Ok(FmolsResult {
            alpha: params[0],
            beta: params.slice(ndarray::s![1..]).to_owned(),
            alpha_se: standard_errors[0],
            beta_se: standard_errors.slice(ndarray::s![1..]).to_owned(),
            beta_t: statistics.slice(ndarray::s![1..]).to_owned(),
            beta_p: probabilities.slice(ndarray::s![1..]).to_owned(),
            omega,
            r_squared,
            n_obs: n,
            n_regressors: k,
            bandwidth,
            variable_names: names,
        })
    }
}

fn invalid(message: &str) -> GreenersError {
    GreenersError::InvalidOperation(format!("FMOLS: {message}"))
}
fn norm(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0, f64::hypot)
}

fn checked_qr(matrix: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), GreenersError> {
    let scales: Vec<_> = matrix
        .columns()
        .into_iter()
        .map(|col| norm(col.iter().copied()))
        .collect();
    if scales.iter().any(|&v| v <= 0.0 || !v.is_finite()) || matrix.iter().any(|v| !v.is_finite()) {
        return Err(invalid("design or long-run score rank is unresolved"));
    }
    let scaled = Array2::from_shape_fn(matrix.dim(), |(i, j)| matrix[[i, j]] / scales[j]);
    let (q, mut r) = scaled.qr()?;
    let (_, singular, _) = r.svd(false, false)?;
    let largest = singular.iter().copied().fold(0.0, f64::max);
    let threshold = 64.0 * f64::EPSILON * matrix.nrows().max(matrix.ncols()) as f64 * largest;
    if singular.len() != matrix.ncols()
        || singular.iter().any(|&v| !v.is_finite() || v <= threshold)
    {
        return Err(invalid("design or long-run score rank is unresolved"));
    }
    for (j, &scale) in scales.iter().enumerate() {
        r.column_mut(j).mapv_inplace(|v| v * scale);
    }
    Ok((q, r))
}
fn back_solve(r: &Array2<f64>, rhs: &Array1<f64>) -> Array1<f64> {
    let mut solution = rhs.clone();
    for i in (0..rhs.len()).rev() {
        for j in i + 1..rhs.len() {
            solution[i] -= r[[i, j]] * solution[j];
        }
        solution[i] /= r[[i, i]];
    }
    solution
}
fn resolved_residual(
    residual: &Array1<f64>,
    original: &Array1<f64>,
    fitted: &Array1<f64>,
    rows: usize,
) -> Result<(), GreenersError> {
    let error_budget = 64.0
        * f64::EPSILON
        * rows as f64
        * (norm(original.iter().copied()) + norm(fitted.iter().copied()));
    let residual_norm = norm(residual.iter().copied());
    if !residual_norm.is_finite() || residual_norm <= error_budget {
        return Err(invalid(
            "conditional or initial residual variation is numerically unresolved",
        ));
    }
    Ok(())
}
// Zero padding gives n+B scores, whose Gram exactly equals the Bartlett lag sum.
fn bartlett(eta: &Array2<f64>, bandwidth: usize) -> (Array2<f64>, Array2<f64>) {
    let n = eta.nrows();
    let k = eta.ncols();
    let divisor = (n as f64 * (bandwidth + 1) as f64).sqrt();
    let scores = Array2::from_shape_fn((n + bandwidth, k), |(h, j)| {
        (h.saturating_sub(bandwidth)..=h.min(n - 1))
            .map(|t| eta[[t, j]])
            .sum::<f64>()
            / divisor
    });
    let mut lambda = eta.t().dot(eta) / n as f64;
    for lag in 1..=bandwidth {
        let weight = 1.0 - lag as f64 / (bandwidth + 1) as f64;
        // Current rows, past columns: transposing this changes the FM bias.
        let gamma = eta
            .slice(ndarray::s![lag.., ..])
            .t()
            .dot(&eta.slice(ndarray::s![..n - lag, ..]))
            / n as f64;
        lambda += &(gamma * weight);
    }
    (scores, lambda)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bartlett_boundary_factor_and_one_sided_orientation() {
        let eta = ndarray::array![[1., 2.], [-3., 1.], [2., 4.], [5., -2.], [-1., 3.]];
        let (scores, lambda) = bartlett(&eta, 2);
        assert_eq!(scores.nrows(), 7);
        let gamma0 = eta.t().dot(&eta) / 5.0;
        let mut direct = gamma0.clone();
        let mut one_sided = gamma0;
        for lag in 1..=2 {
            let gamma = eta
                .slice(ndarray::s![lag.., ..])
                .t()
                .dot(&eta.slice(ndarray::s![..5 - lag, ..]))
                / 5.0;
            let weight = 1.0 - lag as f64 / 3.0;
            direct += &((&gamma + &gamma.t()) * weight);
            one_sided += &(gamma * weight);
        }
        for (a, b) in scores.t().dot(&scores).iter().zip(direct.iter()) {
            assert!((a - b).abs() < 1e-13);
        }
        assert!((lambda[[0, 1]] - lambda[[1, 0]]).abs() > 1.0);
        assert_eq!(lambda, one_sided);
    }
    #[test]
    fn score_rank_and_conditional_precision_are_explicit_errors() {
        let x = ndarray::array![[1., 1.], [2., 2.], [3., 3.]];
        assert!(checked_qr(&x).is_err());
        let original = ndarray::array![1., 2., 3.];
        let residual = ndarray::array![1e-15, -1e-15, 1e-15];
        assert!(resolved_residual(&residual, &original, &(&original - &residual), 3).is_err());
        let residual = ndarray::array![1e-7, -1e-7, 1e-7];
        assert!(resolved_residual(&residual, &original, &(&original - &residual), 3).is_ok());
    }
}
