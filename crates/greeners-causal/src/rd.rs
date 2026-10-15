use faer::{linalg::solvers::Svd, Mat};
use greeners_core::error::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

// ── Kernel ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RdKernel {
    #[default]
    Triangular,
    Uniform,
    Epanechnikov,
}

impl RdKernel {
    fn weight(self, u: f64) -> f64 {
        match self {
            Self::Triangular => (1.0 - u.abs()).max(0.0),
            Self::Uniform => {
                if u.abs() <= 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Epanechnikov => (0.75 * (1.0 - u * u)).max(0.0),
        }
    }
}

impl fmt::Display for RdKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Triangular => write!(f, "Triangular"),
            Self::Uniform => write!(f, "Uniforme"),
            Self::Epanechnikov => write!(f, "Epanechnikov"),
        }
    }
}

// ── RdResult ─────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct RdResult {
    pub tau: f64,
    pub se: f64,
    pub z: f64,
    pub p_value: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub bandwidth: f64,
    pub n_left: usize,
    pub n_right: usize,
    pub n_total: usize,
    pub poly_order: usize,
    pub cutoff: f64,
    pub kernel: RdKernel,
    pub is_fuzzy: bool,
    /// For fuzzy RD: jump in the probability of treatment (first step)
    pub first_stage_tau: Option<f64>,
    pub first_stage_se: Option<f64>,
    pub outcome_name: Option<String>,
    pub running_name: Option<String>,
    pub treatment_name: Option<String>,
}

impl fmt::Display for RdResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let thick = "═".repeat(70);
        let thin = "─".repeat(70);
        let kind = if self.is_fuzzy { "Fuzzy" } else { "Sharp" };
        let p_name = match self.poly_order {
            0 => "Local Constante",
            1 => "Local Linear",
            2 => "Local Quadratic",
            3 => "Local Cubic",
            p => return write!(f, "[poly order {p}]"),
        };
        writeln!(f, "\n{thick}")?;
        writeln!(
            f,
            " Regression Discontinuity  —  {}  —  {} (p={})",
            kind, p_name, self.poly_order
        )?;
        writeln!(f, "{thick}")?;
        let y_label = self.outcome_name.as_deref().unwrap_or("y");
        let x_label = self.running_name.as_deref().unwrap_or("x");
        writeln!(f, " Outcome: {:<18}  Running var: {}", y_label, x_label)?;
        writeln!(
            f,
            " Cutoff: {:.4}   Bandwidth: {:.4}   Kernel: {}",
            self.cutoff, self.bandwidth, self.kernel
        )?;
        writeln!(
            f,
            "Total obs {} N left {} N right {}",
            self.n_total, self.n_left, self.n_right
        )?;
        writeln!(f, "{thin}")?;

        let sig = |p: f64| {
            if p < 0.01 {
                "***"
            } else if p < 0.05 {
                "**"
            } else if p < 0.10 {
                "*"
            } else {
                ""
            }
        };

        if self.is_fuzzy {
            if let (Some(fs_tau), Some(fs_se)) = (self.first_stage_tau, self.first_stage_se) {
                let trt_label = self.treatment_name.as_deref().unwrap_or("D");
                let fs_z = fs_tau / fs_se;
                let fs_p = 2.0 * (1.0 - Normal::standard().cdf(fs_z.abs()));
                writeln!(f, "First Step ({}):", trt_label)?;
                writeln!(
                    f,
                    "   Salto D̂    {:>10.4}   SE {:>10.4}   z {:>8.3}   p {:>8.4}  {}",
                    fs_tau,
                    fs_se,
                    fs_z,
                    fs_p,
                    sig(fs_p)
                )?;
                writeln!(f, "{thin}")?;
            }
        }

        writeln!(f, " Treatment effect (τ̂):")?;
        let z_str = if self.z.abs() > 1e10 {
            format!("{:.3e}", self.z)
        } else {
            format!("{:.3}", self.z)
        };
        writeln!(
            f,
            "   {:>10.4}   SE {:>10.4}   z {:>8}   P>|z| {:>8.4}  {}",
            self.tau,
            self.se,
            z_str,
            self.p_value,
            sig(self.p_value)
        )?;
        writeln!(f, " IC 95%: [{:.4}, {:.4}]", self.ci_lower, self.ci_upper)?;
        writeln!(f, "{thick}")?;
        writeln!(f, " *** p<0.01  ** p<0.05  * p<0.10")
    }
}

// ── RD estimador ─────────────────────────────────────────────────────────────

pub struct RD;

impl RD {
    /// Sharp RD by weighted polynomial local regression.
    ///
    /// * `y` — dependent variable
    /// * `x` — continuous variable assignment
    /// * `cutoff` — treatment threshold
    /// * `bandwidth` — `None` dispara seletor IK (Imbens-Kalyanaraman 2012)
    /// * `poly_order`— ordem do polinômio local (1 = linear, 2 = quadrático)
    /// * `kernel` — kernel function (standard: Triangular)
    pub fn fit(
        y: &Array1<f64>,
        x: &Array1<f64>,
        cutoff: f64,
        bandwidth: Option<f64>,
        poly_order: usize,
        kernel: RdKernel,
        variable_names: Option<(String, String)>,
    ) -> Result<RdResult, GreenersError> {
        let n = y.len();
        if x.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "rd: y and x have different sizes".into(),
            ));
        }
        if y.iter().chain(x.iter()).any(|v| !v.is_finite()) {
            return Err(GreenersError::InvalidOperation(
                "rd: data contain NaN or Inf".into(),
            ));
        }

        validate_configuration(n, cutoff, bandwidth, poly_order)?;
        let h = bandwidth.unwrap_or_else(|| Self::ik_bandwidth(y, x, cutoff, poly_order));
        validate_configuration(n, cutoff, Some(h), poly_order)?;

        let left = Self::side_fit(y, x, cutoff, h, poly_order, kernel, Side::Left)?;
        let right = Self::side_fit(y, x, cutoff, h, poly_order, kernel, Side::Right)?;
        let (n_left, n_right) = (left.n, right.n);
        let tau = right.beta[0] - left.beta[0];
        let se = (left.covariance[(0, 0)] + right.covariance[(0, 0)]).sqrt();
        let (z, p_value) = scalar_inference(tau, se)?;
        let z95 = 1.959_963_985;
        let ci_lower = tau - z95 * se;
        let ci_upper = tau + z95 * se;
        if !ci_lower.is_finite() || !ci_upper.is_finite() {
            return Err(GreenersError::InvalidOperation(
                "rd: confidence interval exceeds numerical precision".into(),
            ));
        }
        let (outcome_name, running_name) = variable_names
            .map(|(a, b)| (Some(a), Some(b)))
            .unwrap_or((None, None));

        Ok(RdResult {
            tau,
            se,
            z,
            p_value,
            ci_lower,
            ci_upper,
            bandwidth: h,
            n_left,
            n_right,
            n_total: n_left + n_right,
            poly_order,
            cutoff,
            kernel,
            is_fuzzy: false,
            first_stage_tau: None,
            first_stage_se: None,
            outcome_name,
            running_name,
            treatment_name: None,
        })
    }

    /// Fuzzy RD local Wald ratio, with the joint HC1 delta-method covariance.
    ///
    /// Treatment may be binary or continuous. A scale-aware numerical guard
    /// rejects an unresolved first-stage jump; it does not diagnose instrument
    /// strength. Conventional Wald intervals are not weak-identification-robust
    /// and do not correct smoothing bias. With zero SE, a zero effect uses the
    /// degenerate convention z=0, p=1; a nonzero effect has signed-infinite z,
    /// p=0 and a collapsed interval.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_fuzzy(
        y: &Array1<f64>,
        d: &Array1<f64>,
        x: &Array1<f64>,
        cutoff: f64,
        bandwidth: Option<f64>,
        poly_order: usize,
        kernel: RdKernel,
        variable_names: Option<(String, String, String)>,
    ) -> Result<RdResult, GreenersError> {
        let n = y.len();
        if d.len() != n || x.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "fuzzy_rd: y, d, x must be the same size".into(),
            ));
        }
        if y.iter()
            .chain(d.iter())
            .chain(x.iter())
            .any(|v| !v.is_finite())
        {
            return Err(GreenersError::InvalidOperation(
                "fuzzy_rd: data contain NaN or Inf".into(),
            ));
        }

        validate_configuration(n, cutoff, bandwidth, poly_order)?;
        let h = bandwidth.unwrap_or_else(|| Self::ik_bandwidth(y, x, cutoff, poly_order));
        validate_configuration(n, cutoff, Some(h), poly_order)?;

        let yl = Self::side_fit(y, x, cutoff, h, poly_order, kernel, Side::Left)?;
        let yr = Self::side_fit(y, x, cutoff, h, poly_order, kernel, Side::Right)?;
        let dl = Self::side_fit(d, x, cutoff, h, poly_order, kernel, Side::Left)?;
        let dr = Self::side_fit(d, x, cutoff, h, poly_order, kernel, Side::Right)?;
        let (n_left, n_right) = (yl.n, yr.n);
        let tau_y = yr.beta[0] - yl.beta[0];
        let tau_d = dr.beta[0] - dl.beta[0];
        let stage_resolution = dl.intercept_error + dr.intercept_error;
        if !tau_d.is_finite() || tau_d.abs() <= stage_resolution {
            return Err(GreenersError::InvalidOperation(
                "fuzzy_rd: first-stage jump is unresolved at the precision of the local design"
                    .into(),
            ));
        }
        let tau = tau_y / tau_d;
        if !tau.is_finite() {
            return Err(GreenersError::InvalidOperation(
                "fuzzy_rd: effect ratio exceeds numerical precision".into(),
            ));
        }
        // Linearity of WLS gives residual u_y - tau*u_d. Its sandwich is
        // V_y + tau² V_d - 2*tau*Cov(y,d), including the shared-sample term,
        // without subtracting nearly equal variances when Y is proportional to D.
        let adjusted_y = y - &d.mapv(|v| tau * v);
        let al = Self::side_fit(&adjusted_y, x, cutoff, h, poly_order, kernel, Side::Left)?;
        let ar = Self::side_fit(&adjusted_y, x, cutoff, h, poly_order, kernel, Side::Right)?;
        let se = (al.covariance[(0, 0)] + ar.covariance[(0, 0)]).sqrt() / tau_d.abs();
        let se_fs = (dl.covariance[(0, 0)] + dr.covariance[(0, 0)]).sqrt();
        let (z, p_value) = scalar_inference(tau, se)?;
        let z95 = 1.959_963_985;
        let ci_lower = tau - z95 * se;
        let ci_upper = tau + z95 * se;
        if !ci_lower.is_finite() || !ci_upper.is_finite() {
            return Err(GreenersError::InvalidOperation(
                "rd: confidence interval exceeds numerical precision".into(),
            ));
        }

        let (outcome_name, running_name, treatment_name) = variable_names
            .map(|(a, b, c)| (Some(a), Some(b), Some(c)))
            .unwrap_or((None, None, None));

        Ok(RdResult {
            tau,
            se,
            z,
            p_value,
            ci_lower,
            ci_upper,
            bandwidth: h,
            n_left,
            n_right,
            n_total: n_left + n_right,
            poly_order,
            cutoff,
            kernel,
            is_fuzzy: true,
            first_stage_tau: Some(tau_d),
            first_stage_se: Some(se_fs),
            outcome_name,
            running_name,
            treatment_name,
        })
    }

    // ── Internos ─────────────────────────────────────────────────────────────

    /// Local polynomial adjustment on one side of the cutoff (WLS + HC1).
    fn side_fit(
        y: &Array1<f64>,
        x: &Array1<f64>,
        cutoff: f64,
        h: f64,
        poly_order: usize,
        kernel: RdKernel,
        side: Side,
    ) -> Result<LocalPolynomialFit, GreenersError> {
        let mut ys = Vec::new();
        let mut xs = Vec::new();
        let mut ws = Vec::new();

        for i in 0..y.len() {
            let in_side = match side {
                Side::Left => x[i] < cutoff,
                Side::Right => x[i] >= cutoff,
            };
            if !in_side {
                continue;
            }
            let u = (x[i] - cutoff) / h;
            let w = kernel.weight(u);
            if w <= 0.0 {
                continue;
            }
            ys.push(y[i]);
            xs.push(x[i] - cutoff);
            ws.push(w);
        }

        let n = ys.len();
        let p = poly_order + 1;

        if n <= p {
            return Err(GreenersError::ShapeMismatch(format!(
                "rd: insufficient observations ({n}) for polynomial of order {poly_order} (side {})",
                match side { Side::Left => "left", Side::Right => "right" }
            )));
        }

        local_poly_wls(&ys, &xs, &ws, poly_order)
    }

    /// Automatic bandwidth selector — Imbens-Kalyanaraman (2012), revision ReStud.
    ///
    /// For linear location (p=1) with triangular kernel:
    ///   h* = [C_K * (σ²₊ + σ²₋) / (n * f(c) * B²)]^(1/5)
    /// where B = jump in second order derivative.
    pub fn ik_bandwidth(y: &Array1<f64>, x: &Array1<f64>, cutoff: f64, poly_order: usize) -> f64 {
        let n = y.len() as f64;
        if n < 10.0 {
            return 1.0;
        }

        let x_mean = x.mean().unwrap_or(0.0);
        let x_sd = ((x.iter().map(|&v| (v - x_mean).powi(2)).sum::<f64>())
            / (x.len().saturating_sub(1)) as f64)
            .sqrt();
        if x_sd < 1e-15 {
            return 1.0;
        }

        let h0 = 1.84 * x_sd * n.powf(-0.2);

        // Ajuste local de ordem (poly_order+1) em cada lado com h0
        // → coeficiente na potência (poly_order+1) estima m^(p+1)(c)/(p+1)!
        let q = poly_order + 1; // ordem piloto

        let side_fit_pilot = |side: Side| -> Option<(f64, f64)> {
            let mut ys = Vec::new();
            let mut xs = Vec::new();
            for i in 0..y.len() {
                let in_side = match side {
                    Side::Left => x[i] < cutoff,
                    Side::Right => x[i] >= cutoff,
                };
                if !in_side {
                    continue;
                }
                let u = (x[i] - cutoff) / h0;
                if u.abs() > 1.0 {
                    continue;
                }
                ys.push(y[i]);
                xs.push(x[i] - cutoff);
            }
            if ys.len() < q + 2 {
                return None;
            }
            // Uniform weights for pilot
            let ws = vec![1.0_f64; ys.len()];
            let fit = local_poly_wls(&ys, &xs, &ws, q).ok()?;
            let beta = fit.beta;
            let deriv_coeff = beta.get(q).copied()?; // coef em x^q
            let n_s = ys.len() as f64;
            let p_s = (q + 1) as f64;
            let resid_var: f64 = ys
                .iter()
                .zip(xs.iter())
                .map(|(&yi, &xi)| {
                    let y_hat: f64 = (0..=q).map(|j| beta[j] * xi.powi(j as i32)).sum();
                    (yi - y_hat).powi(2)
                })
                .sum::<f64>()
                / (n_s - p_s).max(1.0);
            Some((deriv_coeff, resid_var))
        };

        let (m_left, sigma2_left) = side_fit_pilot(Side::Left).unwrap_or((0.0, 1.0));
        let (m_right, sigma2_right) = side_fit_pilot(Side::Right).unwrap_or((0.0, 1.0));

        //Jump on derivative (p+1)-th (divided by (p+1)!)
        let b_jump = m_right - m_left;
        if b_jump.abs() < 1e-12 {
            return h0; //without detectable curvature → fallback
        }

        //C density: count in pilot window / (2 *h0 * n)
        let n_window = x.iter().filter(|&&xi| (xi - cutoff).abs() <= h0).count() as f64;
        let f_c = (n_window / (2.0 * h0 * n)).max(1e-10);

        // Constante triangular / local linear IK 2012 → C_K ≈ 3.4375
        let c_k = 3.4375_f64;
        let exponent = 1.0 / (2.0 * poly_order as f64 + 3.0);

        let h_star =
            (c_k * (sigma2_left + sigma2_right) / (n * f_c * b_jump * b_jump)).powf(exponent);

        //Keep within [0.05 * sd, 2 * sd]
        h_star.max(0.05 * x_sd).min(2.0 * x_sd)
    }
}

// ── Helpers internos ──────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum Side {
    Left,
    Right,
}

/// The same side-specific fit is used for outcomes, treatment and their contrast.
struct LocalPolynomialFit {
    beta: Array1<f64>,
    covariance: Array2<f64>,
    n: usize,
    intercept_error: f64,
}

fn validate_configuration(
    n: usize,
    cutoff: f64,
    bandwidth: Option<f64>,
    order: usize,
) -> Result<(), GreenersError> {
    if !cutoff.is_finite() || bandwidth.is_some_and(|h| !h.is_finite() || h <= 0.0) {
        return Err(GreenersError::InvalidOperation(
            "rd: cutoff must be finite and bandwidth positive and finite".into(),
        ));
    }
    if order >= n || i32::try_from(order).is_err() {
        return Err(GreenersError::InvalidOperation(
            "rd: polynomial order must be supported by the sample size".into(),
        ));
    }
    Ok(())
}

fn scalar_inference(effect: f64, se: f64) -> Result<(f64, f64), GreenersError> {
    if !effect.is_finite() || !se.is_finite() || se < 0.0 {
        return Err(GreenersError::InvalidOperation(
            "rd: effect or uncertainty exceeds numerical precision".into(),
        ));
    }
    let z = if se > 0.0 {
        effect / se
    } else if effect == 0.0 {
        0.0
    } else {
        f64::INFINITY.copysign(effect)
    };
    Ok((z, 2.0 * Normal::standard().sf(z.abs())))
}

/// Local polynomial WLS with HC1. An SVD of the scaled weighted design avoids
/// squaring its condition number and rejects unidentified local polynomials.
fn local_poly_wls(
    y: &[f64],
    x_centered: &[f64],
    weights: &[f64],
    poly_order: usize,
) -> Result<LocalPolynomialFit, GreenersError> {
    let n = y.len();
    let p = poly_order
        .checked_add(1)
        .ok_or_else(|| GreenersError::InvalidOperation("rd: invalid polynomial order".into()))?;
    if n <= p {
        return Err(GreenersError::InvalidOperation(
            "rd: HC1 requires more positive-weight observations than local coefficients".into(),
        ));
    }
    let mut design = Array2::zeros((n, p));
    for i in 0..n {
        let mut power = 1.0;
        for j in 0..p {
            design[(i, j)] = power * weights[i].sqrt();
            power *= x_centered[i];
        }
    }
    if design.iter().chain(y.iter()).any(|v| !v.is_finite()) {
        return Err(GreenersError::InvalidOperation(
            "rd: local design or response is non-finite".into(),
        ));
    }
    let scales: Vec<f64> = design
        .columns()
        .into_iter()
        .map(|col| col.iter().fold(0.0_f64, |norm, &v| norm.hypot(v)))
        .collect();
    if scales.iter().any(|&v| !v.is_finite() || v <= 0.0) {
        return Err(GreenersError::InvalidOperation(
            "rd: local design is rank deficient".into(),
        ));
    }
    let matrix = Mat::from_fn(n, p, |i, j| design[(i, j)] / scales[j]);
    let svd = Svd::new_thin(matrix.as_ref()).map_err(|_| GreenersError::OptimizationFailed)?;
    let singular = svd.S().column_vector();
    let largest = (0..p).map(|j| singular[j]).fold(0.0, f64::max);
    let smallest = (0..p).map(|j| singular[j]).fold(f64::INFINITY, f64::min);
    if !largest.is_finite()
        || !smallest.is_finite()
        || smallest <= f64::EPSILON * n as f64 * largest
    {
        return Err(GreenersError::InvalidOperation(
            "rd: local design is rank deficient".into(),
        ));
    }
    // Pseudoinverse entries below are used only after verifying full column rank.
    let inverse_design = Array2::from_shape_fn((p, n), |(j, i)| {
        (0..p)
            .map(|k| svd.V()[(j, k)] * svd.U()[(i, k)] / singular[k])
            .sum::<f64>()
            / scales[j]
    });
    let weighted_y = Array1::from_shape_fn(n, |i| y[i] * weights[i].sqrt());
    let beta = inverse_design.dot(&weighted_y);
    let residuals: Vec<f64> = (0..n)
        .map(|i| y[i] - design.row(i).dot(&beta) / weights[i].sqrt())
        .collect();
    // Each score contribution is sqrt(w_i)*e_i times a column of the weighted
    // design pseudoinverse, yielding the ordinary w_i² HC1 meat.
    let correction = n as f64 / (n - p) as f64;
    let scores = Array2::from_shape_fn((p, n), |(j, i)| {
        inverse_design[(j, i)] * weights[i].sqrt() * residuals[i]
    });
    let covariance = scores.dot(&scores.t()) * correction;
    if (0..p).any(|j| covariance[(j, j)] == 0.0 && scores.row(j).iter().any(|&v| v != 0.0)) {
        return Err(GreenersError::InvalidOperation(
            "rd: local covariance underflows numerical precision".into(),
        ));
    }
    let absolute_intercept_sum: f64 = (0..n)
        .map(|i| (inverse_design[(0, i)] * weighted_y[i]).abs())
        .sum();
    // Backward-error budget for the SVD solve and intercept dot product. It
    // scales with treatment units and the actual scaled-design condition number.
    let intercept_error =
        8.0 * f64::EPSILON * n as f64 * p as f64 * (largest / smallest) * absolute_intercept_sum;
    if beta.iter().chain(covariance.iter()).any(|v| !v.is_finite()) || !intercept_error.is_finite()
    {
        return Err(GreenersError::InvalidOperation(
            "rd: local fit exceeds numerical precision".into(),
        ));
    }
    Ok(LocalPolynomialFit {
        beta,
        covariance,
        n,
        intercept_error,
    })
}
