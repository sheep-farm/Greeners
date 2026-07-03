use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
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
    /// Para RD fuzzy: salto na probabilidade de tratamento (primeira etapa)
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
            2 => "Local Quadrático",
            3 => "Local Cúbico",
            p => return write!(f, "[poly order {p}]"),
        };
        writeln!(f, "\n{thick}")?;
        writeln!(
            f,
            " Regressão Descontínua  —  {}  —  {} (p={})",
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
            " Obs total: {}   N esquerda: {}   N direita: {}",
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
                let fs_p = 2.0 * (1.0 - Normal::new(0.0, 1.0).unwrap().cdf(fs_z.abs()));
                writeln!(f, " Primeira Etapa ({}):", trt_label)?;
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

        writeln!(f, " Efeito de Tratamento (τ̂):")?;
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
    /// Sharp RD por regressão local polinomial ponderada.
    ///
    /// * `y`         — variável dependente
    /// * `x`         — variável contínua de atribuição (running variable)
    /// * `cutoff`    — limiar de tratamento
    /// * `bandwidth` — `None` dispara seletor IK (Imbens-Kalyanaraman 2012)
    /// * `poly_order`— ordem do polinômio local (1 = linear, 2 = quadrático)
    /// * `kernel`    — função kernel (padrão: Triangular)
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
                "rd: y e x têm tamanhos diferentes".into(),
            ));
        }
        if y.iter().chain(x.iter()).any(|v| !v.is_finite()) {
            return Err(GreenersError::InvalidOperation(
                "rd: dados contêm NaN ou Inf".into(),
            ));
        }

        let h = bandwidth.unwrap_or_else(|| Self::ik_bandwidth(y, x, cutoff, poly_order));

        let (beta_l, vcov_l, n_left) =
            Self::side_fit(y, x, cutoff, h, poly_order, kernel, Side::Left)?;
        let (beta_r, vcov_r, n_right) =
            Self::side_fit(y, x, cutoff, h, poly_order, kernel, Side::Right)?;

        let tau = beta_r[0] - beta_l[0];
        let var_tau = (vcov_l[[0, 0]] + vcov_r[[0, 0]]).max(0.0);
        let se = var_tau.sqrt();
        let z = tau / se;
        let norm = Normal::new(0.0, 1.0).unwrap();
        let p_value = 2.0 * (1.0 - norm.cdf(z.abs()));
        let z95 = 1.959_963_985;
        let (outcome_name, running_name) = variable_names
            .map(|(a, b)| (Some(a), Some(b)))
            .unwrap_or((None, None));

        Ok(RdResult {
            tau,
            se,
            z,
            p_value,
            ci_lower: tau - z95 * se,
            ci_upper: tau + z95 * se,
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

    /// Fuzzy RD — estimador de Wald local (LATE no cutoff).
    ///
    /// * `d` — tratamento real recebido (binário ou contínuo em [0,1])
    ///
    /// τ̂_FRD = salto(Y) / salto(D)  (razão de dois RD sharps)
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
                "fuzzy_rd: y, d, x devem ter o mesmo tamanho".into(),
            ));
        }
        if y.iter()
            .chain(d.iter())
            .chain(x.iter())
            .any(|v| !v.is_finite())
        {
            return Err(GreenersError::InvalidOperation(
                "fuzzy_rd: dados contêm NaN ou Inf".into(),
            ));
        }

        let h = bandwidth.unwrap_or_else(|| Self::ik_bandwidth(y, x, cutoff, poly_order));

        // Reduzida: Y ~ X  (salto em Y)
        let (beta_yl, vcov_yl, n_left) =
            Self::side_fit(y, x, cutoff, h, poly_order, kernel, Side::Left)?;
        let (beta_yr, vcov_yr, _) =
            Self::side_fit(y, x, cutoff, h, poly_order, kernel, Side::Right)?;

        // Primeira etapa: D ~ X  (salto em D)
        let (beta_dl, vcov_dl, _) =
            Self::side_fit(d, x, cutoff, h, poly_order, kernel, Side::Left)?;
        let (beta_dr, vcov_dr, n_right) =
            Self::side_fit(d, x, cutoff, h, poly_order, kernel, Side::Right)?;

        let tau_y = beta_yr[0] - beta_yl[0];
        let tau_d = beta_dr[0] - beta_dl[0];

        if tau_d.abs() < 1e-10 {
            return Err(GreenersError::InvalidOperation(
                "fuzzy_rd: salto na primeira etapa é praticamente zero (τ_D ≈ 0)".into(),
            ));
        }

        // τ̂_FRD = τ_Y / τ_D, delta method SE
        let tau = tau_y / tau_d;
        let var_tau_y = (vcov_yl[[0, 0]] + vcov_yr[[0, 0]]).max(0.0);
        let var_tau_d = (vcov_dl[[0, 0]] + vcov_dr[[0, 0]]).max(0.0);
        let var_tau = (var_tau_y + tau * tau * var_tau_d) / (tau_d * tau_d);
        let se = var_tau.max(0.0).sqrt();

        let var_fs = var_tau_d;
        let se_fs = var_fs.max(0.0).sqrt();

        let z = tau / se;
        let norm = Normal::new(0.0, 1.0).unwrap();
        let p_value = 2.0 * (1.0 - norm.cdf(z.abs()));
        let z95 = 1.959_963_985;

        let (outcome_name, running_name, treatment_name) = variable_names
            .map(|(a, b, c)| (Some(a), Some(b), Some(c)))
            .unwrap_or((None, None, None));

        Ok(RdResult {
            tau,
            se,
            z,
            p_value,
            ci_lower: tau - z95 * se,
            ci_upper: tau + z95 * se,
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

    /// Ajuste de polinômio local em um lado do cutoff (WLS + HC1).
    fn side_fit(
        y: &Array1<f64>,
        x: &Array1<f64>,
        cutoff: f64,
        h: f64,
        poly_order: usize,
        kernel: RdKernel,
        side: Side,
    ) -> Result<(Array1<f64>, Array2<f64>, usize), GreenersError> {
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

        if n < p {
            return Err(GreenersError::ShapeMismatch(format!(
                "rd: observações insuficientes ({n}) para polinômio de ordem {poly_order} (lado {})",
                match side { Side::Left => "esquerdo", Side::Right => "direito" }
            )));
        }

        let (beta, vcov) = local_poly_wls(&ys, &xs, &ws, poly_order)?;
        Ok((beta, vcov, n))
    }

    /// Seletor automático de bandwidth — Imbens-Kalyanaraman (2012), revisão ReStud.
    ///
    /// Para local linear (p=1) com kernel triangular:
    ///   h* = [C_K * (σ²₊ + σ²₋) / (n * f(c) * B²)]^(1/5)
    /// onde B = salto na derivada de segunda ordem.
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
            let (beta, _) = local_poly_wls(&ys, &xs, &ws, q).ok()?;
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

        // Salto na derivada (p+1)-ésima (dividida por (p+1)!)
        let b_jump = m_right - m_left;
        if b_jump.abs() < 1e-12 {
            return h0; // sem curvatura detectável → fallback
        }

        // Densidade em c: contagem na janela piloto / (2 * h0 * n)
        let n_window = x.iter().filter(|&&xi| (xi - cutoff).abs() <= h0).count() as f64;
        let f_c = (n_window / (2.0 * h0 * n)).max(1e-10);

        // Constante triangular / local linear IK 2012 → C_K ≈ 3.4375
        let c_k = 3.4375_f64;
        let exponent = 1.0 / (2.0 * poly_order as f64 + 3.0);

        let h_star =
            (c_k * (sigma2_left + sigma2_right) / (n * f_c * b_jump * b_jump)).powf(exponent);

        // Mantém dentro de [0.05 * sd, 2 * sd]
        h_star.max(0.05 * x_sd).min(2.0 * x_sd)
    }
}

// ── Helpers internos ──────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum Side {
    Left,
    Right,
}

/// Regressão local polinomial por WLS com SE HC1.
///
/// Retorna (β, V̂) onde β[0] = intercepto no cutoff e V̂[0,0] = variância de β[0].
fn local_poly_wls(
    y: &[f64],
    x_centered: &[f64],
    weights: &[f64],
    poly_order: usize,
) -> Result<(Array1<f64>, Array2<f64>), GreenersError> {
    let n = y.len();
    let p = poly_order + 1;

    // X'WX e X'Wy
    let mut xtwx = Array2::<f64>::zeros((p, p));
    let mut xtwy = Array1::<f64>::zeros(p);

    for i in 0..n {
        let w = weights[i];
        let xi: Vec<f64> = (0..p).map(|j| x_centered[i].powi(j as i32)).collect();
        for j in 0..p {
            for k in 0..p {
                xtwx[[j, k]] += w * xi[j] * xi[k];
            }
            xtwy[j] += w * xi[j] * y[i];
        }
    }

    let xtwx_inv = xtwx.inv()?;
    let beta = xtwx_inv.dot(&xtwy);

    // Resíduos
    let resid: Vec<f64> = (0..n)
        .map(|i| {
            let y_hat: f64 = (0..p).map(|j| beta[j] * x_centered[i].powi(j as i32)).sum();
            y[i] - y_hat
        })
        .collect();

    // HC1 meat: Σ w²ᵢ ûᵢ² xᵢxᵢ' * n/(n-p)
    let scale = n as f64 / (n.saturating_sub(p)) as f64;
    let mut meat = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        let w = weights[i];
        let e = resid[i];
        let xi: Vec<f64> = (0..p).map(|j| x_centered[i].powi(j as i32)).collect();
        for j in 0..p {
            for k in 0..p {
                meat[[j, k]] += scale * w * w * e * e * xi[j] * xi[k];
            }
        }
    }

    let vcov = xtwx_inv.dot(&meat).dot(&xtwx_inv);
    Ok((beta, vcov))
}
