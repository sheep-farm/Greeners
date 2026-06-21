/// Heckman Two-Step Selection Model (Heckit)
///
/// Equação de seleção (probit, todos os obs):  z_i* = w_i'γ + u_i,  z_i = 1{z*>0}
/// Equação de resultado (OLS, z_i=1 apenas):   y_i  = x_i'β + ε_i
/// Cov(u, ε) = ρσ_ε  →  E[ε_i | z_i=1, x_i] = ρσ_ε λ_i
///
/// Referência: Heckman (1979) "Sample Selection Bias as a Specification Error".
///             Greene (2012) Econometric Analysis, 7th ed., Cap. 19.

use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use crate::OLS;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

// ── helpers ──────────────────────────────────────────────────────────────────

fn phi(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

fn norm_cdf(x: f64) -> f64 {
    Normal::new(0.0, 1.0).unwrap().cdf(x)
}

// ===========================================================================
// HeckmanResult
// ===========================================================================

#[derive(Debug)]
pub struct HeckmanResult {
    /// Coeficientes da equação de resultado (β)
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    /// Coeficiente sobre λ na equação de resultado (δ = ρ σ_ε)
    pub delta: f64,
    pub delta_se: f64,
    /// ρ̂ = δ̂/σ̂_ε  (correlação entre equações)
    pub rho: f64,
    /// σ̂_ε  (desvio padrão da equação de resultado)
    pub sigma: f64,
    /// Coeficientes do probit de seleção (γ̂)
    pub select_params: Array1<f64>,
    pub select_se: Array1<f64>,
    pub n_obs: usize,
    pub n_selected: usize,
    pub variable_names: Option<Vec<String>>,
    pub select_names: Option<Vec<String>>,
}

impl fmt::Display for HeckmanResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let thick = "═".repeat(70);
        let thin  = "─".repeat(70);
        let sig = |p: f64| {
            if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" }
        };

        writeln!(f, "\n{thick}")?;
        writeln!(f, " Heckman Two-Step (Heckit)  —  Heckman (1979)")?;
        writeln!(f, "{thick}")?;
        writeln!(f, " Obs (total): {:<8}  Selecionadas: {}",
            self.n_obs, self.n_selected)?;
        writeln!(f, " ρ̂: {:.4}   σ̂_ε: {:.4}   δ̂ = ρ̂σ̂_ε: {:.4}",
            self.rho, self.sigma, self.delta)?;
        writeln!(f, "{thin}")?;

        // ── Equação de resultado ──
        writeln!(f, " Equação de resultado  (y | z=1)")?;
        writeln!(f, " {:<18} {:>12}  {:>12}  {:>8}  {:>8}",
            "Variável", "coef", "SE", "z", "P>|z|")?;
        writeln!(f, " {}", "─".repeat(64))?;
        for i in 0..self.params.len() {
            let name = self.variable_names.as_ref()
                .and_then(|v| v.get(i).cloned())
                .unwrap_or_else(|| format!("x{}", i + 1));
            writeln!(f, " {:<18} {:>12.4}  {:>12.4}  {:>8.3}  {:>8.4}  {}",
                name, self.params[i], self.std_errors[i],
                self.t_values[i], self.p_values[i], sig(self.p_values[i]))?;
        }
        writeln!(f, " {:<18} {:>12.4}  {:>12.4}  {:>8.3}  {:>8.4}",
            "lambda (IMR)", self.delta, self.delta_se,
            if self.delta_se > 0.0 { self.delta / self.delta_se } else { f64::NAN },
            {
                let z = if self.delta_se > 0.0 { self.delta / self.delta_se } else { 0.0 };
                2.0 * (1.0 - norm_cdf(z.abs()))
            }
        )?;

        writeln!(f, "\n{thin}")?;
        writeln!(f, " Equação de seleção  (Probit — todos os obs)")?;
        writeln!(f, " {:<18} {:>12}  {:>12}", "Variável", "γ̂", "SE")?;
        writeln!(f, " {}", "─".repeat(44))?;
        for i in 0..self.select_params.len() {
            let name = self.select_names.as_ref()
                .and_then(|v| v.get(i).cloned())
                .unwrap_or_else(|| format!("w{}", i + 1));
            writeln!(f, " {:<18} {:>12.4}  {:>12.4}", name, self.select_params[i], self.select_se[i])?;
        }
        writeln!(f, "{thick}")?;
        writeln!(f, " *** p<0.01  ** p<0.05  * p<0.10")
    }
}

// ===========================================================================
// Heckman estimator
// ===========================================================================

pub struct Heckman;

impl Heckman {
    /// Estima o modelo Heckman Two-Step.
    ///
    /// * `y`            — variável de resultado (n×1, observada apenas para z=1)
    /// * `x_out`        — regressores da equação de resultado (n×k₁, COM intercepto)
    /// * `z`            — indicador de seleção (n×1, valores 0 ou 1)
    /// * `x_sel`        — regressores da equação de seleção (n×k_w, COM intercepto)
    /// * `variable_names` — nomes das colunas de x_out
    /// * `select_names`   — nomes das colunas de x_sel
    pub fn fit(
        y: &Array1<f64>,
        x_out: &Array2<f64>,
        z: &Array1<f64>,
        x_sel: &Array2<f64>,
        variable_names: Option<Vec<String>>,
        select_names: Option<Vec<String>>,
    ) -> Result<HeckmanResult, GreenersError> {
        let n = y.len();
        let k1 = x_out.ncols();   // outcome regressors (incl. intercept)
        let kw = x_sel.ncols();   // selection regressors (incl. intercept)

        if x_out.nrows() != n || z.len() != n || x_sel.nrows() != n {
            return Err(GreenersError::ShapeMismatch(
                "Heckman: dimensões de y, x_out, z e x_sel divergem".into()
            ));
        }
        if y.iter().any(|v| !v.is_finite())
            || x_out.iter().any(|v| !v.is_finite())
            || x_sel.iter().any(|v| !v.is_finite())
        {
            return Err(GreenersError::InvalidOperation(
                "Heckman: dados contêm NaN ou Inf".into()
            ));
        }
        if !z.iter().all(|&v| v == 0.0 || v == 1.0) {
            return Err(GreenersError::InvalidOperation(
                "Heckman: z deve ser binário (0/1)".into()
            ));
        }

        let n_selected: usize = z.iter().filter(|&&v| v == 1.0).count();
        if n_selected < k1 + 1 {
            return Err(GreenersError::ShapeMismatch(
                "Heckman: obs selecionadas insuficientes para a equação de resultado".into()
            ));
        }

        // ── Passo 1: Probit na equação de seleção (todos os obs) ──────────────
        let (gamma, v_gamma) = Self::probit_with_vcov(z, x_sel)?;

        // ── λ_i e δ_i para obs selecionadas ──────────────────────────────────
        let sel_idx: Vec<usize> = (0..n).filter(|&i| z[i] == 1.0).collect();

        let zhat_sel: Vec<f64> = sel_idx.iter()
            .map(|&i| x_sel.row(i).dot(&gamma))
            .collect();

        let lambda_sel: Vec<f64> = zhat_sel.iter().map(|&zh| {
            let phi_val = phi(zh);
            let cdf_val = norm_cdf(zh).max(1e-300);
            phi_val / cdf_val
        }).collect();

        // δ_i = λ_i(λ_i + ẑ_i)  > 0
        let delta_i: Vec<f64> = lambda_sel.iter().zip(zhat_sel.iter())
            .map(|(&lam, &zh)| lam * (lam + zh))
            .collect();

        // ── Passo 2: OLS de y em [X, λ] (obs selecionadas) ───────────────────
        let n1 = sel_idx.len();
        let mut w_aug = Array2::<f64>::zeros((n1, k1 + 1)); // [X₁, λ]
        let mut y1 = Array1::<f64>::zeros(n1);
        for (r, &i) in sel_idx.iter().enumerate() {
            w_aug.row_mut(r).slice_mut(ndarray::s![..k1]).assign(&x_out.row(i));
            w_aug[[r, k1]] = lambda_sel[r];
            y1[r] = y[i];
        }

        let ols = OLS::fit(&y1, &w_aug, crate::CovarianceType::NonRobust)
            .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;

        let beta = ols.params.slice(ndarray::s![..k1]).to_owned();
        let delta_hat = ols.params[k1];

        let resid = &y1 - &w_aug.dot(&ols.params);

        // ── σ̂²_ε corrigido (Heckman 1979) ─────────────────────────────────
        let sum_delta_i: f64 = delta_i.iter().sum();
        let ssr = resid.dot(&resid);
        let sigma2 = (ssr + delta_hat * delta_hat * sum_delta_i) / n1 as f64;
        let sigma = sigma2.sqrt().max(1e-10);

        // ── Variância corrigida de [β̂, δ̂] (Heckman 1979 / Greene 2012) ──────
        // V = σ²(W'W)⁻¹ + δ̂² (W'W)⁻¹ (X_out' D X_sel) V_γ (X_sel' D X_out) (W'W)⁻¹
        //
        // onde D = diag(δ_i), X_sel são as linhas de x_sel para obs selecionadas
        let wtw_inv = {
            let wtw = w_aug.t().dot(&w_aug);
            wtw.inv()?
        };

        // X_sel restrito às obs selecionadas: (n1 × k_w)
        let mut x_sel_1 = Array2::<f64>::zeros((n1, kw));
        for (r, &i) in sel_idx.iter().enumerate() {
            x_sel_1.row_mut(r).assign(&x_sel.row(i));
        }

        // D X_sel: cada linha r de x_sel_1 multiplicada por δ_i[r]
        let mut d_x_sel = x_sel_1.clone();
        for (r, &di) in delta_i.iter().enumerate() {
            d_x_sel.row_mut(r).mapv_inplace(|v| v * di);
        }

        // D X_out (k1+1 colunas = w_aug)
        let mut d_x_out = w_aug.clone();
        for (r, &di) in delta_i.iter().enumerate() {
            d_x_out.row_mut(r).mapv_inplace(|v| v * di);
        }

        // (X_out' D X_sel) é (k1+1 × k_w)
        let xtd_xs = d_x_out.t().dot(&x_sel_1); // = X_out' D X_sel
        // Correction meat: (k1+1 × k1+1)
        let correction_meat = xtd_xs.dot(&v_gamma).dot(&xtd_xs.t());
        let correction = &wtw_inv.dot(&correction_meat).dot(&wtw_inv) * (delta_hat * delta_hat);

        let vcov = &wtw_inv * sigma2 + correction;

        let std_errors: Array1<f64> = (0..k1)
            .map(|i| vcov[[i, i]].max(0.0).sqrt())
            .collect::<Vec<_>>().into();
        let delta_se = vcov[[k1, k1]].max(0.0).sqrt();

        let norm = Normal::new(0.0, 1.0)
            .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let t_values = &beta / &std_errors;
        let p_values: Array1<f64> = t_values.mapv(|z| 2.0 * (1.0 - norm.cdf(z.abs())));

        let rho = (delta_hat / sigma).clamp(-1.0, 1.0);

        // SE do probit (diagonal de V_γ)
        let select_se: Array1<f64> = (0..kw)
            .map(|i| v_gamma[[i, i]].max(0.0).sqrt())
            .collect::<Vec<_>>().into();

        Ok(HeckmanResult {
            params: beta,
            std_errors,
            t_values,
            p_values,
            delta: delta_hat,
            delta_se,
            rho,
            sigma,
            select_params: gamma,
            select_se,
            n_obs: n,
            n_selected,
            variable_names,
            select_names,
        })
    }

    /// Probit via Newton-Raphson. Retorna (γ̂, V_γ) onde V_γ = (-H)⁻¹.
    fn probit_with_vcov(
        y: &Array1<f64>,
        x: &Array2<f64>,
    ) -> Result<(Array1<f64>, Array2<f64>), GreenersError> {
        let n = y.len();
        let k = x.ncols();

        let mut beta = Array1::<f64>::zeros(k);
        let tol = 1e-7;
        let max_iter = 200;
        let mut iter = 0;

        loop {
            let xb = x.dot(&beta);
            let p: Array1<f64> = xb.mapv(|v| norm_cdf(v).clamp(1e-15, 1.0 - 1e-15));
            let phi_v: Array1<f64> = xb.mapv(phi);

            // Score: Σ_i [(y_i - p_i) / (p_i(1-p_i))] * φ(xb_i) * x_i
            let mut grad = Array1::<f64>::zeros(k);
            for i in 0..n {
                let w = phi_v[i] / (p[i] * (1.0 - p[i]));
                let score_i = (y[i] - p[i]) * w;
                grad.scaled_add(score_i, &x.row(i));
            }

            // Hessian: -Σ_i [φ(xb_i)² / (p_i(1-p_i))] * x_i x_i'
            let mut neg_h = Array2::<f64>::zeros((k, k));
            for i in 0..n {
                let w = phi_v[i] * phi_v[i] / (p[i] * (1.0 - p[i]));
                let xi = x.row(i);
                for j in 0..k {
                    for kk in 0..k {
                        neg_h[[j, kk]] += w * xi[j] * xi[kk];
                    }
                }
            }

            let neg_h_inv = match neg_h.inv() {
                Ok(m) => m,
                Err(_) => return Err(GreenersError::OptimizationFailed),
            };
            let step = neg_h_inv.dot(&grad);
            let diff = step.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
            beta = &beta + &step;
            iter += 1;

            if diff < tol || iter >= max_iter { break; }
        }

        if iter >= max_iter {
            return Err(GreenersError::OptimizationFailed);
        }

        // V_γ = (-H)⁻¹ na convergência
        let xb = x.dot(&beta);
        let p: Array1<f64> = xb.mapv(|v| norm_cdf(v).clamp(1e-15, 1.0 - 1e-15));
        let phi_v: Array1<f64> = xb.mapv(phi);
        let mut neg_h = Array2::<f64>::zeros((k, k));
        for i in 0..n {
            let w = phi_v[i] * phi_v[i] / (p[i] * (1.0 - p[i]));
            let xi = x.row(i);
            for j in 0..k {
                for kk in 0..k {
                    neg_h[[j, kk]] += w * xi[j] * xi[kk];
                }
            }
        }
        let v_gamma = neg_h.inv()?;
        Ok((beta, v_gamma))
    }
}
