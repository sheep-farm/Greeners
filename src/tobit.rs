use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use crate::OLS;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

// ── Helpers numéricos ────────────────────────────────────────────────────────

fn phi(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

fn norm_cdf(x: f64) -> f64 {
    Normal::new(0.0, 1.0).unwrap().cdf(x)
}


// ===========================================================================
// TobitResult
// ===========================================================================

#[derive(Debug)]
pub struct TobitResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub sigma: f64,
    pub log_likelihood: f64,
    pub n_obs: usize,
    pub n_censored: usize,
    pub df_resid: usize,
    pub ll: f64,
    pub iterations: usize,
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for TobitResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let thick = "═".repeat(70);
        let thin  = "─".repeat(70);
        let n_unc = self.n_obs - self.n_censored;
        writeln!(f, "\n{thick}")?;
        writeln!(f, " Tobit  —  MLE  (censura inferior em {})", self.ll)?;
        writeln!(f, "{thick}")?;
        writeln!(f, " Obs: {:<8}  Censuradas: {:<6}  Não-cens.: {:<6}  Iter.: {}",
            self.n_obs, self.n_censored, n_unc, self.iterations)?;
        writeln!(f, " Log-L: {:.4}   σ: {:.4}   df_resid: {}",
            self.log_likelihood, self.sigma, self.df_resid)?;
        writeln!(f, "{thin}")?;
        writeln!(f, " {:<18} {:>12}  {:>12}  {:>8}  {:>8}",
            "Variável", "coef", "SE", "z", "P>|z|")?;
        writeln!(f, " {}", "─".repeat(64))?;
        let sig = |p: f64| if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };
        for i in 0..self.params.len() {
            let name = self.variable_names.as_ref()
                .and_then(|v| v.get(i).cloned())
                .unwrap_or_else(|| format!("x{}", i + 1));
            writeln!(f, " {:<18} {:>12.4}  {:>12.4}  {:>8.3}  {:>8.4}  {}",
                name, self.params[i], self.std_errors[i],
                self.t_values[i], self.p_values[i], sig(self.p_values[i]))?;
        }
        writeln!(f, " {}", "─".repeat(64))?;
        writeln!(f, " sigma            {:>12.4}", self.sigma)?;
        writeln!(f, "{thick}")?;
        writeln!(f, " *** p<0.01  ** p<0.05  * p<0.10")
    }
}

// ===========================================================================
// Tobit MLE — censura esquerda em `ll` (default 0)
// ===========================================================================

pub struct Tobit;

impl Tobit {
    /// Estima modelo Tobit por MLE via Newton-Raphson.
    ///
    /// * `y`  — variável dependente (permite y_i = ll para censuradas)
    /// * `x`  — regressores COM intercepto (n × k)
    /// * `ll` — limite inferior de censura (default 0.0)
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        ll: f64,
        variable_names: Option<Vec<String>>,
    ) -> Result<TobitResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();

        if x.nrows() != n {
            return Err(GreenersError::ShapeMismatch("Tobit: y e x têm dimensões incompatíveis".into()));
        }
        if y.iter().any(|v| !v.is_finite()) || x.iter().any(|v| !v.is_finite()) {
            return Err(GreenersError::InvalidOperation("Tobit: dados contêm NaN ou Inf".into()));
        }
        if n <= k {
            return Err(GreenersError::ShapeMismatch("Tobit: graus de liberdade insuficientes".into()));
        }

        // ── Indicador de censura ──
        let d: Vec<bool> = y.iter().map(|&yi| yi > ll).collect();
        let n_censored = d.iter().filter(|&&b| !b).count();

        // ── Inicialização: OLS nos não-censurados ──
        let unc_idx: Vec<usize> = (0..n).filter(|&i| d[i]).collect();
        let y_unc: Array1<f64> = unc_idx.iter().map(|&i| y[i]).collect::<Vec<_>>().into();
        let x_unc: Array2<f64> = {
            let rows: Vec<ndarray::ArrayView1<f64>> =
                unc_idx.iter().map(|&i| x.row(i)).collect();
            ndarray::stack(ndarray::Axis(0), &rows).unwrap()
        };

        let ols_init = OLS::fit(&y_unc, &x_unc, crate::CovarianceType::NonRobust)
            .unwrap_or_else(|_| {
                OLS::fit(y, x, crate::CovarianceType::NonRobust)
                    .expect("Tobit: falha na inicialização OLS")
            });

        let mut beta = ols_init.params.clone();
        let init_sigma = {
            let resid = &y_unc - &x_unc.dot(&beta);
            let ssr = resid.dot(&resid);
            (ssr / (y_unc.len().saturating_sub(k)) as f64).sqrt().max(1e-6)
        };
        let mut gamma = init_sigma.ln(); // γ = ln σ

        let norm = Normal::new(0.0, 1.0)
            .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;

        let tol = 1e-7;
        let max_iter = 200;
        let mut iter = 0;
        let mut log_lik = f64::NEG_INFINITY;

        loop {
            let sigma = gamma.exp();
            let s2 = sigma * sigma;

            // ── Gradient e Hessian ──
            let mut g_beta = Array1::<f64>::zeros(k);
            let mut g_gamma = 0.0_f64;

            // Hessian k+1 × k+1 organizado como bloco [[H_bb, H_bg], [H_bg', H_gg]]
            let mut h_bb = Array2::<f64>::zeros((k, k));
            let mut h_bg = Array1::<f64>::zeros(k);
            let mut h_gg = 0.0_f64;

            let mut ll_val = 0.0_f64;
            const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_7;

            for i in 0..n {
                let xb = x.row(i).dot(&beta);
                if d[i] {
                    // Não censurada
                    let e = (y[i] - xb) / sigma;
                    ll_val += -gamma - LOG_SQRT_2PI - 0.5 * e * e;

                    let e_s = e / sigma; // e/σ
                    g_beta.scaled_add(e_s, &x.row(i));
                    g_gamma += e * e - 1.0;

                    // H_bb[j,k] -= x_ij * x_ik / σ²
                    let xi = x.row(i);
                    for j in 0..k {
                        for kk in 0..k {
                            h_bb[[j, kk]] -= xi[j] * xi[kk] / s2;
                        }
                    }
                    // H_bg[j] -= 2*e/σ * x_ij
                    h_bg.scaled_add(-2.0 * e / sigma, &xi);
                    // H_gg -= 2*e²
                    h_gg -= 2.0 * e * e;
                } else {
                    // Censurada
                    let a = (xb - ll) / sigma;
                    let phi_neg = norm_cdf(-a).max(1e-300);
                    ll_val += phi_neg.ln();

                    let lam = phi(a) / phi_neg;
                    let delta = lam * (lam - a); // dλ/da > 0

                    // g_beta -= λ/σ * x_i
                    g_beta.scaled_add(-lam / sigma, &x.row(i));
                    // g_gamma += λ*a
                    g_gamma += lam * a;

                    let xi = x.row(i);
                    // H_bb -= δ/σ² * x_i x_i'
                    for j in 0..k {
                        for kk in 0..k {
                            h_bb[[j, kk]] -= delta * xi[j] * xi[kk] / s2;
                        }
                    }
                    // termo comum para H_bg e H_gg
                    let c = lam * (a * (lam - a) + 1.0);
                    // H_bg += c/σ * x_i
                    h_bg.scaled_add(c / sigma, &xi);
                    // H_gg -= a*λ*(a*(λ-a)+1)
                    h_gg -= a * c;
                }
            }

            // ── Monta Hessian k+1 × k+1 e gradiente k+1 ──
            let m = k + 1;
            let mut h_full = Array2::<f64>::zeros((m, m));
            let mut g_full = Array1::<f64>::zeros(m);

            for j in 0..k {
                for kk in 0..k {
                    h_full[[j, kk]] = h_bb[[j, kk]];
                }
                h_full[[j, k]] = h_bg[j];
                h_full[[k, j]] = h_bg[j];
                g_full[j] = g_beta[j];
            }
            h_full[[k, k]] = h_gg;
            g_full[k] = g_gamma;

            // Newton: θ += (-H)⁻¹ g  (sobe na log-verossimilhança)
            let neg_h = h_full.mapv(|v| -v);
            let neg_h_inv = match neg_h.inv() {
                Ok(m) => m,
                Err(_) => return Err(GreenersError::OptimizationFailed),
            };
            let step = neg_h_inv.dot(&g_full);

            // backtracking line search para garantir subida
            let mut alpha = 1.0_f64;
            for _ in 0..20 {
                let b_new = &beta + &step.slice(ndarray::s![..k]).to_owned() * alpha;
                let g_new = gamma + step[k] * alpha;
                let ll_new = Self::log_lik(y, x, &d, ll, &b_new, g_new, &norm);
                if ll_new > ll_val - 1e-10 {
                    beta = b_new;
                    gamma = g_new;
                    break;
                }
                alpha *= 0.5;
            }

            let diff = (log_lik - ll_val).abs();
            log_lik = ll_val;
            iter += 1;

            if diff < tol || iter >= max_iter {
                break;
            }
        }

        if iter >= max_iter {
            return Err(GreenersError::OptimizationFailed);
        }

        // ── SE: diagonal de (-H)⁻¹ na convergência ──
        let sigma = gamma.exp();
        let s2 = sigma * sigma;

        let mut h_bb = Array2::<f64>::zeros((k, k));
        let mut h_bg = Array1::<f64>::zeros(k);
        let mut h_gg = 0.0_f64;

        for i in 0..n {
            let xb = x.row(i).dot(&beta);
            let xi = x.row(i);
            if d[i] {
                let e = (y[i] - xb) / sigma;
                for j in 0..k {
                    for kk in 0..k {
                        h_bb[[j, kk]] -= xi[j] * xi[kk] / s2;
                    }
                }
                h_bg.scaled_add(-2.0 * e / sigma, &xi);
                h_gg -= 2.0 * e * e;
            } else {
                let a = (xb - ll) / sigma;
                let phi_neg = norm_cdf(-a).max(1e-300);
                let lam = phi(a) / phi_neg;
                let delta = lam * (lam - a);
                let c = lam * (a * (lam - a) + 1.0);
                for j in 0..k {
                    for kk in 0..k {
                        h_bb[[j, kk]] -= delta * xi[j] * xi[kk] / s2;
                    }
                }
                h_bg.scaled_add(c / sigma, &xi);
                h_gg -= a * c;
            }
        }

        let m = k + 1;
        let mut h_full = Array2::<f64>::zeros((m, m));
        for j in 0..k {
            for kk in 0..k { h_full[[j, kk]] = h_bb[[j, kk]]; }
            h_full[[j, k]] = h_bg[j];
            h_full[[k, j]] = h_bg[j];
        }
        h_full[[k, k]] = h_gg;

        let neg_h = h_full.mapv(|v| -v);
        let vcov = neg_h.inv()?;

        let std_errors: Array1<f64> = (0..k)
            .map(|i| vcov[[i, i]].max(0.0).sqrt())
            .collect::<Vec<_>>().into();
        let t_values = &beta / &std_errors;
        let p_values: Array1<f64> = t_values.mapv(|z| 2.0 * (1.0 - norm.cdf(z.abs())));

        Ok(TobitResult {
            params: beta,
            std_errors,
            t_values,
            p_values,
            sigma,
            log_likelihood: log_lik,
            n_obs: n,
            n_censored,
            df_resid: n - k,
            ll,
            iterations: iter,
            variable_names,
        })
    }

    fn log_lik(
        y: &Array1<f64>,
        x: &Array2<f64>,
        d: &[bool],
        ll: f64,
        beta: &Array1<f64>,
        gamma: f64,
        norm: &Normal,
    ) -> f64 {
        let sigma = gamma.exp();
        const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_7;
        let mut val = 0.0_f64;
        for i in 0..y.len() {
            let xb = x.row(i).dot(beta);
            if d[i] {
                let e = (y[i] - xb) / sigma;
                val += -gamma - LOG_SQRT_2PI - 0.5 * e * e;
            } else {
                let a = (xb - ll) / sigma;
                val += norm.cdf(-a).max(1e-300).ln();
            }
        }
        val
    }
}
