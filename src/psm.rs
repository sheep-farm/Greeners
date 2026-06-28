use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

// ── Structs de resultado ──────────────────────────────────────────────────────

#[derive(Debug)]
pub struct BalanceRow {
    pub covariate: String,
    pub mean_treated: f64,
    pub mean_control_raw: f64,
    pub mean_control_matched: f64,
    /// Diferença padronizada antes do matching (|μ_T - μ_C| / σ_pooled).
    pub smd_before: f64,
    /// Diferença padronizada após o matching.
    pub smd_after: f64,
}

#[derive(Debug)]
pub struct PsmResult {
    /// Efeito médio do tratamento nos tratados (ATT).
    pub att: f64,
    pub se: f64,
    pub z: f64,
    pub p_value: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub n_treated: usize,
    pub n_control: usize,
    /// Unidades tratadas com ao menos um match (descartadas se caliper viola).
    pub n_matched_treated: usize,
    /// Pares/grupos de matching: (idx_tratado, [idxs_controle]).
    pub matched_pairs: Vec<(usize, Vec<usize>)>,
    /// Propensity scores estimados (todos os obs, na ordem original).
    pub propensity_scores: Array1<f64>,
    pub balance: Vec<BalanceRow>,
    pub outcome_name: String,
    pub treatment_name: String,
    pub covariate_names: Vec<String>,
    pub k: usize,
    pub caliper: Option<f64>,
    pub n_boot: usize,
}

impl fmt::Display for PsmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let thick = "═".repeat(72);
        let thin  = "─".repeat(72);
        let sig = |p: f64| if p < 0.01 { "***" } else if p < 0.05 { "**" } else if p < 0.10 { "*" } else { "" };

        writeln!(f, "\n{thick}")?;
        writeln!(f, " Propensity Score Matching  —  ATT")?;
        writeln!(f, "{thick}")?;
        writeln!(f, " Outcome: {}   Tratamento: {}",
            self.outcome_name, self.treatment_name)?;
        let cal_str = self.caliper.map(|c| format!("{c:.4}")).unwrap_or("nenhum".into());
        writeln!(f, " k={} match   Caliper: {}   Bootstrap SE: {} reps",
            self.k, cal_str, self.n_boot)?;
        writeln!(f, " N tratados: {}   N controles: {}   N tratados matchados: {}",
            self.n_treated, self.n_control, self.n_matched_treated)?;
        writeln!(f, "{thin}")?;
        writeln!(f, " ATT = {:.4}   SE = {:.4}   z = {:.3}   P>|z| = {:.4}  {}",
            self.att, self.se, self.z, self.p_value, sig(self.p_value))?;
        writeln!(f, " IC 95%: [{:.4}, {:.4}]", self.ci_lower, self.ci_upper)?;
        writeln!(f, "{thin}")?;

        // Tabela de balanço
        writeln!(f, " Balanço de covariáveis (SMD = diferença padronizada):")?;
        writeln!(f, " {:<20} {:>10}  {:>10}  {:>10}  {:>8}  {:>8}",
            "Covariável", "μ_Trat", "μ_Ctrl(raw)", "μ_Ctrl(mtch)", "SMD_ant", "SMD_dep")?;
        writeln!(f, " {}", "─".repeat(70))?;
        for row in &self.balance {
            let flag = if row.smd_after.abs() > 0.1 { " !" } else { "  " };
            writeln!(f, "{flag}{:<20} {:>10.4}  {:>10.4}  {:>10.4}  {:>8.3}  {:>8.3}",
                row.covariate, row.mean_treated, row.mean_control_raw,
                row.mean_control_matched, row.smd_before, row.smd_after)?;
        }
        writeln!(f, " (!) SMD > 0.10 após matching — covariável mal balanceada")?;
        writeln!(f, "{thick}")?;
        writeln!(f, " *** p<0.01  ** p<0.05  * p<0.10")
    }
}

// ── PSM ───────────────────────────────────────────────────────────────────────

pub struct PSM;

impl PSM {
    /// Estima ATT por Propensity Score Matching.
    ///
    /// * `y`               — resultado (todos os obs)
    /// * `d`               — tratamento (0/1, todos os obs)
    /// * `x`               — covariáveis SEM intercepto e SEM tratamento (n × p)
    /// * `k`               — número de controles por tratado (padrão 1)
    /// * `caliper`         — limite máximo de distância no PS (None = sem caliper)
    /// * `with_replacement`— reposição no matching (padrão false)
    /// * `n_boot`          — replicações bootstrap para SE (padrão 200)
    /// * `variable_names`  — (outcome, treatment, covariates)
    pub fn fit(
        y: &Array1<f64>,
        d: &Array1<f64>,
        x: &Array2<f64>,
        k: usize,
        caliper: Option<f64>,
        with_replacement: bool,
        n_boot: usize,
        variable_names: Option<(String, String, Vec<String>)>,
    ) -> Result<PsmResult, GreenersError> {
        let n = y.len();
        if d.len() != n || x.nrows() != n {
            return Err(GreenersError::ShapeMismatch(
                "psm: y, d, x devem ter o mesmo número de observações".into()
            ));
        }
        if y.iter().chain(d.iter()).chain(x.iter()).any(|v| !v.is_finite()) {
            return Err(GreenersError::InvalidOperation(
                "psm: dados contêm NaN ou Inf".into()
            ));
        }
        if k == 0 {
            return Err(GreenersError::InvalidOperation("psm: k deve ser ≥ 1".into()));
        }

        // ── 1. Adicionar intercepto às covariáveis para o logit ───────────────
        let x_aug = add_intercept(x);

        // ── 2. Estimar propensity scores via logit ────────────────────────────
        let beta = fit_logit(d, &x_aug)?;
        let ps   = predict_proba(&beta, &x_aug);

        // ── 3. Matching NN ────────────────────────────────────────────────────
        let ps_vec: Vec<f64> = ps.to_vec();
        let d_vec:  Vec<f64> = d.to_vec();
        let matched_pairs = nearest_neighbor_match(&ps_vec, &d_vec, k, caliper, with_replacement);

        // ── 4. ATT ────────────────────────────────────────────────────────────
        let att = compute_att(y, &matched_pairs);
        if !att.is_finite() {
            return Err(GreenersError::InvalidOperation(
                "psm: ATT não calculável — nenhum tratado obteve match".into()
            ));
        }

        // ── 5. Bootstrap SE ───────────────────────────────────────────────────
        let se = bootstrap_se(y, d, &x_aug, k, caliper, with_replacement, n_boot);

        // ── 6. Inferência ─────────────────────────────────────────────────────
        let z = att / se;
        let normal_dist = Normal::new(0.0, 1.0).unwrap();
        let p_value = 2.0 * (1.0 - normal_dist.cdf(z.abs()));
        let z95 = 1.959_963_985;

        // ── 7. Tamanhos de amostra ────────────────────────────────────────────
        let n_treated = d_vec.iter().filter(|&&di| di > 0.5).count();
        let n_control = n - n_treated;
        let n_matched_treated = matched_pairs.iter().filter(|(_, cs)| !cs.is_empty()).count();

        // ── 8. Balanço ────────────────────────────────────────────────────────
        let (outcome_name, treatment_name, cov_names) = variable_names
            .unwrap_or_else(|| (
                "y".into(),
                "d".into(),
                (0..x.ncols()).map(|i| format!("x{}", i+1)).collect(),
            ));

        let balance = compute_balance(x, d, &matched_pairs, &cov_names);

        Ok(PsmResult {
            att, se, z, p_value,
            ci_lower: att - z95 * se,
            ci_upper: att + z95 * se,
            n_treated, n_control, n_matched_treated,
            matched_pairs,
            propensity_scores: ps,
            balance,
            outcome_name, treatment_name, covariate_names: cov_names,
            k, caliper, n_boot,
        })
    }
}

// ── Logit para propensity score ───────────────────────────────────────────────

fn add_intercept(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let p = x.ncols();
    let mut out = Array2::<f64>::ones((n, p + 1));
    for i in 0..n {
        for j in 0..p {
            out[[i, j + 1]] = x[[i, j]];
        }
    }
    out
}

fn fit_logit(d: &Array1<f64>, x: &Array2<f64>) -> Result<Array1<f64>, GreenersError> {
    let n = d.len();
    let k = x.ncols();
    let mut beta = Array1::<f64>::zeros(k);

    for _ in 0..100 {
        let xb  = x.dot(&beta);
        let p: Array1<f64> = xb.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        let w: Array1<f64> = p.mapv(|pi| (pi * (1.0 - pi)).max(1e-12));
        let resid = d - &p;

        // Score e Hessian
        let mut score = Array1::<f64>::zeros(k);
        let mut hess  = Array2::<f64>::zeros((k, k));
        for i in 0..n {
            let xi = x.row(i);
            score.scaled_add(resid[i], &xi);
            for j in 0..k {
                for l in 0..k {
                    hess[[j, l]] -= w[i] * xi[j] * xi[l];
                }
            }
        }

        let neg_hess = hess.mapv(|v| -v);
        let neg_hess_inv = neg_hess.inv()?;
        let step = neg_hess_inv.dot(&score);
        let diff: f64 = step.iter().map(|v| v.abs()).sum();
        beta = beta + step;
        if diff < 1e-8 { break; }
    }
    Ok(beta)
}

fn predict_proba(beta: &Array1<f64>, x: &Array2<f64>) -> Array1<f64> {
    x.dot(beta).mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

// ── Matching ──────────────────────────────────────────────────────────────────

fn nearest_neighbor_match(
    ps: &[f64],
    d: &[f64],
    k: usize,
    caliper: Option<f64>,
    with_replacement: bool,
) -> Vec<(usize, Vec<usize>)> {
    let treated: Vec<usize> = d.iter().enumerate()
        .filter(|&(_, &di)| di > 0.5).map(|(i, _)| i).collect();
    let control: Vec<usize> = d.iter().enumerate()
        .filter(|&(_, &di)| di <= 0.5).map(|(i, _)| i).collect();

    let mut matched: Vec<(usize, Vec<usize>)> = Vec::with_capacity(treated.len());
    let mut used = std::collections::HashSet::<usize>::new();

    for &ti in &treated {
        let ps_t = ps[ti];

        let mut cands: Vec<(usize, f64)> = control.iter()
            .filter(|&&ci| {
                let dist = (ps_t - ps[ci]).abs();
                caliper.map_or(true, |cap| dist <= cap)
            })
            .filter(|&&ci| with_replacement || !used.contains(&ci))
            .map(|&ci| (ci, (ps_t - ps[ci]).abs()))
            .collect();

        cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let matches: Vec<usize> = cands.iter().take(k).map(|(ci, _)| *ci).collect();
        if !with_replacement {
            for &ci in &matches { used.insert(ci); }
        }
        matched.push((ti, matches));
    }

    matched
}

fn compute_att(y: &Array1<f64>, pairs: &[(usize, Vec<usize>)]) -> f64 {
    let mut total = 0.0_f64;
    let mut count = 0usize;
    for (ti, cis) in pairs {
        if cis.is_empty() { continue; }
        let y_ctrl = cis.iter().map(|&ci| y[ci]).sum::<f64>() / cis.len() as f64;
        total += y[*ti] - y_ctrl;
        count += 1;
    }
    if count == 0 { f64::NAN } else { total / count as f64 }
}

// ── Bootstrap SE ─────────────────────────────────────────────────────────────

fn bootstrap_se(
    y: &Array1<f64>,
    d: &Array1<f64>,
    x_aug: &Array2<f64>,
    k: usize,
    caliper: Option<f64>,
    with_replacement: bool,
    n_boot: usize,
) -> f64 {
    let n = y.len();
    let mut att_boot: Vec<f64> = Vec::with_capacity(n_boot);
    let mut state = 0x123456789abcdef0u64;

    for _ in 0..n_boot {
        let idx: Vec<usize> = (0..n).map(|_| lcg_next(&mut state) % n).collect();

        let y_b: Array1<f64> = idx.iter().map(|&i| y[i]).collect::<Vec<_>>().into();
        let d_b: Array1<f64> = idx.iter().map(|&i| d[i]).collect::<Vec<_>>().into();
        let x_b: Array2<f64> = {
            let mut m = Array2::<f64>::zeros((n, x_aug.ncols()));
            for (r, &i) in idx.iter().enumerate() {
                for c in 0..x_aug.ncols() { m[[r, c]] = x_aug[[i, c]]; }
            }
            m
        };

        // Requer ao menos 1 tratado e 1 controle no resample
        let n_t_b = d_b.iter().filter(|&&v| v > 0.5).count();
        let n_c_b = n - n_t_b;
        if n_t_b == 0 || n_c_b == 0 { continue; }

        let Ok(beta_b) = fit_logit(&d_b, &x_b) else { continue };
        let ps_b  = predict_proba(&beta_b, &x_b);
        let ps_v  = ps_b.to_vec();
        let dv    = d_b.to_vec();
        let pairs = nearest_neighbor_match(&ps_v, &dv, k, caliper, with_replacement);
        let att_b = compute_att(&y_b, &pairs);
        if att_b.is_finite() { att_boot.push(att_b); }
    }

    if att_boot.len() < 10 { return f64::NAN; }
    let mean = att_boot.iter().sum::<f64>() / att_boot.len() as f64;
    let var  = att_boot.iter().map(|&v| (v - mean).powi(2)).sum::<f64>()
        / (att_boot.len() - 1) as f64;
    var.sqrt()
}

fn lcg_next(s: &mut u64) -> usize {
    *s = s.wrapping_mul(6_364_136_223_846_793_005)
          .wrapping_add(1_442_695_040_888_963_407);
    (*s >> 33) as usize
}

// ── Tabela de balanço ─────────────────────────────────────────────────────────

fn compute_balance(
    x: &Array2<f64>,
    d: &Array1<f64>,
    pairs: &[(usize, Vec<usize>)],
    cov_names: &[String],
) -> Vec<BalanceRow> {
    let n = d.len();
    let p = x.ncols();

    let treated_idx: Vec<usize> = (0..n).filter(|&i| d[i] > 0.5).collect();
    let control_idx: Vec<usize> = (0..n).filter(|&i| d[i] <= 0.5).collect();
    let matched_ctrl: Vec<usize> = pairs.iter()
        .flat_map(|(_, cs)| cs.iter().cloned())
        .collect();

    (0..p).map(|j| {
        let col: Vec<f64> = (0..n).map(|i| x[[i, j]]).collect();

        let mu_t = mean_at(&col, &treated_idx);
        let mu_c = mean_at(&col, &control_idx);
        let mu_m = if matched_ctrl.is_empty() { f64::NAN } else { mean_at(&col, &matched_ctrl) };

        let sd_t = std_at(&col, &treated_idx);
        let sd_c = std_at(&col, &control_idx);
        let sd_pool = ((sd_t * sd_t + sd_c * sd_c) / 2.0).sqrt().max(1e-10);

        BalanceRow {
            covariate: cov_names.get(j).cloned().unwrap_or_else(|| format!("x{}", j + 1)),
            mean_treated: mu_t,
            mean_control_raw: mu_c,
            mean_control_matched: mu_m,
            smd_before: (mu_t - mu_c) / sd_pool,
            smd_after:  if mu_m.is_finite() { (mu_t - mu_m) / sd_pool } else { f64::NAN },
        }
    }).collect()
}

fn mean_at(v: &[f64], idx: &[usize]) -> f64 {
    if idx.is_empty() { return f64::NAN; }
    idx.iter().map(|&i| v[i]).sum::<f64>() / idx.len() as f64
}

fn std_at(v: &[f64], idx: &[usize]) -> f64 {
    if idx.len() < 2 { return 0.0; }
    let mu = mean_at(v, idx);
    let var = idx.iter().map(|&i| (v[i] - mu).powi(2)).sum::<f64>() / (idx.len() - 1) as f64;
    var.sqrt()
}
