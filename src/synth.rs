use crate::error::GreenersError;
use crate::DataFrame;
use ndarray::{Array1, Array2};
use indexmap::IndexMap;
use std::fmt;

// ── SynthResult ───────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct SynthResult {
    /// Pesos dos doadores: (id_string, peso). Inclui pesos ≈ 0.
    pub weights: Vec<(String, f64)>,
    /// Série do controle sintético para todos os períodos T.
    pub synthetic_series: Vec<f64>,
    /// Série observada da unidade tratada para todos os períodos T.
    pub actual_series: Vec<f64>,
    /// Valores de tempo ordenados (ex: anos).
    pub time_index: Vec<f64>,
    /// Primeiro período pós-tratamento.
    pub t0: f64,
    /// RMSPE no período pré-tratamento (qualidade do ajuste).
    pub rmspe_pre: f64,
    /// RMSPE no período pós-tratamento (magnitude do efeito estimado).
    pub rmspe_post: Option<f64>,
    pub treated_unit: String,
    pub outcome_name: String,
    pub n_donors: usize,
    pub t_pre: usize,
    pub t_post: usize,
}

impl fmt::Display for SynthResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let thick = "═".repeat(68);
        let thin = "─".repeat(68);
        writeln!(f, "\n{thick}")?;
        writeln!(
            f,
            " Controle Sintético  —  Abadie, Diamond, Hainmueller (2010)"
        )?;
        writeln!(f, "{thick}")?;
        writeln!(
            f,
            " Unidade tratada : {}   Outcome: {}",
            self.treated_unit, self.outcome_name
        )?;
        writeln!(
            f,
            " T₀ (1ª pós-trat): {}   Doadores: {}   T pré: {}   T pós: {}",
            self.t0, self.n_donors, self.t_pre, self.t_post
        )?;
        writeln!(f, " RMSPE pré  : {:.4}", self.rmspe_pre)?;
        if let Some(rp) = self.rmspe_post {
            let ratio = if self.rmspe_pre > 1e-12 {
                rp / self.rmspe_pre
            } else {
                f64::NAN
            };
            writeln!(f, " RMSPE pós  : {:.4}   razão pós/pré: {:.3}", rp, ratio)?;
        }
        writeln!(f, "{thin}")?;
        writeln!(f, " Pesos dos doadores (w > 0.001):")?;
        let mut nonzero: Vec<&(String, f64)> =
            self.weights.iter().filter(|(_, w)| *w > 0.001).collect();
        nonzero.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if nonzero.is_empty() {
            writeln!(f, "   (nenhum peso > 0.001)")?;
        } else {
            for (unit, w) in &nonzero {
                writeln!(f, "   {:<24}  {:.4}", unit, w)?;
            }
        }
        writeln!(f, "{thin}")?;
        writeln!(
            f,
            " {:>8}  {:>12}  {:>12}  {:>12}",
            "Período", "Real", "Sintético", "Efeito"
        )?;
        writeln!(f, " {}", "─".repeat(50))?;
        for (i, &t) in self.time_index.iter().enumerate() {
            let actual = self.actual_series[i];
            let synth = self.synthetic_series[i];
            let post = t >= self.t0;
            let effect_str = if post {
                format!("{:>12.4}", actual - synth)
            } else {
                "            ".to_string()
            };
            let marker = if post { " *" } else { "  " };
            writeln!(
                f,
                "{marker}{:>8.0}  {:>12.4}  {:>12.4}  {}",
                t, actual, synth, effect_str
            )?;
        }
        writeln!(f, "{thick}")?;
        writeln!(f, " * pós-tratamento")
    }
}

// ── SyntheticControl ──────────────────────────────────────────────────────────

pub struct SyntheticControl;

impl SyntheticControl {
    /// Estima o Controle Sintético por mínimos quadrados com restrição de simplex.
    ///
    /// Parâmetros
    /// ----------
    /// * `outcome_col`     — coluna com a variável de resultado
    /// * `treated_unit`    — ID (string ou numérico) da unidade tratada
    /// * `t0`              — primeiro período pós-tratamento
    /// * `df`              — painel em formato longo (uma linha por unidade × período)
    /// * `id_col`          — coluna de ID de entidade
    /// * `time_col`        — coluna de tempo
    /// * `covariate_cols`  — covariáveis para matching (None → apenas séries temporais)
    ///
    /// Algoritmo: minimiza ||Y₁_pré - Y₀_pré w||² (+ penalidade de covariáveis)
    /// sujeito a w ≥ 0, Σwⱼ = 1.  Solver: gradiente projetado no simplex.
    pub fn fit(
        outcome_col: &str,
        treated_unit: &str,
        t0: f64,
        df: &DataFrame,
        id_col: &str,
        time_col: &str,
        covariate_cols: Option<&[String]>,
    ) -> Result<SynthResult, GreenersError> {
        // ── 1. Extrair colunas base ───────────────────────────────────────────
        let y_col = df.get(outcome_col)?.to_owned();
        let t_col = df.get(time_col)?.to_owned();
        let n = df.n_rows();

        // IDs como strings
        let unit_ids: Vec<String> = unit_ids_as_strings(df, id_col)?;

        // ── 2. Períodos e unidades únicas ordenadas ────────────────────────
        let mut times: Vec<f64> = t_col.to_vec();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        times.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        let n_t = times.len();

        let time_idx: IndexMap<String, usize> = times
            .iter()
            .enumerate()
            .map(|(i, &t)| (float_key(t), i))
            .collect();

        let mut units: Vec<String> = unit_ids.clone();
        units.sort();
        units.dedup();
        let n_units = units.len();

        let unit_idx: IndexMap<&str, usize> = units
            .iter()
            .enumerate()
            .map(|(i, u)| (u.as_str(), i))
            .collect();

        let treated_j = *unit_idx.get(treated_unit).ok_or_else(|| {
            GreenersError::InvalidOperation(format!(
                "synth: unidade tratada '{treated_unit}' não encontrada em '{id_col}'"
            ))
        })?;

        // ── 3. Montar matriz de outcomes Y (n_t × n_units) ─────────────────
        let mut y_mat = vec![f64::NAN; n_t * n_units];

        for i in 0..n {
            let t_key = float_key(t_col[i]);
            let t_i = *time_idx.get(&t_key).ok_or_else(|| {
                GreenersError::InvalidOperation("synth: tempo inconsistente".into())
            })?;
            let u_i = *unit_idx.get(unit_ids[i].as_str()).ok_or_else(|| {
                GreenersError::InvalidOperation("synth: unidade inconsistente".into())
            })?;
            y_mat[t_i * n_units + u_i] = y_col[i];
        }

        // Checar NaN
        if y_mat.iter().any(|v| v.is_nan()) {
            return Err(GreenersError::InvalidOperation(
                "synth: painel desbalanceado ou missing — preencha antes de estimar".into(),
            ));
        }

        let y_matrix = Array2::from_shape_vec((n_t, n_units), y_mat)
            .map_err(|e| GreenersError::ShapeMismatch(e.to_string()))?;

        // ── 4. Índices pré/pós e doadores ────────────────────────────────────
        let t_pre = times.iter().filter(|&&t| t < t0).count();
        let t_post = n_t - t_pre;

        if t_pre < 2 {
            return Err(GreenersError::InvalidOperation(format!(
                "synth: apenas {t_pre} período(s) pré-tratamento (mínimo 2)"
            )));
        }

        let donor_idxs: Vec<usize> = (0..n_units).filter(|&j| j != treated_j).collect();
        let n_donors = donor_idxs.len();
        if n_donors == 0 {
            return Err(GreenersError::InvalidOperation(
                "synth: sem doadores no pool de controle".into(),
            ));
        }

        // Y1_pre (T_pre), Y0_pre (T_pre × J)
        let y1_pre: Array1<f64> = (0..t_pre)
            .map(|t| y_matrix[[t, treated_j]])
            .collect::<Vec<_>>()
            .into();
        let y0_pre: Array2<f64> = build_donor_matrix(&y_matrix, t_pre, &donor_idxs, 0);

        // ── 5. Montar Q (J×J) e c (J) para QP ───────────────────────────────
        // Objetivo: minimizar ||Y1_pre - Y0_pre w||² = w'Qw - 2c'w + cst
        let mut q = y0_pre.t().dot(&y0_pre);
        let mut c = y0_pre.t().dot(&y1_pre);

        // Covariáveis opcionais: augmentar Q e c com matching de médias pré-trat
        if let Some(cov_cols) = covariate_cols {
            for col_name in cov_cols {
                let x_col = df.get(col_name)?.to_owned();

                // Médias pré-tratamento: treated e cada doador
                let (x1_mean, x1_std) = pre_mean_std(&x_col, &t_col, &unit_ids, treated_unit, t0);
                let scale = x1_std.max(1e-10);

                let x0_means: Vec<f64> = donor_idxs
                    .iter()
                    .map(|&j| {
                        let uid = &units[j];
                        let (m, _) = pre_mean_std(&x_col, &t_col, &unit_ids, uid, t0);
                        m
                    })
                    .collect();

                // Contribuição: (x1_mean - X0_means' w)² / scale²
                // Q += (1/scale²) * x0_means x0_means'
                // c += (x1_mean / scale²) * x0_means
                for j in 0..n_donors {
                    for k in 0..n_donors {
                        q[[j, k]] += x0_means[j] * x0_means[k] / (scale * scale);
                    }
                    c[j] += x1_mean * x0_means[j] / (scale * scale);
                }
            }
        }

        // ── 6. Resolver QP no simplex ─────────────────────────────────────────
        let weights = simplex_qp(&q, &c);

        // ── 7. Série do controle sintético completa ───────────────────────────
        let y0_all = build_donor_matrix(&y_matrix, n_t, &donor_idxs, 0);
        let synthetic_series: Vec<f64> = y0_all.dot(&weights).to_vec();
        let actual_series: Vec<f64> = (0..n_t).map(|t| y_matrix[[t, treated_j]]).collect();

        // ── 8. RMSPE ──────────────────────────────────────────────────────────
        let rmspe_pre = {
            let sse: f64 = (0..t_pre)
                .map(|t| (actual_series[t] - synthetic_series[t]).powi(2))
                .sum();
            (sse / t_pre as f64).sqrt()
        };
        let rmspe_post = if t_post > 0 {
            let sse: f64 = (t_pre..n_t)
                .map(|t| (actual_series[t] - synthetic_series[t]).powi(2))
                .sum();
            Some((sse / t_post as f64).sqrt())
        } else {
            None
        };

        let donor_weights: Vec<(String, f64)> = donor_idxs
            .iter()
            .enumerate()
            .map(|(i, &j)| (units[j].clone(), weights[i]))
            .collect();

        Ok(SynthResult {
            weights: donor_weights,
            synthetic_series,
            actual_series,
            time_index: times,
            t0,
            rmspe_pre,
            rmspe_post,
            treated_unit: treated_unit.to_string(),
            outcome_name: outcome_col.to_string(),
            n_donors,
            t_pre,
            t_post,
        })
    }
}

// ── Internos ──────────────────────────────────────────────────────────────────

/// Representa f64 como chave de HashMap (via bits).
fn float_key(v: f64) -> String {
    format!("{}", v.to_bits())
}

/// Extrai IDs de unidade como Vec<String> (tenta string, int, float).
fn unit_ids_as_strings(df: &DataFrame, id_col: &str) -> Result<Vec<String>, GreenersError> {
    if let Ok(arr) = df.get_string(id_col) {
        return Ok(arr.to_vec());
    }
    if let Ok(arr) = df.get_int(id_col) {
        return Ok(arr.iter().map(|v| v.to_string()).collect());
    }
    if let Ok(arr) = df.get(id_col) {
        return Ok(arr.iter().map(|v| (*v as i64).to_string()).collect());
    }
    Err(GreenersError::VariableNotFound(id_col.to_string()))
}

/// Extrai submatriz de doadores: linhas [0..n_rows), colunas = donor_idxs.
fn build_donor_matrix(
    y: &Array2<f64>,
    n_rows: usize,
    donors: &[usize],
    _offset: usize,
) -> Array2<f64> {
    let n_donors = donors.len();
    let mut out = Array2::<f64>::zeros((n_rows, n_donors));
    for (j, &dj) in donors.iter().enumerate() {
        for t in 0..n_rows {
            out[[t, j]] = y[[t, dj]];
        }
    }
    out
}

/// Média e desvio padrão pré-tratamento de uma série para uma unidade específica.
fn pre_mean_std(
    x: &Array1<f64>,
    t_col: &Array1<f64>,
    unit_ids: &[String],
    target_unit: &str,
    t0: f64,
) -> (f64, f64) {
    let vals: Vec<f64> = x
        .iter()
        .enumerate()
        .filter(|(i, _)| unit_ids[*i] == target_unit && t_col[*i] < t0)
        .map(|(_, &v)| v)
        .collect();
    if vals.is_empty() {
        return (0.0, 1.0);
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let var = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
    (mean, var.sqrt())
}

// ── QP no simplex ─────────────────────────────────────────────────────────────

/// Minimiza ½ w' Q w - c' w sujeito a w ≥ 0, Σw = 1.
///
/// Usa gradiente projetado com passo fixo: α = 1 / λ_max(Q).
/// λ_max(Q) ≤ trace(Q) / 1 é um upper bound simples e seguro.
fn simplex_qp(q: &Array2<f64>, c: &Array1<f64>) -> Array1<f64> {
    let j = c.len();
    if j == 0 {
        return Array1::zeros(0);
    }

    // Inicializa uniforme
    let mut w: Vec<f64> = vec![1.0 / j as f64; j];

    // Passo: 1 / trace(Q) (conservador mas estável)
    let trace = q.diag().iter().sum::<f64>();
    let lr = if trace > 1e-15 { 1.0 / trace } else { 1e-4 };

    for _ in 0..20_000 {
        // gradiente: Qw - c
        let w_arr = Array1::from_vec(w.clone());
        let grad = q.dot(&w_arr) - c;

        let mut w_new: Vec<f64> = w
            .iter()
            .zip(grad.iter())
            .map(|(&wi, &gi)| wi - lr * gi)
            .collect();
        project_simplex(&mut w_new);

        // critério de convergência
        let diff: f64 = w_new
            .iter()
            .zip(w.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        w = w_new;
        if diff < 1e-14 {
            break;
        }
    }

    Array1::from_vec(w)
}

/// Projeção de v no simplex padrão: w ≥ 0, Σw = 1.
/// Algoritmo: Duchi et al. (2008), O(n log n).
fn project_simplex(v: &mut [f64]) {
    let mut u = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut cssv = 0.0_f64;
    let mut rho = 0usize;
    for (i, &ui) in u.iter().enumerate() {
        cssv += ui;
        if ui - (cssv - 1.0) / (i + 1) as f64 > 0.0 {
            rho = i;
        }
    }
    let theta = (u[..=rho].iter().sum::<f64>() - 1.0) / (rho + 1) as f64;
    for vi in v.iter_mut() {
        *vi = (*vi - theta).max(0.0);
    }
}
