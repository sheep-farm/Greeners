use crate::error::GreenersError;
use crate::linalg::LinalgInverse as _;
use ndarray::{s, Array1, Array2};
use std::fmt;

/// Resultado do estimador Arellano-Bond (Diff-GMM).
#[derive(Debug, Clone)]
pub struct ArellanoBondResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub sargan_stat: f64,
    pub sargan_pvalue: f64,
    pub sargan_df: usize,
    pub n_obs: usize,        // observações efetivas (após FD)
    pub n_entities: usize,
    pub t_bar: f64,          // média de T por entidade
    pub n_instruments: usize,
    pub max_lags: usize,
    pub step: usize,
    pub m1_stat: f64,
    pub m1_pval: f64,
    pub m2_stat: f64,
    pub m2_pval: f64,
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for ArellanoBondResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let step_label = if self.step == 2 { "Two-Step" } else { "One-Step" };
        writeln!(f, "\n{:=^78}", format!(" Arellano-Bond Diff-GMM ({step_label}) "))?;
        writeln!(f, "{:<24} {:>10} || {:<20} {:>12}",
            "Observations:", self.n_obs, "Entities:", self.n_entities)?;
        writeln!(f, "{:<24} {:>10.2} || {:<20} {:>12}",
            "Avg. T:", self.t_bar, "Instruments:", self.n_instruments)?;
        writeln!(f, "{:<24} {:>10} || {:<20} {:>12}",
            "Lags used:", self.max_lags, "Sargan df:", self.sargan_df)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "{:<14} | {:>10} | {:>10} | {:>8} | {:>8}",
            "Variable", "Coef", "Std Err", "z", "P>|z|")?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.params.len() {
            let name = self.variable_names
                .as_ref()
                .and_then(|n| n.get(i).cloned())
                .unwrap_or_else(|| if i == 0 { "LD.y".into() } else { format!("Δx{}", i) });
            writeln!(f, "{:<14} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3}",
                name, self.params[i], self.std_errors[i],
                self.t_values[i], self.p_values[i])?;
        }

        writeln!(f, "{:-^78}", "")?;
        writeln!(f, "\n── Sargan Test (H₀: instrumentos válidos)")?;
        if self.sargan_df == 0 {
            writeln!(f, "   Modelo exatamente identificado — sem teste de sobreidentificação")?;
        } else {
            let sig = if self.sargan_pvalue < 0.01 { "***" }
                      else if self.sargan_pvalue < 0.05 { "**" }
                      else if self.sargan_pvalue < 0.10 { "*" } else { "" };
            writeln!(f, "   χ²({}) = {:.4}   p = {:.4}  {}",
                self.sargan_df, self.sargan_stat, self.sargan_pvalue, sig)?;
            if self.sargan_pvalue < 0.05 {
                writeln!(f, "   ⚠  Rejeita H₀ — considere reduzir lags ou revisar instrumentos")?;
            }
        }

        writeln!(f, "\n── Arellano-Bond Autocorrelation Tests")?;
        let sig_m = |p: f64| if p < 0.01 { "***" } else if p < 0.05 { "**" }
                             else if p < 0.10 { "*" } else { "" };
        writeln!(f, "   m1: z = {:>8.4}   p = {:.4}  {}   (deve rejeitar — AR(1) esperado em FD)",
            self.m1_stat, self.m1_pval, sig_m(self.m1_pval))?;
        writeln!(f, "   m2: z = {:>8.4}   p = {:.4}  {}   (não deve rejeitar — valida instrumentos)",
            self.m2_stat, self.m2_pval, sig_m(self.m2_pval))?;
        if self.m2_pval < 0.05 {
            writeln!(f, "   ⚠  m2 rejeita H₀ — AR(2) detectado; instrumentos y_{{t-2}} podem ser inválidos")?;
        }

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "   *** p<0.01  ** p<0.05  * p<0.10   |   SE robustos (sandwich)")?;
        writeln!(f, "{:=^78}", "")
    }
}

pub struct ArellanoBond;

impl ArellanoBond {
    /// Estima o modelo dinâmico de painel via Diff-GMM (Arellano-Bond 1991).
    ///
    /// Modelo: y_it = ρ y_{i,t-1} + X_it'β + α_i + ε_it
    ///
    /// Método:
    ///   1. Primeira diferença para eliminar α_i
    ///   2. Instrumenta Δy_{i,t-1} com lags de nível y_{i,t-2}, ..., y_{i,t-max_lags-1}
    ///      (matriz de instrumentos collapsed — max_lags colunas por lag)
    ///   3. GMM em 1 passo (com matriz H) ou 2 passos (peso ótimo)
    ///   4. SE robustos sandwich em 1 passo; SE GMM em 2 passos
    ///   5. Teste de Sargan + m1/m2
    ///
    /// Argumentos:
    ///   y           — variável dependente (níveis)
    ///   x           — regressores estritamente exógenos (níveis, inclui const)
    ///   entity_ids  — ID de entidade por observação
    ///   time_ids    — ID de tempo por observação (inteiros)
    ///   max_lags    — número máximo de lags de y como instrumentos (default 2)
    ///   two_step    — se true, estima em 2 passos com peso ótimo
    ///   variable_names — nomes dos regressores (da fórmula)
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        entity_ids: &[i64],
        time_ids: &[i64],
        max_lags: usize,
        two_step: bool,
        variable_names: Option<Vec<String>>,
    ) -> Result<ArellanoBondResult, GreenersError> {
        use statrs::distribution::{ContinuousCDF, Normal};

        let n_total = y.len();
        let k_x = x.ncols();

        if max_lags < 1 {
            return Err(GreenersError::InvalidOperation(
                "max_lags deve ser >= 1".into(),
            ));
        }
        if entity_ids.len() != n_total || time_ids.len() != n_total {
            return Err(GreenersError::ShapeMismatch("IDs mismatch".into()));
        }

        // 1. Ordenar por entidade, depois por tempo
        let mut ord: Vec<usize> = (0..n_total).collect();
        ord.sort_by_key(|&i| (entity_ids[i], time_ids[i]));

        let ys: Vec<f64> = ord.iter().map(|&i| y[i]).collect();
        let xs: Vec<Vec<f64>> = ord.iter()
            .map(|&i| (0..k_x).map(|c| x[[i, c]]).collect())
            .collect();
        let ids: Vec<i64> = ord.iter().map(|&i| entity_ids[i]).collect();

        // 2. Agrupar por entidade
        let mut entity_slices: Vec<std::ops::Range<usize>> = Vec::new();
        let mut start = 0;
        while start < n_total {
            let eid = ids[start];
            let end = ids[start..].iter().position(|&id| id != eid)
                .map(|p| start + p)
                .unwrap_or(n_total);
            entity_slices.push(start..end);
            start = end;
        }
        let n_entities = entity_slices.len();

        // 3. Construir dados de primeira diferença + instrumentos
        //    Equações FD por entidade i: j=2,...,T_i-1  (T_i-2 equações, precisa T >= 3)
        //    W = [ΔYlag | ΔX_active]  (regressores)
        //    Z = [Y_lags_collapsed | ΔX_active]  (instrumentos)
        let mut dy_vec: Vec<f64> = Vec::new();      // Δy_jt
        let mut dyl_vec: Vec<f64> = Vec::new();     // Δy_{j,t-1} (endógeno)
        let mut dx_rows: Vec<Vec<f64>> = Vec::new(); // ΔX rows
        let mut zinst_rows: Vec<Vec<f64>> = Vec::new(); // Y instrument rows
        let mut entity_fd_count: Vec<usize> = Vec::new();

        for slice in &entity_slices {
            let t_i = slice.len();
            if t_i < 3 {
                entity_fd_count.push(0);
                continue;
            }
            let idx: Vec<usize> = slice.clone().collect();
            let mut count = 0;

            for j in 2..t_i {
                // Δy[j] = y[j] - y[j-1]
                dy_vec.push(ys[idx[j]] - ys[idx[j - 1]]);
                // ΔYlag[j] = y[j-1] - y[j-2]
                dyl_vec.push(ys[idx[j - 1]] - ys[idx[j - 2]]);
                // ΔX[j]
                dx_rows.push((0..k_x).map(|c| xs[idx[j]][c] - xs[idx[j - 1]][c]).collect());
                // Y instruments: lag l+2 = y[j-(l+2)] for l=0,...,max_lags-1
                zinst_rows.push(
                    (0..max_lags)
                        .map(|l| {
                            let lag = l + 2;
                            if j >= lag { ys[idx[j - lag]] } else { 0.0 }
                        })
                        .collect(),
                );
                count += 1;
            }
            entity_fd_count.push(count);
        }

        let n_eff = dy_vec.len();
        if n_eff == 0 {
            return Err(GreenersError::InvalidOperation(
                "Nenhuma equação FD efetiva — precisa T ≥ 3 por entidade".into(),
            ));
        }

        let dy = Array1::from_vec(dy_vec);

        // 4. Identificar colunas ativas de ΔX (variância > 0 após FD)
        let active_x: Vec<usize> = (0..k_x)
            .filter(|&c| dx_rows.iter().any(|row| row[c].abs() > 1e-12))
            .collect();
        let k_dx = active_x.len();
        let k_reg = 1 + k_dx; // ΔYlag + ΔX_active
        let n_inst = max_lags + k_dx;

        if n_inst < k_reg {
            return Err(GreenersError::InvalidOperation(format!(
                "Sub-identificado: {} instrumentos < {} regressores. Aumente max_lags.",
                n_inst, k_reg
            )));
        }

        // 5. Construir matrizes W e Z
        let mut w_mat = Array2::<f64>::zeros((n_eff, k_reg));
        let mut z_mat = Array2::<f64>::zeros((n_eff, n_inst));
        for i in 0..n_eff {
            w_mat[[i, 0]] = dyl_vec[i];
            for (nc, &oc) in active_x.iter().enumerate() {
                w_mat[[i, 1 + nc]] = dx_rows[i][oc];
                z_mat[[i, max_lags + nc]] = dx_rows[i][oc];
            }
            for l in 0..max_lags {
                z_mat[[i, l]] = zinst_rows[i][l];
            }
        }

        // 6. Matriz de peso A_1 = (Z' H Z)^{-1}
        //    H = block_diag(H_1,...,H_N)  H_i = tridiag(2,-1,-1)
        let mut zthz = Array2::<f64>::zeros((n_inst, n_inst));
        let mut rptr = 0usize;
        for &fc in &entity_fd_count {
            if fc == 0 { continue; }
            let zi = z_mat.slice(s![rptr..rptr + fc, ..]).to_owned();
            let mut hi = Array2::<f64>::zeros((fc, fc));
            for s in 0..fc {
                hi[[s, s]] = 2.0;
                if s > 0 { hi[[s, s - 1]] = -1.0; }
                if s < fc - 1 { hi[[s, s + 1]] = -1.0; }
            }
            zthz = zthz + zi.t().dot(&hi).dot(&zi);
            rptr += fc;
        }
        let a1 = zthz.inv().map_err(|_| GreenersError::SingularMatrix)?;

        // 7. Estimador GMM 1 passo
        let wtz = w_mat.t().dot(&z_mat);   // (k_reg × n_inst)
        let zty = z_mat.t().dot(&dy);       // (n_inst,)
        let wtz_a1 = wtz.dot(&a1);          // (k_reg × n_inst)
        let lhs1 = wtz_a1.dot(&wtz.t());    // (k_reg × k_reg)
        let lhs1_inv = lhs1.inv().map_err(|_| GreenersError::SingularMatrix)?;
        let params1 = lhs1_inv.dot(&wtz_a1.dot(&zty));
        let resid1 = &dy - &w_mat.dot(&params1);

        // 8. Variância sandwich robusta (1 passo)
        let mut sigma = Array2::<f64>::zeros((n_inst, n_inst));
        rptr = 0;
        for &fc in &entity_fd_count {
            if fc == 0 { continue; }
            let zi = z_mat.slice(s![rptr..rptr + fc, ..]).to_owned();
            let ui = resid1.slice(s![rptr..rptr + fc]).to_owned();
            let zui = zi.t().dot(&ui); // (n_inst,)
            for r in 0..n_inst {
                for c in 0..n_inst {
                    sigma[[r, c]] += zui[r] * zui[c];
                }
            }
            rptr += fc;
        }
        let meat1 = wtz_a1.dot(&sigma).dot(&a1).dot(&wtz.t());
        let var1 = lhs1_inv.dot(&meat1).dot(&lhs1_inv);
        let se1: Array1<f64> = var1.diag().mapv(|v| v.max(0.0).sqrt());

        // 9. 2 passos (opcional)
        let (params, std_errors, step_used) = if two_step {
            let a2 = sigma.inv().map_err(|_| GreenersError::SingularMatrix)?;
            let wtz_a2 = wtz.dot(&a2);
            let lhs2 = wtz_a2.dot(&wtz.t());
            let lhs2_inv = lhs2.inv().map_err(|_| GreenersError::SingularMatrix)?;
            let params2 = lhs2_inv.dot(&wtz_a2.dot(&zty));
            let se2: Array1<f64> = lhs2_inv.diag().mapv(|v| v.max(0.0).sqrt());
            (params2, se2, 2usize)
        } else {
            (params1.clone(), se1, 1usize)
        };

        // 10. Estatísticas t e p (assintoticamente normais)
        let normal = Normal::new(0.0, 1.0).unwrap();
        let t_values = &params / &std_errors;
        let p_values = t_values.mapv(|t| 2.0 * (1.0 - normal.cdf(t.abs())));

        // 11. Teste de Sargan (1 passo, resíduos de 1 passo)
        let sargan_df = n_inst.saturating_sub(k_reg);
        let (sargan_stat, sargan_pvalue) = if sargan_df > 0 {
            use statrs::distribution::ChiSquared;
            let zu1 = z_mat.t().dot(&resid1);
            let s = zu1.dot(&a1.dot(&zu1)) * (n_eff as f64 / resid1.dot(&resid1));
            let chi2 = ChiSquared::new(sargan_df as f64)
                .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
            (s, 1.0 - chi2.cdf(s.max(0.0)))
        } else {
            (0.0, 1.0)
        };

        // 12. Testes m1 / m2 de autocorrelação serial
        let (m1_stat, m1_pval, m2_stat, m2_pval) =
            compute_m_stats(&resid1, &entity_fd_count, &normal)?;

        // 13. Nomes das variáveis
        let vnames = variable_names.map(|vn| {
            let non_const: Vec<&str> = vn.iter()
                .filter(|n| n.as_str() != "const")
                .map(|s| s.as_str())
                .collect();
            let mut names = vec!["LD.y".to_string()];
            for (ni, &oc) in active_x.iter().enumerate() {
                // oc é índice na matriz x original (com const em 0)
                // se há const, non_const[oc-1] é o nome; senão non_const[oc]
                let nm = non_const
                    .get(oc.saturating_sub(if vn.contains(&"const".to_string()) { 1 } else { 0 }))
                    .copied()
                    .unwrap_or("x");
                let _ = ni; // evita warning
                names.push(format!("Δ{nm}"));
            }
            names
        });

        Ok(ArellanoBondResult {
            params,
            std_errors,
            t_values,
            p_values,
            sargan_stat,
            sargan_pvalue,
            sargan_df,
            n_obs: n_eff,
            n_entities,
            t_bar: n_eff as f64 / n_entities as f64,
            n_instruments: n_inst,
            max_lags,
            step: step_used,
            m1_stat,
            m1_pval,
            m2_stat,
            m2_pval,
            variable_names: vnames,
        })
    }
}

/// Calcula as estatísticas m1 e m2 de autocorrelação serial de Arellano-Bond.
fn compute_m_stats(
    fd_resid: &Array1<f64>,
    entity_fd_count: &[usize],
    normal: &statrs::distribution::Normal,
) -> Result<(f64, f64, f64, f64), GreenersError> {
    use statrs::distribution::ContinuousCDF;

    let m_stat = |p: usize| -> Option<(f64, f64)> {
        let mut c_p = 0.0f64;
        let mut v_p = 0.0f64;
        let mut rptr = 0usize;
        for &fc in entity_fd_count {
            if fc > p {
                let entity_sum: f64 = (p..fc)
                    .map(|t| fd_resid[rptr + t] * fd_resid[rptr + t - p])
                    .sum();
                c_p += entity_sum;
                v_p += entity_sum * entity_sum;
            }
            rptr += fc;
        }
        if v_p < 1e-20 { return None; }
        let stat = c_p / v_p.sqrt();
        let pval = 2.0 * (1.0 - normal.cdf(stat.abs()));
        Some((stat, pval))
    };

    let (m1, p1) = m_stat(1).ok_or_else(|| {
        GreenersError::InvalidOperation(
            "Dados insuficientes para m1 (precisa T ≥ 4 total)".into(),
        )
    })?;
    let (m2, p2) = m_stat(2).ok_or_else(|| {
        GreenersError::InvalidOperation(
            "Dados insuficientes para m2 (precisa T ≥ 5 total)".into(),
        )
    })?;

    Ok((m1, p1, m2, p2))
}

// ─────────────────────────────────────────────────────────────────────────────
// System GMM — Blundell-Bond (1998)
//
// Empilha equações em 1ª diferença (Arellano-Bond) com equações em nível,
// instrumentando as equações em nível com Δy_{t-1} e ΔX_{t-1}.
// Condição de momento extra: E[Δy_{it} α_i] = 0 (estacionariedade do processo).
// ─────────────────────────────────────────────────────────────────────────────

/// Resultado do estimador System GMM (Blundell-Bond 1998).
#[derive(Debug, Clone)]
pub struct SystemGmmResult {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub sargan_stat: f64,
    pub sargan_pvalue: f64,
    pub sargan_df: usize,
    pub n_obs_fd: usize,      // equações em 1ª diferença
    pub n_obs_lev: usize,     // equações em nível
    pub n_entities: usize,
    pub n_instruments: usize,
    pub max_lags: usize,
    pub step: usize,
    pub m1_stat: f64,
    pub m1_pval: f64,
    pub m2_stat: f64,
    pub m2_pval: f64,
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for SystemGmmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let step_label = if self.step == 2 { "Two-Step" } else { "One-Step" };
        writeln!(f, "\n{:=^78}", format!(" System GMM — Blundell-Bond 1998 ({step_label}) "))?;
        writeln!(f, "{:<24} {:>10} || {:<20} {:>12}",
            "Obs FD:", self.n_obs_fd, "Obs nível:", self.n_obs_lev)?;
        writeln!(f, "{:<24} {:>10} || {:<20} {:>12}",
            "Entidades:", self.n_entities, "Instrumentos:", self.n_instruments)?;
        writeln!(f, "{:<24} {:>10} || {:<20} {:>12}",
            "Lags (FD):", self.max_lags, "Sargan df:", self.sargan_df)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "{:<14} | {:>10} | {:>10} | {:>8} | {:>8}",
            "Variável", "Coef", "Std Err", "z", "P>|z|")?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..self.params.len() {
            let name = self.variable_names
                .as_ref()
                .and_then(|n| n.get(i).cloned())
                .unwrap_or_else(|| if i == 0 { "L.y".into() } else { format!("x{}", i) });
            writeln!(f, "{:<14} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3}",
                name, self.params[i], self.std_errors[i],
                self.t_values[i], self.p_values[i])?;
        }

        writeln!(f, "{:-^78}", "")?;
        writeln!(f, "\n── Sargan/Hansen (H₀: instrumentos válidos)")?;
        if self.sargan_df == 0 {
            writeln!(f, "   Exatamente identificado — sem teste de sobreidentificação")?;
        } else {
            let sig = if self.sargan_pvalue < 0.01 { "***" }
                      else if self.sargan_pvalue < 0.05 { "**" }
                      else if self.sargan_pvalue < 0.10 { "*" } else { "" };
            writeln!(f, "   χ²({}) = {:.4}   p = {:.4}  {}",
                self.sargan_df, self.sargan_stat, self.sargan_pvalue, sig)?;
            if self.sargan_pvalue < 0.05 {
                writeln!(f, "   ⚠  Rejeita H₀ — verifique condição de estacionariedade")?;
            }
        }

        writeln!(f, "\n── Arellano-Bond (resíduos FD)")?;
        let sig_m = |p: f64| if p < 0.01 { "***" } else if p < 0.05 { "**" }
                             else if p < 0.10 { "*" } else { "" };
        writeln!(f, "   m1: z = {:>8.4}   p = {:.4}  {}",
            self.m1_stat, self.m1_pval, sig_m(self.m1_pval))?;
        writeln!(f, "   m2: z = {:>8.4}   p = {:.4}  {}",
            self.m2_stat, self.m2_pval, sig_m(self.m2_pval))?;
        if self.m2_pval < 0.05 {
            writeln!(f, "   ⚠  m2 rejeita H₀ — AR(2) nos erros; rever lags")?;
        }

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "   *** p<0.01  ** p<0.05  * p<0.10   |   SE robustos (sandwich)")?;
        writeln!(f, "{:=^78}", "")
    }
}

pub struct SystemGmm;

impl SystemGmm {
    /// Estima o modelo dinâmico de painel via System GMM (Blundell-Bond 1998).
    ///
    /// Empilha:
    ///   1. Equações FD instrumentadas com lags de nível (Arellano-Bond)
    ///   2. Equações em nível instrumentadas com Δy_{t-1} e ΔX_{t-1}
    ///
    /// Condição extra: E[Δy_{it} α_i] = 0 — processo próximo a raiz unitária.
    /// Preferir Diff-GMM se a condição de estacionariedade não for plausível.
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        entity_ids: &[i64],
        time_ids: &[i64],
        max_lags: usize,
        two_step: bool,
        variable_names: Option<Vec<String>>,
    ) -> Result<SystemGmmResult, GreenersError> {
        use statrs::distribution::{ContinuousCDF, Normal};

        let n_total = y.len();
        let k_x = x.ncols();

        if max_lags < 1 {
            return Err(GreenersError::InvalidOperation("max_lags deve ser >= 1".into()));
        }

        // 1. Ordenar por entidade, depois tempo
        let mut ord: Vec<usize> = (0..n_total).collect();
        ord.sort_by_key(|&i| (entity_ids[i], time_ids[i]));

        let ys: Vec<f64> = ord.iter().map(|&i| y[i]).collect();
        let xs: Vec<Vec<f64>> = ord.iter()
            .map(|&i| (0..k_x).map(|c| x[[i, c]]).collect())
            .collect();
        let ids: Vec<i64> = ord.iter().map(|&i| entity_ids[i]).collect();

        // 2. Agrupar por entidade
        let mut entity_slices: Vec<std::ops::Range<usize>> = Vec::new();
        let mut start = 0;
        while start < n_total {
            let eid = ids[start];
            let end = ids[start..].iter().position(|&id| id != eid)
                .map(|p| start + p).unwrap_or(n_total);
            entity_slices.push(start..end);
            start = end;
        }
        let n_entities = entity_slices.len();

        // 3. Construir dados FD + nível
        //    Equações FD:  j = 2,...,T-1  (T-2 por entidade)
        //    Equações nív: j = 2,...,T-1  (T-2 por entidade — mesma janela)
        //      Instrumento nível: Δy_{j-1} = y[j-1] - y[j-2]  (sempre disponível para j>=2)

        let mut dy_vec:   Vec<f64> = Vec::new();  // Δy_jt   (FD dep)
        let mut dyl_vec:  Vec<f64> = Vec::new();  // Δy_{j,t-1} (FD endog)
        let mut dx_rows:  Vec<Vec<f64>> = Vec::new(); // ΔX (FD exog)
        let mut zinst_fd: Vec<Vec<f64>> = Vec::new(); // instrumentos FD (lags nível)

        let mut y_lev:    Vec<f64> = Vec::new();  // y_jt   (nível dep)
        let mut yl_lev:   Vec<f64> = Vec::new();  // y_{j-1} (nível endog)
        let mut x_lev:    Vec<Vec<f64>> = Vec::new(); // X_jt nível (exog)
        let mut zinst_lv: Vec<Vec<f64>> = Vec::new(); // instrumentos nível

        let mut entity_fd_count:  Vec<usize> = Vec::new();
        let mut entity_lev_count: Vec<usize> = Vec::new();

        for slice in &entity_slices {
            let t_i = slice.len();
            if t_i < 3 {
                entity_fd_count.push(0);
                entity_lev_count.push(0);
                continue;
            }
            let idx: Vec<usize> = slice.clone().collect();

            for j in 2..t_i {
                // ── FD equation ──────────────────────────────────────────────
                dy_vec.push(ys[idx[j]] - ys[idx[j-1]]);
                dyl_vec.push(ys[idx[j-1]] - ys[idx[j-2]]);
                dx_rows.push((0..k_x).map(|c| xs[idx[j]][c] - xs[idx[j-1]][c]).collect());
                zinst_fd.push((0..max_lags).map(|l| {
                    let lag = l + 2;
                    if j >= lag { ys[idx[j-lag]] } else { 0.0 }
                }).collect());

                // ── Levels equation ───────────────────────────────────────────
                y_lev.push(ys[idx[j]]);
                yl_lev.push(ys[idx[j-1]]);
                x_lev.push((0..k_x).map(|c| xs[idx[j]][c]).collect());
                // Instrumento: Δy_{j-1} sempre disponível (j>=2), ΔX_{j-1}
                let dy_lag_inst = ys[idx[j-1]] - ys[idx[j-2]];
                let mut zinst_row = vec![dy_lag_inst];
                for c in 0..k_x {
                    zinst_row.push(xs[idx[j-1]][c] - xs[idx[j-2]][c]);
                }
                zinst_lv.push(zinst_row);
            }
            entity_fd_count.push(t_i - 2);
            entity_lev_count.push(t_i - 2);
        }

        let n_fd  = dy_vec.len();
        let n_lev = y_lev.len();
        let n_sys = n_fd + n_lev;

        if n_sys == 0 {
            return Err(GreenersError::InvalidOperation(
                "Nenhuma equação efetiva — precisa T ≥ 3 por entidade".into(),
            ));
        }

        // 4. Colunas ativas de ΔX
        let active_x: Vec<usize> = (0..k_x).filter(|&c| {
            dx_rows.iter().any(|r| r[c].abs() > 1e-12)
        }).collect();
        let k_dx  = active_x.len();
        let k_reg = 1 + k_dx;               // [y_{t-1} | X_active]

        let n_inst_fd  = max_lags + k_dx;   // FD block
        let n_inst_lv  = 1 + k_dx;          // levels block (Δy_{t-1} + ΔX_{t-1})
        let n_inst_sys = n_inst_fd + n_inst_lv;

        if n_inst_sys < k_reg {
            return Err(GreenersError::InvalidOperation(format!(
                "Sub-identificado: {} instrumentos < {} regressores.",
                n_inst_sys, k_reg
            )));
        }

        // 5. Montar W_sys e Z_sys
        //    W_sys = [W_fd ; W_lev]
        //    Z_sys = [[Z_fd, 0] ; [0, Z_lev]]  (block diagonal)
        let mut w_sys = Array2::<f64>::zeros((n_sys, k_reg));
        let mut z_sys = Array2::<f64>::zeros((n_sys, n_inst_sys));

        for i in 0..n_fd {
            w_sys[[i, 0]] = dyl_vec[i];
            for (nc, &oc) in active_x.iter().enumerate() {
                w_sys[[i, 1 + nc]] = dx_rows[i][oc];
                z_sys[[i, max_lags + nc]] = dx_rows[i][oc];
            }
            for l in 0..max_lags { z_sys[[i, l]] = zinst_fd[i][l]; }
        }
        for i in 0..n_lev {
            let row = n_fd + i;
            w_sys[[row, 0]] = yl_lev[i];
            for (nc, &oc) in active_x.iter().enumerate() {
                w_sys[[row, 1 + nc]] = x_lev[i][oc];
                z_sys[[row, n_inst_fd + 1 + nc]] = zinst_lv[i][1 + nc];
            }
            z_sys[[row, n_inst_fd]] = zinst_lv[i][0]; // Δy_{t-1}
        }

        // 6. Peso inicial A_1 = (Z' H_sys Z)^{-1}
        //    H_sys_i = block_diag(H_fd_i, I_lev_i)
        let mut zthz = Array2::<f64>::zeros((n_inst_sys, n_inst_sys));
        let mut rptr_fd  = 0usize;
        let mut rptr_lev = n_fd;
        for (ei, (&fc_fd, &fc_lev)) in entity_fd_count.iter().zip(&entity_lev_count).enumerate() {
            let _ = ei;
            if fc_fd == 0 { continue; }

            // FD block con H_i
            let zfd = z_sys.slice(s![rptr_fd..rptr_fd + fc_fd, ..]).to_owned();
            let mut h_fd = Array2::<f64>::zeros((fc_fd, fc_fd));
            for s in 0..fc_fd {
                h_fd[[s, s]] = 2.0;
                if s > 0 { h_fd[[s, s-1]] = -1.0; }
                if s < fc_fd-1 { h_fd[[s, s+1]] = -1.0; }
            }
            zthz = zthz + zfd.t().dot(&h_fd).dot(&zfd);

            // Levels block con I (identidade)
            let zlv = z_sys.slice(s![rptr_lev..rptr_lev + fc_lev, ..]).to_owned();
            zthz = zthz + zlv.t().dot(&zlv);

            rptr_fd  += fc_fd;
            rptr_lev += fc_lev;
        }
        let a1 = zthz.inv().map_err(|_| GreenersError::SingularMatrix)?;

        // 7. Estimador GMM 1 passo
        let dy_sys: Array1<f64> = dy_vec.iter().chain(y_lev.iter()).copied().collect();
        let wtz     = w_sys.t().dot(&z_sys);
        let zty     = z_sys.t().dot(&dy_sys);
        let wtz_a1  = wtz.dot(&a1);
        let lhs1    = wtz_a1.dot(&wtz.t());
        let lhs1_inv = lhs1.inv().map_err(|_| GreenersError::SingularMatrix)?;
        let params1  = lhs1_inv.dot(&wtz_a1.dot(&zty));
        let resid1   = &dy_sys - &w_sys.dot(&params1);

        // 8. Variância sandwich robusta
        let mut sigma = Array2::<f64>::zeros((n_inst_sys, n_inst_sys));
        let mut rfd  = 0usize;
        let mut rlev = n_fd;
        for (&fc_fd, &fc_lev) in entity_fd_count.iter().zip(&entity_lev_count) {
            if fc_fd == 0 { continue; }
            let fc = fc_fd + fc_lev;
            // Concatena linhas da entidade (FD seguidas de nível) para cálculo do sandwich
            let mut z_ent = Array2::<f64>::zeros((fc, n_inst_sys));
            let mut u_ent = Array1::<f64>::zeros(fc);
            for r in 0..fc_fd {
                z_ent.row_mut(r).assign(&z_sys.row(rfd + r));
                u_ent[r] = resid1[rfd + r];
            }
            for r in 0..fc_lev {
                z_ent.row_mut(fc_fd + r).assign(&z_sys.row(rlev + r));
                u_ent[fc_fd + r] = resid1[rlev + r];
            }
            let zu = z_ent.t().dot(&u_ent);
            for a in 0..n_inst_sys {
                for b in 0..n_inst_sys {
                    sigma[[a, b]] += zu[a] * zu[b];
                }
            }
            rfd  += fc_fd;
            rlev += fc_lev;
        }
        let meat1 = wtz_a1.dot(&sigma).dot(&a1).dot(&wtz.t());
        let var1  = lhs1_inv.dot(&meat1).dot(&lhs1_inv);
        let se1: Array1<f64> = var1.diag().mapv(|v| v.max(0.0).sqrt());

        // 9. 2 passos
        let (params, std_errors, step_used) = if two_step {
            let a2 = sigma.inv().map_err(|_| GreenersError::SingularMatrix)?;
            let wtz_a2   = wtz.dot(&a2);
            let lhs2     = wtz_a2.dot(&wtz.t());
            let lhs2_inv = lhs2.inv().map_err(|_| GreenersError::SingularMatrix)?;
            let params2  = lhs2_inv.dot(&wtz_a2.dot(&zty));
            let se2: Array1<f64> = lhs2_inv.diag().mapv(|v| v.max(0.0).sqrt());
            (params2, se2, 2usize)
        } else {
            (params1.clone(), se1, 1usize)
        };

        // 10. Estatísticas z e p
        let normal = Normal::new(0.0, 1.0).unwrap();
        let t_values = &params / &std_errors;
        let p_values = t_values.mapv(|t| 2.0 * (1.0 - normal.cdf(t.abs())));

        // 11. Sargan
        let sargan_df = n_inst_sys.saturating_sub(k_reg);
        let (sargan_stat, sargan_pvalue) = if sargan_df > 0 {
            use statrs::distribution::ChiSquared;
            let zu1 = z_sys.t().dot(&resid1);
            let s = zu1.dot(&a1.dot(&zu1)) * (n_sys as f64 / resid1.dot(&resid1));
            let chi2 = ChiSquared::new(sargan_df as f64)
                .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
            (s, 1.0 - chi2.cdf(s.max(0.0)))
        } else {
            (0.0, 1.0)
        };

        // 12. m1/m2 — apenas nos resíduos FD
        let fd_resid: Array1<f64> = resid1.slice(s![..n_fd]).to_owned();
        let (m1_stat, m1_pval, m2_stat, m2_pval) =
            compute_m_stats(&fd_resid, &entity_fd_count, &normal)?;

        // 13. Nomes das variáveis (interpretação em nível)
        let vnames = variable_names.map(|vn| {
            let non_const: Vec<&str> = vn.iter()
                .filter(|n| n.as_str() != "const")
                .map(|s| s.as_str())
                .collect();
            let mut names = vec!["L.y".to_string()];
            for (_, &oc) in active_x.iter().enumerate() {
                let offset = if vn.contains(&"const".to_string()) { 1 } else { 0 };
                let nm = non_const.get(oc.saturating_sub(offset)).copied().unwrap_or("x");
                names.push(nm.to_string());
            }
            names
        });

        Ok(SystemGmmResult {
            params,
            std_errors,
            t_values,
            p_values,
            sargan_stat,
            sargan_pvalue,
            sargan_df,
            n_obs_fd: n_fd,
            n_obs_lev: n_lev,
            n_entities,
            n_instruments: n_inst_sys,
            max_lags,
            step: step_used,
            m1_stat,
            m1_pval,
            m2_stat,
            m2_pval,
            variable_names: vnames,
        })
    }
}
