//! Panel VAR (PVAR) via GMM estimation.
//!
//! Holtz-Eakin, Newey & Rosen (1988). Estimates a VAR on panel data
//! where each cross-section unit is treated as an individual time
//! series. Uses lagged levels as instruments for first-differenced
//! equations (Arellano-Bond style GMM).
//!
//! y_{i,t} = A_1 * y_{i,t-1} + ... + A_p * y_{i,t-p} + mu_i + eps_{i,t}
//!
//! After first-differencing to remove fixed effects:
//! Delta y_{i,t} = A_1 * Delta y_{i,t-1} + ... + A_p * Delta y_{i,t-p} + Delta eps_{i,t}
//!
//! Instruments: y_{i,t-2}, y_{i,t-3}, ... (lagged levels)

use crate::linalg::LinalgInverse as _;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::ContinuousCDF;
use std::fmt;

/// Result of Panel VAR estimation.
#[derive(Debug)]
pub struct PanelVarResult {
    /// VAR coefficient matrix (k x (k*p)), each column = coefficients for one equation
    pub coeffs: Array2<f64>,
    /// Standard errors
    pub std_errors: Array2<f64>,
    /// t-values
    pub t_values: Array2<f64>,
    /// p-values
    pub p_values: Array2<f64>,
    /// J-test statistic (overidentifying restrictions)
    pub j_stat: f64,
    /// p-value for J-test
    pub j_p: f64,
    /// Number of instruments
    pub n_instruments: usize,
    /// Number of observations (after differencing + lags)
    pub n_obs: usize,
    /// Number of entities
    pub n_entities: usize,
    /// Number of variables
    pub n_vars: usize,
    /// VAR lag order
    pub lags: usize,
    /// Variable names
    pub var_names: Vec<String>,
}

impl fmt::Display for PanelVarResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Panel VAR (GMM) ")?;
        writeln!(f, "Holtz-Eakin, Newey & Rosen (1988)")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Entities:", self.n_entities)?;
        writeln!(f, "{:<20} {:>12}", "Variables:", self.n_vars)?;
        writeln!(f, "{:<20} {:>12}", "Lags:", self.lags)?;
        writeln!(f, "{:<20} {:>12}", "Instruments:", self.n_instruments)?;
        writeln!(f, "{:<20} {:>12.4}", "J-stat:", self.j_stat)?;
        writeln!(f, "{:<20} {:>12.4}", "J p-value:", self.j_p)?;

        let k = self.n_vars;
        let p = self.lags;
        for eq in 0..k {
            let eq_name = self
                .var_names
                .get(eq)
                .cloned()
                .unwrap_or_else(|| format!("y{}", eq));
            writeln!(f, "\n{:-^78}", format!(" Equation: {} ", eq_name))?;
            writeln!(
                f,
                "{:<14} {:>12} {:>12} {:>10} {:>10}",
                "Variable", "Coef.", "Std.Err.", "t", "P>|t|"
            )?;
            writeln!(f, "{:-^78}", "")?;
            for lag in 0..p {
                for j in 0..k {
                    let var_name = self
                        .var_names
                        .get(j)
                        .cloned()
                        .unwrap_or_else(|| format!("y{}", j));
                    let col = lag * k + j;
                    let label = format!("L{}.{}", lag + 1, var_name);
                    writeln!(
                        f,
                        "{:<14} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
                        label,
                        self.coeffs[(eq, col)],
                        self.std_errors[(eq, col)],
                        self.t_values[(eq, col)],
                        self.p_values[(eq, col)]
                    )?;
                }
            }
        }
        write!(f, "{:=^78}", "")
    }
}

pub struct PanelVAR;

impl PanelVAR {
    /// Estimate Panel VAR via GMM (Arellano-Bond style).
    ///
    /// # Arguments
    /// * `y` - Data matrix (n x k), stacked by entity then period
    /// * `entity_ids` - Entity identifier (n)
    /// * `lags` - VAR lag order
    /// * `var_names` - Optional variable names
    pub fn fit(
        y: &Array2<f64>,
        entity_ids: &[i64],
        lags: usize,
        var_names: Option<Vec<String>>,
    ) -> Result<PanelVarResult, GreenersError> {
        let n = y.nrows();
        let k = y.ncols();
        if entity_ids.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "PVAR: dimension mismatch".into(),
            ));
        }
        if lags == 0 {
            return Err(GreenersError::InvalidOperation(
                "PVAR: lags must be >= 1".into(),
            ));
        }

        let names = var_names.unwrap_or_else(|| (0..k).map(|i| format!("y{}", i)).collect());

        // Identify entities and periods
        let mut unique_ids: Vec<i64> = entity_ids
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        unique_ids.sort();
        let n_entities = unique_ids.len();

        // Group data by entity
        let mut entity_data: std::collections::HashMap<i64, Array2<f64>> =
            std::collections::HashMap::new();
        for &id in &unique_ids {
            let rows: Vec<usize> = (0..n).filter(|&i| entity_ids[i] == id).collect();
            let t_e = rows.len();
            let mut data = Array2::zeros((t_e, k));
            for (t_i, &r) in rows.iter().enumerate() {
                for j in 0..k {
                    data[(t_i, j)] = y[(r, j)];
                }
            }
            entity_data.insert(id, data);
        }

        // Build GMM moment conditions
        // For each entity, for each period t >= lags+1:
        //   Differenced equation: Delta y_{i,t} = sum A_p * Delta y_{i,t-p} + error
        //   Instruments: y_{i,t-p-1}, ..., y_{i,0} (lagged levels)
        let n_reg = k * lags; // regressors per equation

        // Collect all differenced observations and instruments
        let mut all_dy: Vec<Array1<f64>> = Vec::new();
        let mut all_dx: Vec<Array1<f64>> = Vec::new();
        let mut all_instr: Vec<Array1<f64>> = Vec::new();

        for &id in &unique_ids {
            let data = &entity_data[&id];
            let t_e = data.nrows();
            if t_e < lags + 2 {
                continue;
            }

            // First differences
            let mut dy_data = Array2::zeros((t_e - 1, k));
            for t_i in 0..t_e - 1 {
                for j in 0..k {
                    dy_data[(t_i, j)] = data[(t_i + 1, j)] - data[(t_i, j)];
                }
            }

            // For each usable period
            for t_i in (lags)..t_e - 1 {
                // Dependent: Delta y_{i,t}
                let dy_t = dy_data.row(t_i).to_owned();
                all_dy.push(dy_t);

                // Regressors: Delta y_{i,t-1}, ..., Delta y_{i,t-p}
                let mut dx_t = Array1::zeros(n_reg);
                for p in 0..lags {
                    for j in 0..k {
                        dx_t[p * k + j] = dy_data[(t_i - 1 - p, j)];
                    }
                }
                all_dx.push(dx_t);

                // Instruments: y_{i,t-p-1}, y_{i,t-p-2}, ..., y_{i,0} (levels)
                let mut instr: Vec<f64> = Vec::new();
                let max_instr_period = t_i.saturating_sub(lags + 1);
                for s in (0..=max_instr_period).rev() {
                    for j in 0..k {
                        instr.push(data[(s, j)]);
                    }
                }
                all_instr.push(Array1::from_vec(instr));
            }
        }

        let n_obs = all_dy.len();
        if n_obs < n_reg + 1 {
            return Err(GreenersError::InvalidOperation(
                "PVAR: too few observations after differencing".into(),
            ));
        }

        let n_instr = all_instr[0].len();
        let _total_instr: usize = all_instr.iter().map(|v| v.len()).sum();

        // Stack into matrices
        let mut dy_mat = Array2::zeros((n_obs, k));
        let mut dx_mat = Array2::zeros((n_obs, n_reg));
        let mut z_mat = Array2::zeros((n_obs, n_instr));

        for i in 0..n_obs {
            for j in 0..k {
                dy_mat[(i, j)] = all_dy[i][j];
            }
            for j in 0..n_reg {
                dx_mat[(i, j)] = all_dx[i][j];
            }
            let instr_len = all_instr[i].len();
            for j in 0..instr_len.min(n_instr) {
                z_mat[(i, j)] = all_instr[i][j];
            }
        }

        // GMM estimation (2-step)
        // Step 1: Identity weighting
        let zt = z_mat.t();
        let ztz = zt.dot(&z_mat);
        let ztz_inv = (&ztz + Array2::<f64>::eye(n_instr) * 1e-6).inv()?;

        // Estimate each equation separately
        let mut coeffs = Array2::zeros((k, n_reg));
        let mut std_errors = Array2::zeros((k, n_reg));
        let mut t_values = Array2::zeros((k, n_reg));
        let mut p_values = Array2::zeros((k, n_reg));
        let mut j_stat_total = 0.0_f64;

        for eq in 0..k {
            let dy_eq = dy_mat.column(eq).to_owned();

            // 2SLS / GMM
            let ztdy = zt.dot(&dy_eq);
            let ztdx = zt.dot(&dx_mat);
            let w_inv = ztz_inv.clone();
            let middle = ztdx.t().dot(&w_inv).dot(&ztdx);
            let middle_inv = (&middle + Array2::<f64>::eye(n_reg) * 1e-8).inv()?;
            let beta: Array1<f64> = middle_inv.dot(&ztdx.t().dot(&w_inv).dot(&ztdy));

            // Residuals
            let resid = &dy_eq - dx_mat.dot(&beta);
            let s2 = resid.dot(&resid) / n_obs as f64;

            // SE
            let cov = s2 * &middle_inv;
            let se = cov.diag().mapv(|v| v.sqrt());
            let tv = &beta / &se;

            // p-values (normal approximation)
            let normal = statrs::distribution::Normal::new(0.0, 1.0)
                .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
            let pv = tv.mapv(|t| 2.0 * (1.0 - normal.cdf(t.abs())));

            for col in 0..n_reg {
                coeffs[(eq, col)] = beta[col];
                std_errors[(eq, col)] = se[col];
                t_values[(eq, col)] = tv[col];
                p_values[(eq, col)] = pv[col];
            }

            // J-test
            let zr = zt.dot(&resid);
            let j = zr.dot(&ztz_inv.dot(&zr)) / s2;
            j_stat_total += j;
        }

        let df_j = n_instr.saturating_sub(n_reg * k).max(1) as f64;
        let j_p = 1.0 - chi2_cdf(j_stat_total, df_j);

        Ok(PanelVarResult {
            coeffs,
            std_errors,
            t_values,
            p_values,
            j_stat: j_stat_total,
            j_p,
            n_instruments: n_instr,
            n_obs,
            n_entities,
            n_vars: k,
            lags,
            var_names: names,
        })
    }
}

fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Wilson-Hilferty approximation
    let h = 2.0 / (9.0 * df);
    let z = ((x / df).powf(1.0 / 3.0) - (1.0 - h)) / h.sqrt();
    let normal = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
    normal.cdf(z)
}
