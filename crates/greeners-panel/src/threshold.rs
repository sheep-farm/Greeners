use crate::panel::FixedEffects;
use greeners_core::GreenersError;
use ndarray::{s, Array1, Array2};
use std::fmt;

#[derive(Debug)]
pub struct ThresholdResult {
    pub threshold_gamma: f64,
    pub params_regime1: Array1<f64>, //Coefficients when q <= gamma
    pub params_regime2: Array1<f64>, //Coefficients when q > gamma
    pub r_squared: f64,
    pub ssr_min: f64,
    pub n_search: usize, // Quantos candidatos testamos
}

impl fmt::Display for ThresholdResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Panel Threshold Model (Hansen 1999) ")?;
        writeln!(
            f,
            "{:<25} {:>10.4}",
            "Estimated Threshold (Gamma):", self.threshold_gamma
        )?;
        writeln!(
            f,
            "{:<25} {:>10.4}",
            "R-squared (Combined):", self.r_squared
        )?;
        writeln!(f, "{:<25} {:>10.4e}", "Min SSR:", self.ssr_min)?;

        writeln!(f, "\n{:-^78}", " Regime 1 (Below Threshold) ")?;
        writeln!(f, "{:<10} | {:>12}", "Variable", "Coef")?;
        for i in 0..self.params_regime1.len() {
            writeln!(f, "x{:<9} | {:>12.4}", i, self.params_regime1[i])?;
        }

        writeln!(f, "\n{:-^78}", " Regime 2 (Above Threshold) ")?;
        writeln!(f, "{:<10} | {:>12}", "Variable", "Coef")?;
        for i in 0..self.params_regime2.len() {
            writeln!(f, "x{:<9} | {:>12.4}", i, self.params_regime2[i])?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

pub struct PanelThreshold;

impl PanelThreshold {
    /// Estimates the Single Threshold model.
    /// Grid Search on 'q' to find the great breakpoint.
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        q: &Array1<f64>, //Threshold Variable
        entity_ids: &Array1<i64>,
    ) -> Result<ThresholdResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();

        if q.len() != n || entity_ids.len() != n {
            return Err(GreenersError::ShapeMismatch("Input lengths differ".into()));
        }

        //1. Set Search Grid (Trimmed)
        //We need the unique values of q, ordered
        let mut q_vec: Vec<f64> = q.to_vec();
        q_vec.sort_by(|a, b| a.total_cmp(b));
        q_vec.dedup(); //Only single

        let n_unique = q_vec.len();
        // Hansen recomenda descartar 15% das pontas (trimming parameter)
        //to avoid regimes with very little data.
        let trim_idx = (n_unique as f64 * 0.15).ceil() as usize;

        if n_unique < 2 * trim_idx + 5 {
            return Err(GreenersError::OptimizationFailed); // "Not enough variability in threshold variable"
        }

        let candidates = &q_vec[trim_idx..(n_unique - trim_idx)];

        //Optimisation: If there are many candidates (>300), skip some to speed
        let step = if candidates.len() > 300 {
            candidates.len() / 100
        } else {
            1
        };

        let mut best_gamma = 0.0;
        let mut min_ssr = f64::INFINITY;
        let mut best_params = Array1::<f64>::zeros(2 * k); //You'll save [Beta1, Beta2]
        let mut best_r2 = 0.0;

        //Raw IDs for FE
        let id_slice = entity_ids.as_slice().ok_or_else(|| {
            GreenersError::InvalidOperation("Non-contiguous entity IDs".to_string())
        })?;

        //2. Grid Search Loop
        for i in (0..candidates.len()).step_by(step) {
            let gamma = candidates[i];

            // Construir Matriz Expandida [X_low, X_high]
            // X_low = X * I(q <= gamma)
            // X_high = X * I(q > gamma)

            //It is more efficient to create flat vectors and transform into Array2
            let mut x_expanded_vec = Vec::with_capacity(n * 2 * k);

            for row_idx in 0..n {
                let q_val = q[row_idx];
                let x_row = x.row(row_idx);

                // if q_val <= gamma {
                //     // Regime 1 Ativo: [x, 0]
                //     for val in x_row {
                //         x_expanded_vec.push(*val);
                //     }
                //     for _ in 0..k {
                //         x_expanded_vec.push(0.0);
                //     }
                //     // Regime 2 Ativo: [0, x]
                //     for _ in 0..k {
                //         x_expanded_vec.push(0.0);
                //     }
                //     for val in x_row {
                //         x_expanded_vec.push(*val);
                //     }
                // }
                if q_val <= gamma {
                    // Regime 1 Ativo: [x, 0]
                    x_expanded_vec.extend(x_row);
                    x_expanded_vec.extend(std::iter::repeat_n(0.0, k));
                } else {
                    // Regime 2 Ativo: [0, x]
                    x_expanded_vec.extend(std::iter::repeat_n(0.0, k));
                    x_expanded_vec.extend(x_row);
                }
            }

            let x_expanded = Array2::from_shape_vec((n, 2 * k), x_expanded_vec)
                .map_err(|e| GreenersError::ShapeMismatch(e.to_string()))?;

            //Rotate Fixed Effects for this Gamma
            // Isso já cuida da remoção das médias individuais (mu_i)
            let fe_res = FixedEffects::fit(y, &x_expanded, id_slice);

            if let Ok(model) = fe_res {
                //Calculate model SSR
                // SSR = Sigma^2 * df_resid (revertendo a conta do sigma)
                let ssr = model.sigma.powi(2) * (model.df_resid as f64);

                if ssr < min_ssr {
                    min_ssr = ssr;
                    best_gamma = gamma;
                    best_params = model.params;
                    best_r2 = model.r_squared;
                }
            }
        }

        //3. Separate parameters
        let params_regime1 = best_params.slice(s![0..k]).to_owned();
        let params_regime2 = best_params.slice(s![k..2 * k]).to_owned();

        Ok(ThresholdResult {
            threshold_gamma: best_gamma,
            params_regime1,
            params_regime2,
            r_squared: best_r2,
            ssr_min: min_ssr,
            n_search: candidates.len() / step,
        })
    }
}
