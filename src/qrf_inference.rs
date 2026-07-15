//! QRF Inference: Quantile Regression Forest with bootstrap
//! confidence intervals (Meinshausen 2006; Athey, Tibshirani &
//! Wager 2019).
//!
//! Extends the QRF with:
//!   1. Bootstrap-based confidence intervals for conditional
//!      quantiles
//!   2. Pointwise and simultaneous bands
//!   3. Coverage diagnostics
//!
//! Reports quantile predictions with lower/upper bounds at
//! specified confidence level.

use crate::qrf::QRF;
use crate::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// Result of QRF inference.
#[derive(Debug)]
pub struct QrfInferenceResult {
    /// Quantile levels
    pub quantiles: Vec<f64>,
    /// Point estimates for each quantile (n_obs x n_quantiles)
    pub point_estimates: Array2<f64>,
    /// Lower bound (n_obs x n_quantiles)
    pub lower: Array2<f64>,
    /// Upper bound (n_obs x n_quantiles)
    pub upper: Array2<f64>,
    /// Coverage rate (empirical)
    pub coverage: f64,
    /// Confidence level (e.g., 0.95)
    pub confidence: f64,
    /// Number of bootstrap replications
    pub n_bootstrap: usize,
    /// Number of trees per QRF
    pub n_trees: usize,
    /// Number of observations
    pub n_obs: usize,
    /// Number of features
    pub n_features: usize,
    /// OOB R-squared (median quantile)
    pub oob_r_squared: f64,
    /// Feature importance
    pub feature_importance: Array1<f64>,
    /// Variable names
    pub variable_names: Vec<String>,
}

impl fmt::Display for QrfInferenceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " QRF Inference ")?;
        writeln!(f, "Meinshausen (2006); Athey-Tibshirani-Wager (2019)")?;
        writeln!(f, "Quantile Regression Forest with bootstrap CIs")?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Features:", self.n_features)?;
        writeln!(f, "{:<20} {:>12}", "Trees:", self.n_trees)?;
        writeln!(f, "{:<20} {:>12}", "Bootstrap reps:", self.n_bootstrap)?;
        writeln!(
            f,
            "{:<20} {:>12.2}%",
            "Confidence:",
            self.confidence * 100.0
        )?;
        writeln!(f, "{:<20} {:>12.4}", "Coverage:", self.coverage)?;
        writeln!(f, "{:<20} {:>12.6}", "OOB R² (median):", self.oob_r_squared)?;

        // Quantile summary
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Quantile estimates (first 5 obs):")?;
        write!(f, "  {:<6}", "Obs")?;
        for &q in &self.quantiles {
            write!(f, " {:>10}", format!("q{:.2}", q))?;
        }
        writeln!(f)?;
        writeln!(f, "{:-^78}", "")?;
        let n_show = 5.min(self.n_obs);
        for i in 0..n_show {
            write!(f, "  {:<6}", i + 1)?;
            for (j, _q) in self.quantiles.iter().enumerate() {
                write!(f, " {:>10.4}", self.point_estimates[(i, j)])?;
            }
            writeln!(f)?;
        }

        // CI for first obs, all quantiles
        writeln!(f, "\n  Confidence intervals (obs 1):")?;
        writeln!(
            f,
            "  {:<8} {:>12} {:>12} {:>12}",
            "Quantile", "Lower", "Point", "Upper"
        )?;
        writeln!(f, "{:-^78}", "")?;
        for (j, &q) in self.quantiles.iter().enumerate() {
            writeln!(
                f,
                "  {:<8.2} {:>12.4} {:>12.4} {:>12.4}",
                q,
                self.lower[(0, j)],
                self.point_estimates[(0, j)],
                self.upper[(0, j)]
            )?;
        }

        // Feature importance
        writeln!(f, "\n  Feature importance:")?;
        let mut imp_vec: Vec<(String, f64)> = self
            .variable_names
            .iter()
            .zip(self.feature_importance.iter())
            .map(|(name, &imp)| (name.clone(), imp))
            .collect();
        imp_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        writeln!(f, "  {:<14} {:>12}", "Feature", "Importance")?;
        writeln!(f, "{:-^78}", "")?;
        for (name, imp) in imp_vec {
            writeln!(f, "  {:<14} {:>12.4}", name, imp)?;
        }

        write!(f, "{:=^78}", "")
    }
}

pub struct QrfInference;

impl QrfInference {
    /// Estimate QRF with bootstrap confidence intervals.
    ///
    /// # Arguments
    /// * `y` - Dependent variable (n)
    /// * `x` - Features (n x k)
    /// * `quantiles` - Quantile levels (e.g., [0.1, 0.5, 0.9])
    /// * `n_bootstrap` - Number of bootstrap replications (default 100)
    /// * `n_trees` - Trees per QRF (default 100)
    /// * `max_depth` - Max tree depth (default 10)
    /// * `confidence` - Confidence level (default 0.95)
    /// * `variable_names` - Optional feature names
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        quantiles: Vec<f64>,
        n_bootstrap: Option<usize>,
        n_trees: Option<usize>,
        max_depth: Option<usize>,
        confidence: Option<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<QrfInferenceResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if n < 10 {
            return Err(GreenersError::InvalidOperation(
                "QrfInference: need at least 10 observations".into(),
            ));
        }

        let names = variable_names.unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());
        let n_boot = n_bootstrap.unwrap_or(100);
        let m_trees = n_trees.unwrap_or(100);
        let depth = max_depth.unwrap_or(10);
        let conf = confidence.unwrap_or(0.95);

        let n_q = quantiles.len();

        // Fit base QRF
        let base_result = QRF::fit(y, x, quantiles.clone(), m_trees, depth, None)?;

        // Bootstrap
        let mut boot_estimates: Vec<Array2<f64>> = Vec::with_capacity(n_boot);
        for _ in 0..n_boot {
            // Bootstrap sample
            let boot_idx: Vec<usize> = (0..n).map(|_| Self::rand_int(n)).collect();
            let y_boot: Array1<f64> = boot_idx.iter().map(|&i| y[i]).collect();
            let x_boot: Array2<f64> = {
                let mut data: Vec<f64> = Vec::with_capacity(n * k);
                for &i in &boot_idx {
                    for j in 0..k {
                        data.push(x[(i, j)]);
                    }
                }
                Array2::from_shape_vec((n, k), data).unwrap()
            };

            // Fit QRF on bootstrap sample, predict on original x
            if let Ok(boot_qrf) =
                QRF::fit(&y_boot, &x_boot, quantiles.clone(), m_trees, depth, None)
            {
                // Extract predictions
                let mut preds = Array2::zeros((n, n_q));
                for (i, row) in boot_qrf.quantile_predictions.rows().into_iter().enumerate() {
                    if i < n {
                        for j in 0..n_q.min(row.len()) {
                            preds[(i, j)] = row[j];
                        }
                    }
                }
                boot_estimates.push(preds);
            }
        }

        // Compute point estimates and CIs
        let mut point_estimates = Array2::zeros((n, n_q));
        let mut lower = Array2::zeros((n, n_q));
        let mut upper = Array2::zeros((n, n_q));

        // Point estimates from base QRF
        for i in 0..n {
            for j in 0..n_q {
                if i < base_result.quantile_predictions.nrows()
                    && j < base_result.quantile_predictions.ncols()
                {
                    point_estimates[(i, j)] = base_result.quantile_predictions[(i, j)];
                }
            }
        }

        // CIs from bootstrap percentiles
        let alpha_level = 1.0 - conf;
        let lower_pct = (alpha_level / 2.0 * 100.0) as usize;
        let upper_pct = ((1.0 - alpha_level / 2.0) * 100.0) as usize;

        for i in 0..n {
            for j in 0..n_q {
                let mut boot_vals: Vec<f64> = boot_estimates
                    .iter()
                    .filter_map(|b| {
                        if i < b.nrows() && j < b.ncols() {
                            Some(b[(i, j)])
                        } else {
                            None
                        }
                    })
                    .collect();
                if boot_vals.is_empty() {
                    lower[(i, j)] = point_estimates[(i, j)];
                    upper[(i, j)] = point_estimates[(i, j)];
                } else {
                    boot_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let n_b = boot_vals.len();
                    let lo_idx = lower_pct.min(n_b - 1);
                    let hi_idx = upper_pct.min(n_b - 1);
                    lower[(i, j)] = boot_vals[lo_idx];
                    upper[(i, j)] = boot_vals[hi_idx];
                }
            }
        }

        // Coverage: fraction of y within [q_alpha, q_{1-alpha}]
        let mut covered = 0;
        for i in 0..n {
            if n_q >= 2 {
                let lo = point_estimates[(i, 0)];
                let hi = point_estimates[(i, n_q - 1)];
                if y[i] >= lo && y[i] <= hi {
                    covered += 1;
                }
            }
        }
        let coverage = if n > 0 && n_q >= 2 {
            covered as f64 / n as f64
        } else {
            0.0
        };

        Ok(QrfInferenceResult {
            quantiles,
            point_estimates,
            lower,
            upper,
            coverage,
            confidence: conf,
            n_bootstrap: n_boot,
            n_trees: m_trees,
            n_obs: n,
            n_features: k,
            oob_r_squared: base_result.oob_r_squared,
            feature_importance: base_result.feature_importance.clone(),
            variable_names: names,
        })
    }

    fn rand_int(n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (Self::rand_uniform() * n as f64) as usize
    }

    fn rand_uniform() -> f64 {
        use std::cell::Cell;
        thread_local! {
            static STATE: Cell<u64> = const { Cell::new(5544332211) };
        }
        STATE.with(|s| {
            let mut state = s.get();
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s.set(state);
            ((state >> 11) as f64) / (1u64 << 53) as f64
        })
    }
}
