//! Local Projections Difference-in-Differences (LP-DiD) estimator.
//!
//! Native Rust implementation of the estimator in Dube, Girardi, Jordà & Taylor
//! (2025, *Journal of Applied Econometrics*). Supports the three target
//! estimands (`vw`, `rw`, `ra`) and the clean-control conditions for both
//! absorbing and non-absorbing treatment settings described in the paper.
//!
//! This implementation is adapted from `pylpdid`
//! (<https://github.com/Daniel-Uhr/pylpdid>).
//! Original code copyright (c) 2026 Daniel de Abreu Pereira Uhr,
//! used under the MIT License. See the `pylpdid/LICENSE` file or
//! `Greeners/LICENSE` for the full license text.

#![allow(warnings, clippy::all)]

use crate::error::GreenersError;
use crate::linalg::{LinalgInverse, LinalgPinv};
use crate::{Column, CovarianceType, DataFrame, OLS};
use indexmap::IndexMap;
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::HashMap;
use std::fmt;

// -----------------------------------------------------------------------------
// Public result type
// -----------------------------------------------------------------------------

/// Result of an LP-DiD estimation.
#[derive(Debug, Clone)]
pub struct LpDidResult {
    /// Horizon values (relative to treatment)
    pub horizons: Vec<i64>,
    /// ATT estimates per horizon
    pub estimates: Array1<f64>,
    /// Standard errors per horizon
    pub standard_errors: Array1<f64>,
    /// t-statistics per horizon
    pub t_values: Array1<f64>,
    /// p-values per horizon
    pub p_values: Array1<f64>,
    /// Lower bound of the confidence interval per horizon
    pub conf_lower: Array1<f64>,
    /// Upper bound of the confidence interval per horizon
    pub conf_upper: Array1<f64>,
    /// Number of observations per horizon
    pub n_obs_per_horizon: Vec<usize>,

    /// Scalar summary labels (`ATT avg`, `ATT pooled`)
    pub scalar_terms: Vec<String>,
    /// Scalar summary estimates
    pub scalar_estimates: Array1<f64>,
    /// Scalar summary standard errors
    pub scalar_standard_errors: Array1<f64>,
    /// Scalar summary t-statistics
    pub scalar_t_values: Array1<f64>,
    /// Scalar summary p-values
    pub scalar_p_values: Array1<f64>,
    /// Scalar summary confidence lower bounds
    pub scalar_conf_lower: Array1<f64>,
    /// Scalar summary confidence upper bounds
    pub scalar_conf_upper: Array1<f64>,

    /// Total number of panel observations used
    pub n_obs: usize,
    /// Number of ever-treated units
    pub n_treated_units: usize,
    /// Number of never-treated units
    pub n_control_units: usize,
    /// Number of treatment cohorts
    pub n_cohorts: usize,
    /// Number of distinct time periods
    pub n_periods: usize,

    /// Base period label
    pub base_period: String,
    /// Clean-control strategy
    pub clean_control: String,
    /// Target estimand
    pub estimand: String,
    /// Maximum pre-treatment horizon
    pub max_pre: usize,
    /// Maximum post-treatment horizon
    pub max_post: usize,
}

impl fmt::Display for LpDidResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let w = 100;
        writeln!(f, "\n{:=^w$}", " LP-DiD Results ")?;
        writeln!(
            f,
            "Estimand: {:<18} Clean control: {:<22} Base: {}",
            self.estimand, self.clean_control, self.base_period
        )?;
        writeln!(
            f,
            "Max pre: {:<20} Max post: {:<20} N={}",
            self.max_pre, self.max_post, self.n_obs
        )?;
        writeln!(
            f,
            "Treated units: {:<14} Control units: {:<14} Cohorts: {}  Periods: {}",
            self.n_treated_units, self.n_control_units, self.n_cohorts, self.n_periods
        )?;

        fn fmt_num(x: f64, width: usize, digits: usize) -> String {
            if x.is_finite() {
                format!("{:>width$.digits$}", x)
            } else {
                format!("{:>width$}", "nan")
            }
        }

        let sig = |p: f64| -> &'static str {
            if !p.is_finite() {
                ""
            } else if p < 0.001 {
                "***"
            } else if p < 0.01 {
                "**"
            } else if p < 0.05 {
                "*"
            } else if p < 0.1 {
                "."
            } else {
                ""
            }
        };

        if !self.scalar_terms.is_empty() {
            writeln!(f, "\n{:-^w$}", " Scalar summaries ")?;
            writeln!(
                f,
                "{:<38} {:>12} {:>14} {:>14} {:>14} {:>8}",
                "Term", "Estimate", "Std.Err.", "t", "P>|t|", "Sig."
            )?;
            writeln!(f, "{:-^w$}", "")?;
            for (i, term) in self.scalar_terms.iter().enumerate() {
                writeln!(
                    f,
                    "{:<38} {:>12} {:>14} {:>14} {:>14} {:>8}",
                    term,
                    fmt_num(self.scalar_estimates[i], 12, 4),
                    fmt_num(self.scalar_standard_errors[i], 14, 4),
                    fmt_num(self.scalar_t_values[i], 14, 3),
                    fmt_num(self.scalar_p_values[i], 14, 4),
                    sig(self.scalar_p_values[i])
                )?;
            }
            writeln!(f, "{:-^w$}", "")?;
        }

        writeln!(f, "\n{:-^w$}", " Event-study path ")?;
        writeln!(
            f,
            "{:<12} {:>12} {:>14} {:>14} {:>14} {:>14} {:>14} {:>8}",
            "Horizon", "Estimate", "Std.Err.", "t", "P>|t|", "CI lower", "CI upper", "N"
        )?;
        writeln!(f, "{:-^w$}", "")?;
        for (i, &h) in self.horizons.iter().enumerate() {
            writeln!(
                f,
                "h={:<8} {:>12} {:>14} {:>14} {:>14} {:>14} {:>14} {:>8}",
                h,
                fmt_num(self.estimates[i], 12, 4),
                fmt_num(self.standard_errors[i], 14, 4),
                fmt_num(self.t_values[i], 14, 3),
                fmt_num(self.p_values[i], 14, 4),
                fmt_num(self.conf_lower[i], 14, 4),
                fmt_num(self.conf_upper[i], 14, 4),
                self.n_obs_per_horizon[i]
            )?;
        }
        writeln!(f, "{:=^w$}", "")
    }
}

// -----------------------------------------------------------------------------
// Public estimator struct and builder
// -----------------------------------------------------------------------------

/// LP-DiD estimator.
///
/// Configure the estimator with the chainable `with_*` methods and then call
/// [`LpDid::fit`].
#[derive(Debug, Clone)]
pub struct LpDid {
    target_estimand: String,
    base_period: BasePeriod,
    clean_control: String,
    nonabsorbing: bool,
    effect_stabilization: Option<usize>,
    anticipation: usize,
    include_lagged_outcome_change: bool,
    n_lagged_outcome_changes: usize,
    alpha: f64,
    max_pre: Option<usize>,
    max_post: Option<usize>,
    lag_covariates: bool,
    fixed_composition: bool,
    control_pool: String,
    switch_in: String,
}

impl Default for LpDid {
    fn default() -> Self {
        Self {
            target_estimand: "vw".to_string(),
            base_period: BasePeriod::Single(-1),
            clean_control: "not_yet_treated".to_string(),
            nonabsorbing: false,
            effect_stabilization: None,
            anticipation: 0,
            include_lagged_outcome_change: false,
            n_lagged_outcome_changes: 0,
            alpha: 0.05,
            max_pre: None,
            max_post: None,
            lag_covariates: false,
            fixed_composition: false,
            control_pool: "stabilized_all".to_string(),
            switch_in: "sustained".to_string(),
        }
    }
}

impl LpDid {
    /// Create an LP-DiD estimator with the default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Target estimand: `"vw"`, `"rw"` or `"ra"`.
    pub fn with_target_estimand(mut self, estimand: &str) -> Self {
        self.target_estimand = estimand.to_lowercase();
        self
    }

    /// Single negative integer base period (e.g. `-1`).
    pub fn with_base_period_int(mut self, base: i64) -> Self {
        self.base_period = BasePeriod::Single(base);
        self
    }

    /// List of negative integer base periods (PMD base).
    pub fn with_base_period_list(mut self, base: &[i64]) -> Self {
        self.base_period = BasePeriod::List(base.to_vec());
        self
    }

    /// Use the average of all pre-treatment outcomes as the base.
    pub fn with_base_period_all_pre(mut self) -> Self {
        self.base_period = BasePeriod::AllPre;
        self
    }

    /// Clean-control condition:
    /// `"not_yet_treated"`, `"never_treated"`, `"first_entry"` or `"stabilized"`.
    pub fn with_clean_control(mut self, clean_control: &str) -> Self {
        self.clean_control = clean_control.to_lowercase();
        self
    }

    /// Allow treatment to switch off (non-absorbing).
    pub fn with_nonabsorbing(mut self, nonabsorbing: bool) -> Self {
        self.nonabsorbing = nonabsorbing;
        self
    }

    /// Effect-stabilization horizon `L` (required for `clean_control="stabilized"`).
    pub fn with_effect_stabilization(mut self, l: Option<usize>) -> Self {
        self.effect_stabilization = l;
        self
    }

    /// Number of pre-treatment periods excluded from pre-trend estimation.
    pub fn with_anticipation(mut self, anticipation: usize) -> Self {
        self.anticipation = anticipation;
        self
    }

    /// Include lagged first-differences of the outcome as controls.
    pub fn with_include_lagged_outcome_change(mut self, include: bool) -> Self {
        self.include_lagged_outcome_change = include;
        self
    }

    /// Number of lagged first-difference controls to include.
    pub fn with_n_lagged_outcome_changes(mut self, n: usize) -> Self {
        self.n_lagged_outcome_changes = n;
        self
    }

    /// Lag user-supplied covariates by one period inside each local stack.
    pub fn with_lag_covariates(mut self, lag: bool) -> Self {
        self.lag_covariates = lag;
        self
    }

    /// Hold the unit set constant across horizons (Section 3.6).
    pub fn with_fixed_composition(mut self, fixed: bool) -> Self {
        self.fixed_composition = fixed;
        self
    }

    /// Control pool for `clean_control="stabilized"`:
    /// `"stabilized_all"` or `"untreated_only"`.
    pub fn with_control_pool(mut self, pool: &str) -> Self {
        self.control_pool = pool.to_lowercase();
        self
    }

    /// Switch-in estimand for `clean_control="stabilized"`:
    /// `"sustained"` or `"onset"`.
    pub fn with_switch_in(mut self, switch_in: &str) -> Self {
        self.switch_in = switch_in.to_lowercase();
        self
    }

    /// Maximum pre-treatment horizon to include (data-determined if `None`).
    pub fn with_max_pre(mut self, max_pre: Option<usize>) -> Self {
        self.max_pre = max_pre;
        self
    }

    /// Maximum post-treatment horizon to include (data-determined if `None`).
    pub fn with_max_post(mut self, max_post: Option<usize>) -> Self {
        self.max_post = max_post;
        self
    }

    /// Significance level for confidence intervals (default `0.05`).
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Fit the LP-DiD estimator.
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        &self,
        df: &DataFrame,
        outcome: &str,
        unit: &str,
        time: &str,
        first_treat: Option<&str>,
        treatment: Option<&str>,
        covariates: Option<&[String]>,
    ) -> Result<LpDidResult, GreenersError> {
        self.validate_options()?;

        let covariates: Vec<String> = covariates.unwrap_or(&[]).to_vec();
        if !covariates.is_empty() {
            let _ = df
                .get_column(&covariates[0])
                .map_err(|_| GreenersError::VariableNotFound(covariates[0].clone()))?;
            for c in &covariates {
                let _ = df
                    .get_column(c)
                    .map_err(|_| GreenersError::VariableNotFound(c.clone()))?;
            }
        }

        let mut panel = prepare_panel(
            df,
            outcome,
            unit,
            time,
            first_treat,
            treatment,
            self.nonabsorbing,
        )?;

        let (max_pre, max_post) =
            infer_windows(&panel, self.max_pre, self.max_post, &self.base_period);

        if let Some(l) = self.effect_stabilization {
            panel.precompute_ccs(l, max_pre, max_post);
        }

        panel.precompute_base(&self.base_period);

        let controls = _ctrl_cols(
            &covariates,
            self.include_lagged_outcome_change,
            self.n_lagged_outcome_changes,
        );
        let control_matrix = build_control_matrix(df, &panel, &controls, self.lag_covariates)?;

        let z = z_crit(self.alpha);
        let mut rows: Vec<EventRow> = Vec::new();

        let fc_h = if self.fixed_composition {
            Some(max_post)
        } else {
            None
        };

        // Pre-compute h=0 RW weights for pre-period horizons.
        let mut rw0_map: Option<HashMap<i64, f64>> = None;
        if self.target_estimand == "rw" {
            let local0 = build_local_sample(
                &panel,
                0,
                &self.clean_control,
                self.effect_stabilization,
                &controls,
                &control_matrix,
                self.control_pool.as_str(),
                self.switch_in.as_str(),
                fc_h,
            )?;
            if local0.n_treated > 0 && local0.n_controls > 0 {
                let weights = compute_rw_weights(&local0)?;
                let mut tmap: HashMap<i64, f64> = HashMap::new();
                for (idx, &w) in weights.iter().enumerate() {
                    if w.is_finite() && w > 0.0 {
                        tmap.entry(local0.time[idx]).or_insert(w);
                    }
                }
                rw0_map = Some(tmap);
            }
        }

        // Pre and post horizons.
        let mut pre_horizons: Vec<i64> = (-(max_pre as i64)..-(self.anticipation as i64)).collect();
        if let BasePeriod::Single(bp) = self.base_period {
            pre_horizons.retain(|&h| h != bp);
        }
        let post_horizons: Vec<i64> = (0..=max_post as i64).collect();
        let all_horizons: Vec<i64> = pre_horizons
            .iter()
            .chain(post_horizons.iter())
            .copied()
            .collect();

        let horizon_results: Vec<EventRow> = all_horizons
            .par_iter()
            .map(|&h| {
                let local = build_local_sample(
                    &panel,
                    h,
                    &self.clean_control,
                    self.effect_stabilization,
                    &controls,
                    &control_matrix,
                    self.control_pool.as_str(),
                    self.switch_in.as_str(),
                    fc_h,
                )?;

                let pre_rw = if self.target_estimand == "rw" && h < 0 {
                    rw0_map.as_ref()
                } else {
                    None
                };

                let (estimate, psi) = if self.target_estimand == "ra" {
                    fit_ra(&local, &controls, &control_matrix, z)
                } else {
                    fit_linear(
                        &local,
                        &controls,
                        &control_matrix,
                        self.target_estimand.as_str(),
                        pre_rw,
                        z,
                    )
                }
                .unwrap_or_else(|_| (Estimate::nan(), HashMap::new()));
                Ok(EventRow {
                    horizon: h,
                    estimate: estimate.estimate,
                    se: estimate.se,
                    t_stat: estimate.t_stat,
                    p_value: estimate.p_value,
                    ci_lower: estimate.ci_lower,
                    ci_upper: estimate.ci_upper,
                    n_obs: local.indices.len(),
                    psi_by_cluster: psi,
                })
            })
            .collect::<Result<Vec<EventRow>, GreenersError>>()?;
        rows.extend(horizon_results);

        // Normalised base-period row for a single integer base period.
        if let BasePeriod::Single(bp) = self.base_period {
            rows.push(EventRow {
                horizon: bp,
                estimate: 0.0,
                se: 0.0,
                t_stat: f64::NAN,
                p_value: f64::NAN,
                ci_lower: 0.0,
                ci_upper: 0.0,
                n_obs: 0,
                psi_by_cluster: HashMap::new(),
            });
        }

        rows.sort_by(|a, b| a.horizon.cmp(&b.horizon));

        // Scalar summaries.
        let mut scalar_rows: Vec<ScalarRow> = Vec::new();
        let post_rows: Vec<&EventRow> = rows
            .iter()
            .filter(|r| r.horizon >= 0 && r.estimate.is_finite())
            .collect();
        let post_est: Vec<f64> = post_rows.iter().map(|r| r.estimate).collect();
        if !post_est.is_empty() {
            let avg = post_est.iter().sum::<f64>() / post_est.len() as f64;
            let se = scalar_avg_se(&post_rows);
            scalar_rows.push(inference_to_scalar("ATT avg".to_string(), avg, se, z));
        }

        if let Some(pw) = rows
            .iter()
            .filter(|r| r.horizon >= 0 && r.estimate.is_finite())
            .map(|r| r.horizon)
            .max()
        {
            let pw = pw as usize;
            if let Some(pooled_local) = build_pooled_local(
                &panel,
                pw,
                &self.clean_control,
                &controls,
                &control_matrix,
                self.control_pool.as_str(),
                self.switch_in.as_str(),
            ) {
                if pooled_local.n_treated > 0 && pooled_local.n_controls > 0 {
                    let (estimate, _psi) = if self.target_estimand == "ra" {
                        fit_ra(&pooled_local, &controls, &control_matrix, z)
                    } else {
                        fit_linear(
                            &pooled_local,
                            &controls,
                            &control_matrix,
                            self.target_estimand.as_str(),
                            None,
                            z,
                        )
                    }
                    .unwrap_or_else(|_| (Estimate::nan(), HashMap::new()));
                    scalar_rows.push(ScalarRow {
                        term: "ATT pooled".to_string(),
                        estimate: estimate.estimate,
                        se: estimate.se,
                        t_stat: estimate.t_stat,
                        p_value: estimate.p_value,
                        ci_lower: estimate.ci_lower,
                        ci_upper: estimate.ci_upper,
                    });
                }
            }
        }

        // Aggregate counts.
        let n_treated_units = panel
            .unit_indices
            .iter()
            .filter(|idxs| {
                idxs.first()
                    .map(|&i| panel.first_treat[i] > 0)
                    .unwrap_or(false)
            })
            .count();
        let n_control_units = panel
            .unit_indices
            .iter()
            .filter(|idxs| {
                idxs.first()
                    .map(|&i| panel.first_treat[i] == 0)
                    .unwrap_or(false)
            })
            .count();
        let mut cohorts: Vec<i64> = panel
            .first_treat
            .iter()
            .copied()
            .filter(|&g| g > 0)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        cohorts.sort_unstable();
        let n_cohorts = cohorts.len();
        let mut periods: Vec<i64> = panel
            .time
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        periods.sort_unstable();
        let n_periods = periods.len();

        let horizons: Vec<i64> = rows.iter().map(|r| r.horizon).collect();
        let n_per_h = rows.iter().map(|r| r.n_obs).collect();

        fn collect_vec(xs: &[EventRow], f: impl Fn(&EventRow) -> f64) -> Array1<f64> {
            Array1::from(xs.iter().map(f).collect::<Vec<_>>())
        }
        fn collect_scalar_vec(xs: &[ScalarRow], f: impl Fn(&ScalarRow) -> f64) -> Array1<f64> {
            Array1::from(xs.iter().map(f).collect::<Vec<_>>())
        }

        let scalar_terms: Vec<String> = scalar_rows.iter().map(|r| r.term.clone()).collect();

        Ok(LpDidResult {
            horizons,
            estimates: collect_vec(&rows, |r| r.estimate),
            standard_errors: collect_vec(&rows, |r| r.se),
            t_values: collect_vec(&rows, |r| r.t_stat),
            p_values: collect_vec(&rows, |r| r.p_value),
            conf_lower: collect_vec(&rows, |r| r.ci_lower),
            conf_upper: collect_vec(&rows, |r| r.ci_upper),
            n_obs_per_horizon: n_per_h,
            scalar_terms,
            scalar_estimates: collect_scalar_vec(&scalar_rows, |r| r.estimate),
            scalar_standard_errors: collect_scalar_vec(&scalar_rows, |r| r.se),
            scalar_t_values: collect_scalar_vec(&scalar_rows, |r| r.t_stat),
            scalar_p_values: collect_scalar_vec(&scalar_rows, |r| r.p_value),
            scalar_conf_lower: collect_scalar_vec(&scalar_rows, |r| r.ci_lower),
            scalar_conf_upper: collect_scalar_vec(&scalar_rows, |r| r.ci_upper),
            n_obs: panel.n,
            n_treated_units,
            n_control_units,
            n_cohorts,
            n_periods,
            base_period: self.base_period.to_string(),
            clean_control: self.clean_control.clone(),
            estimand: self.target_estimand.to_uppercase(),
            max_pre,
            max_post,
        })
    }

    fn validate_options(&self) -> Result<(), GreenersError> {
        if !["vw", "rw", "ra"].contains(&self.target_estimand.as_str()) {
            return Err(GreenersError::InvalidOperation(format!(
                "target_estimand must be one of {{'vw', 'rw', 'ra'}}, got '{}'",
                self.target_estimand
            )));
        }
        if ![
            "not_yet_treated",
            "never_treated",
            "first_entry",
            "stabilized",
        ]
        .contains(&self.clean_control.as_str())
        {
            return Err(GreenersError::InvalidOperation(format!(
                "clean_control must be one of {{'not_yet_treated', 'never_treated', 'first_entry', 'stabilized'}}, got '{}'",
                self.clean_control
            )));
        }
        if self.clean_control == "stabilized" && self.effect_stabilization.is_none() {
            return Err(GreenersError::InvalidOperation(
                "clean_control='stabilized' requires effect_stabilization".into(),
            ));
        }
        if !["stabilized_all", "untreated_only"].contains(&self.control_pool.as_str()) {
            return Err(GreenersError::InvalidOperation(format!(
                "control_pool must be 'stabilized_all' or 'untreated_only', got '{}'",
                self.control_pool
            )));
        }
        if !["sustained", "onset"].contains(&self.switch_in.as_str()) {
            return Err(GreenersError::InvalidOperation(format!(
                "switch_in must be 'sustained' or 'onset', got '{}'",
                self.switch_in
            )));
        }
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err(GreenersError::InvalidOperation(
                "alpha must be in [0, 1]".into(),
            ));
        }
        self.base_period.validate()?;
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// Internal types
// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum BasePeriod {
    Single(i64),
    List(Vec<i64>),
    AllPre,
}

impl BasePeriod {
    fn validate(&self) -> Result<(), GreenersError> {
        match self {
            BasePeriod::Single(b) => {
                if *b >= 0 {
                    return Err(GreenersError::InvalidOperation(
                        "base_period integer must be negative".into(),
                    ));
                }
            }
            BasePeriod::List(bs) => {
                if bs.is_empty() || bs.iter().any(|&b| b >= 0) {
                    return Err(GreenersError::InvalidOperation(
                        "base_period list must contain only negative integers".into(),
                    ));
                }
            }
            BasePeriod::AllPre => {}
        }
        Ok(())
    }
}

impl fmt::Display for BasePeriod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BasePeriod::Single(b) => write!(f, "{}", b),
            BasePeriod::List(bs) => {
                let s = bs
                    .iter()
                    .map(|b| b.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "[{}]", s)
            }
            BasePeriod::AllPre => write!(f, "all_pre"),
        }
    }
}

#[derive(Debug, Clone)]
struct EventRow {
    horizon: i64,
    estimate: f64,
    se: f64,
    t_stat: f64,
    p_value: f64,
    ci_lower: f64,
    ci_upper: f64,
    n_obs: usize,
    psi_by_cluster: HashMap<usize, f64>,
}

struct ScalarRow {
    term: String,
    estimate: f64,
    se: f64,
    t_stat: f64,
    p_value: f64,
    ci_lower: f64,
    ci_upper: f64,
}

#[derive(Debug, Clone)]
struct LocalSample {
    indices: Vec<usize>,
    d_local: Vec<i8>,
    outcome_local: Vec<f64>,
    time: Vec<i64>,
    unit_id: Vec<usize>,
    n_treated: usize,
    n_controls: usize,
}

impl LocalSample {
    fn len(&self) -> usize {
        self.indices.len()
    }
}

#[derive(Debug, Clone, Copy)]
struct Estimate {
    estimate: f64,
    se: f64,
    t_stat: f64,
    p_value: f64,
    ci_lower: f64,
    ci_upper: f64,
}

impl Estimate {
    fn nan() -> Self {
        Self {
            estimate: f64::NAN,
            se: f64::NAN,
            t_stat: f64::NAN,
            p_value: f64::NAN,
            ci_lower: f64::NAN,
            ci_upper: f64::NAN,
        }
    }
}

struct Panel {
    n: usize,
    outcome: Vec<f64>,
    time: Vec<i64>,
    unit_id: Vec<usize>,
    unit_indices: Vec<Vec<usize>>,
    first_treat: Vec<i64>,
    treat: Vec<i8>,
    d_treat: Vec<i8>,
    treat_obs: Vec<i8>,
    never_treated: Vec<bool>,
    rel_time: Vec<Option<i64>>,
    dy: Vec<f64>,
    base: Vec<f64>,
    pos_in_unit: Vec<usize>,
    ccs_pos: Vec<Vec<i8>>,
    ccs_neg: Vec<Vec<i8>>,
}

impl Panel {
    fn lag_index(&self, i: usize, k: usize) -> Option<usize> {
        if k == 0 {
            return Some(i);
        }
        let uid = self.unit_id[i];
        let pos = self.pos_in_unit[i];
        if pos < k {
            return None;
        }
        Some(self.unit_indices[uid][pos - k])
    }

    fn lead_index(&self, i: usize, h: usize) -> Option<usize> {
        let uid = self.unit_id[i];
        let pos = self.pos_in_unit[i];
        if pos + h >= self.unit_indices[uid].len() {
            return None;
        }
        Some(self.unit_indices[uid][pos + h])
    }

    fn outcome_at(&self, i: usize, h: i64) -> Option<f64> {
        if h >= 0 {
            self.lead_index(i, h as usize).map(|j| self.outcome[j])
        } else {
            self.lag_index(i, (-h) as usize).map(|j| self.outcome[j])
        }
    }

    fn dy_lag(&self, i: usize, k: usize) -> f64 {
        self.lag_index(i, k).map(|j| self.dy[j]).unwrap_or(f64::NAN)
    }

    fn lead_treat(&self, i: usize, h: usize) -> Option<i8> {
        self.lead_index(i, h).map(|j| self.treat[j])
    }

    fn lead_treat_obs(&self, i: usize, h: usize) -> Option<i8> {
        self.lead_index(i, h).map(|j| self.treat_obs[j])
    }

    fn lag_treat(&self, i: usize, k: usize) -> Option<i8> {
        self.lag_index(i, k).map(|j| self.treat[j])
    }

    fn lag_treat_obs(&self, i: usize, k: usize) -> Option<i8> {
        self.lag_index(i, k).map(|j| self.treat_obs[j])
    }

    fn ccs_at(&self, h: i64, i: usize) -> i8 {
        if h >= 0 {
            let idx = h as usize;
            if idx < self.ccs_pos.len() {
                self.ccs_pos[idx][i]
            } else {
                0
            }
        } else {
            let idx = (-h) as usize;
            if idx < self.ccs_neg.len() {
                self.ccs_neg[idx][i]
            } else {
                0
            }
        }
    }

    fn precompute_base(&mut self, base_period: &BasePeriod) {
        self.base = vec![f64::NAN; self.n];
        for idxs in self.unit_indices.iter() {
            for (pos, &i) in idxs.iter().enumerate() {
                self.base[i] = match base_period {
                    BasePeriod::Single(k) => {
                        let k = k.unsigned_abs() as usize;
                        if pos >= k {
                            self.outcome[idxs[pos - k]]
                        } else {
                            f64::NAN
                        }
                    }
                    BasePeriod::List(ks) => {
                        let mut sum = 0.0;
                        let mut count = 0usize;
                        for k in ks {
                            let k = k.unsigned_abs() as usize;
                            if pos >= k {
                                let v = self.outcome[idxs[pos - k]];
                                if v.is_finite() {
                                    sum += v;
                                    count += 1;
                                }
                            }
                        }
                        if count > 0 {
                            sum / count as f64
                        } else {
                            f64::NAN
                        }
                    }
                    BasePeriod::AllPre => {
                        if pos == 0 {
                            f64::NAN
                        } else {
                            let mut sum = 0.0;
                            let mut count = 0usize;
                            for &prev in idxs.iter().take(pos) {
                                let v = self.outcome[prev];
                                if v.is_finite() {
                                    sum += v;
                                    count += 1;
                                }
                            }
                            if count > 0 {
                                sum / count as f64
                            } else {
                                f64::NAN
                            }
                        }
                    }
                };
            }
        }
    }

    fn precompute_ccs(&mut self, l: usize, max_pre: usize, max_post: usize) {
        self.ccs_pos = vec![vec![0i8; self.n]; max_post + 1];
        self.ccs_neg = vec![vec![0i8; self.n]; max_pre + 1];

        // CCS_0
        for i in 0..self.n {
            let mut clean = true;
            for k in 1..=l {
                if let Some(li) = self.lag_index(i, k) {
                    if self.d_treat[li].abs() == 1 {
                        clean = false;
                        break;
                    }
                }
            }
            self.ccs_pos[0][i] = if clean { 1 } else { 0 };
        }

        if max_pre >= 1 {
            self.ccs_neg[1] = self.ccs_pos[0].clone();
        }

        for h in 1..=max_post {
            for i in 0..self.n {
                let prev = self.ccs_pos[h - 1][i] == 1;
                let no_future = self
                    .lead_index(i, h)
                    .map(|j| self.d_treat[j].abs() != 1)
                    .unwrap_or(true);
                self.ccs_pos[h][i] = if prev && no_future { 1 } else { 0 };
            }
        }

        for h in 2..=max_pre {
            for i in 0..self.n {
                let prev = self.ccs_neg[h - 1][i] == 1;
                let lag_prev = self
                    .lag_index(i, 1)
                    .map(|j| self.ccs_neg[h - 1][j] == 1)
                    .unwrap_or(true);
                self.ccs_neg[h][i] = if prev && lag_prev { 1 } else { 0 };
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Panel preparation
// -----------------------------------------------------------------------------

fn prepare_panel(
    df: &DataFrame,
    outcome_col: &str,
    unit_col: &str,
    time_col: &str,
    first_treat_col: Option<&str>,
    treatment_col: Option<&str>,
    nonabsorbing: bool,
) -> Result<Panel, GreenersError> {
    if first_treat_col.is_none() && treatment_col.is_none() {
        return Err(GreenersError::InvalidOperation(
            "Provide either first_treat or treatment".into(),
        ));
    }

    let n = df.n_rows();
    if n == 0 {
        return Err(GreenersError::ShapeMismatch("empty DataFrame".into()));
    }

    let mut outcome = column_to_f64(df.get_column(outcome_col)?, outcome_col)?;
    let time_f = column_to_f64(df.get_column(time_col)?, time_col)?;
    if time_f.iter().any(|v| !v.is_finite()) {
        return Err(GreenersError::InvalidOperation(
            "time column contains non-finite values".into(),
        ));
    }
    let mut time: Vec<i64> = time_f.iter().map(|&v| v as i64).collect();
    let mut unit_key = column_to_strings(df.get_column(unit_col)?, unit_col)?;

    let mut first_treat = vec![0i64; n];
    let mut treat = vec![0i8; n];
    let mut treat_obs = vec![1i8; n];

    if let Some(ft) = first_treat_col {
        let ft_f = column_to_f64(df.get_column(ft)?, ft)?;
        for i in 0..n {
            first_treat[i] = if ft_f[i].is_finite() {
                let v = ft_f[i] as i64;
                if v < 0 {
                    0
                } else {
                    v
                }
            } else {
                0
            };
        }
        for i in 0..n {
            treat[i] = if first_treat[i] > 0 && time[i] >= first_treat[i] {
                1
            } else {
                0
            };
        }
    } else if let Some(tr) = treatment_col {
        let tr_f = column_to_f64(df.get_column(tr)?, tr)?;
        for i in 0..n {
            if tr_f[i].is_nan() {
                treat_obs[i] = 0;
                treat[i] = 0;
            } else {
                let v = tr_f[i].round() as i8;
                if v != 0 && v != 1 {
                    return Err(GreenersError::InvalidOperation(format!(
                        "treatment column '{}' contains non-binary value {}",
                        tr, tr_f[i]
                    )));
                }
                treat[i] = v;
                treat_obs[i] = 1;
            }
        }
    }

    // Sort by unit then time.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        unit_key[a]
            .cmp(&unit_key[b])
            .then_with(|| time[a].cmp(&time[b]))
    });

    outcome = reorder(&outcome, &order);
    time = reorder(&time, &order);
    unit_key = reorder(&unit_key, &order);
    first_treat = reorder(&first_treat, &order);
    treat = reorder(&treat, &order);
    treat_obs = reorder(&treat_obs, &order);

    // Derive first_treat from treatment if necessary.
    if treatment_col.is_some() {
        let tmp_unit_indices = build_unit_indices(&unit_key);
        for idxs in tmp_unit_indices.iter() {
            let ft = idxs
                .iter()
                .filter(|&&i| treat[i] == 1 && treat_obs[i] == 1)
                .map(|&i| time[i])
                .next()
                .unwrap_or(0);
            for &i in idxs {
                first_treat[i] = ft;
            }
        }
    }

    // Recompute treat from first_treat (covers first_treat input).
    for i in 0..n {
        treat[i] = if first_treat[i] > 0 && time[i] >= first_treat[i] {
            1
        } else {
            0
        };
    }

    let unit_indices = build_unit_indices(&unit_key);

    // Absorbing treatment check.
    if !nonabsorbing {
        for idxs in unit_indices.iter() {
            for w in idxs.windows(2) {
                if treat[w[0]] == 1 && treat[w[1]] == 0 {
                    return Err(GreenersError::InvalidOperation(format!(
                        "treatment switches 1->0 for unit '{}'. Set nonabsorbing=true.",
                        unit_key[w[0]]
                    )));
                }
            }
        }
    }

    // Drop left-censored units.
    let mut keep = vec![true; n];
    let mut any_drop = false;
    for idxs in unit_indices.iter() {
        let fot = time[idxs[0]];
        let drop = idxs
            .iter()
            .any(|&i| treat[i] == 1 && first_treat[i] > 0 && first_treat[i] < fot);
        if drop {
            any_drop = true;
            for &i in idxs {
                keep[i] = false;
            }
        }
    }

    let (outcome, time, unit_key, first_treat, treat, treat_obs) = if any_drop {
        let kept: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
        (
            reorder(&outcome, &kept),
            reorder(&time, &kept),
            reorder(&unit_key, &kept),
            reorder(&first_treat, &kept),
            reorder(&treat, &kept),
            reorder(&treat_obs, &kept),
        )
    } else {
        (outcome, time, unit_key, first_treat, treat, treat_obs)
    };

    let n = outcome.len();

    // Rebuild unit structures after possible drop.
    let unit_indices = build_unit_indices(&unit_key);
    let mut pos_in_unit = vec![0usize; n];
    let mut unit_id = vec![0usize; n];
    for (uid, idxs) in unit_indices.iter().enumerate() {
        for (pos, &i) in idxs.iter().enumerate() {
            pos_in_unit[i] = pos;
            unit_id[i] = uid;
        }
    }

    let mut never_treated = vec![false; n];
    let mut rel_time: Vec<Option<i64>> = vec![None; n];
    for idxs in unit_indices.iter() {
        for &i in idxs {
            never_treated[i] = first_treat[i] == 0;
            if first_treat[i] > 0 {
                rel_time[i] = Some(time[i] - first_treat[i]);
            }
        }
    }

    // First-differenced outcome.
    let mut dy = vec![f64::NAN; n];
    for idxs in unit_indices.iter() {
        for (pos, &i) in idxs.iter().enumerate() {
            if pos > 0 {
                let prev = idxs[pos - 1];
                dy[i] = outcome[i] - outcome[prev];
            }
        }
    }

    // Treatment-change indicator D_treat.
    let mut d_treat = vec![0i8; n];
    for idxs in unit_indices.iter() {
        for (pos, &i) in idxs.iter().enumerate() {
            if pos == 0 {
                d_treat[i] = treat[i];
            } else {
                let prev = idxs[pos - 1];
                d_treat[i] = treat[i] - treat[prev];
            }
        }
    }

    Ok(Panel {
        n,
        outcome,
        time,
        unit_id,
        unit_indices,
        first_treat,
        treat,
        d_treat,
        treat_obs,
        never_treated,
        rel_time,
        dy,
        base: vec![f64::NAN; n],
        pos_in_unit,
        ccs_pos: Vec::new(),
        ccs_neg: Vec::new(),
    })
}

fn build_unit_indices(unit_key: &[String]) -> Vec<Vec<usize>> {
    let mut map: IndexMap<String, Vec<usize>> = IndexMap::new();
    for (i, u) in unit_key.iter().enumerate() {
        map.entry(u.clone()).or_default().push(i);
    }
    map.into_values().collect()
}

fn reorder<T: Clone>(vec: &[T], order: &[usize]) -> Vec<T> {
    order.iter().map(|&i| vec[i].clone()).collect()
}

fn infer_windows(
    panel: &Panel,
    max_pre: Option<usize>,
    max_post: Option<usize>,
    base_period: &BasePeriod,
) -> (usize, usize) {
    let mut min_rt = 0i64;
    let mut max_rt = 0i64;
    for t in panel.rel_time.iter().flatten() {
        if *t < min_rt {
            min_rt = *t;
        }
        if *t > max_rt {
            max_rt = *t;
        }
    }

    let mut pre_data = (-min_rt).max(0) as usize;
    match base_period {
        BasePeriod::Single(k) => {
            pre_data = pre_data.max(k.unsigned_abs() as usize);
        }
        BasePeriod::List(ks) => {
            if let Some(mk) = ks.iter().map(|k| k.abs()).max() {
                pre_data = pre_data.max(mk as usize);
            }
        }
        BasePeriod::AllPre => {
            pre_data = pre_data.max(1);
        }
    }

    let post_data = max_rt.max(0) as usize;
    let pre = max_pre.map(|x| x.min(pre_data)).unwrap_or(pre_data);
    let post = max_post.map(|x| x.min(post_data)).unwrap_or(post_data);
    (pre, post)
}

// -----------------------------------------------------------------------------
// Control handling
// -----------------------------------------------------------------------------

fn _ctrl_cols(covariates: &[String], include: bool, n: usize) -> Vec<String> {
    let mut out = Vec::new();
    if include && n > 0 {
        for k in 1..=n {
            out.push(format!("ldy{k}"));
        }
    }
    out.extend(covariates.iter().cloned());
    out
}

fn build_control_matrix(
    df: &DataFrame,
    panel: &Panel,
    controls: &[String],
    lag_covariates: bool,
) -> Result<Vec<Vec<f64>>, GreenersError> {
    let mut matrix = Vec::with_capacity(controls.len());
    for name in controls {
        if name.len() >= 4 && name.starts_with("ldy") {
            if let Ok(k) = name[3..].parse::<usize>() {
                let vals: Vec<f64> = (0..panel.n).map(|i| panel.dy_lag(i, k)).collect();
                matrix.push(vals);
                continue;
            }
        }
        let col = df
            .get_column(name)
            .map_err(|_| GreenersError::VariableNotFound(name.clone()))?;
        let raw = column_to_f64(col, name)?;
        let vals: Vec<f64> = if lag_covariates {
            (0..panel.n)
                .map(|i| panel.lag_index(i, 1).map(|j| raw[j]).unwrap_or(f64::NAN))
                .collect()
        } else {
            raw
        };
        matrix.push(vals);
    }
    Ok(matrix)
}

// -----------------------------------------------------------------------------
// Local sample construction
// -----------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_local_sample(
    panel: &Panel,
    h: i64,
    clean_control: &str,
    effect_stabilization: Option<usize>,
    controls: &[String],
    control_matrix: &[Vec<f64>],
    control_pool: &str,
    switch_in: &str,
    fixed_composition_h: Option<usize>,
) -> Result<LocalSample, GreenersError> {
    let fixed_comp = fixed_composition_h.is_some();
    let h_us = if h >= 0 { h as usize } else { (-h) as usize };

    let mut indices = Vec::new();
    let mut d_local = Vec::new();
    let mut outcome_local = Vec::new();
    let mut time = Vec::new();
    let mut unit_id = Vec::new();
    let mut n_treated = 0usize;
    let mut n_controls = 0usize;

    for (i, _) in panel.outcome.iter().enumerate().take(panel.n) {
        // Outcome at horizon h.
        let y_h = match panel.outcome_at(i, h) {
            Some(v) => v,
            None => continue,
        };
        let ol = y_h - panel.base[i];
        if !ol.is_finite() || !panel.base[i].is_finite() {
            continue;
        }

        // Fixed-composition pre-computation.
        let (not_late, ctrl_fixed) = if let Some(hh) = fixed_composition_h {
            let t_max = panel.time.iter().copied().max().unwrap_or(0);
            let not_late = panel.time[i] <= t_max - (hh as i64);
            let ctrl_fixed = panel.lead_treat(i, hh).map(|v| v == 0).unwrap_or(false);
            (Some(not_late), Some(ctrl_fixed))
        } else {
            (None, None)
        };

        // Treated mask.
        let treat_flag = match clean_control {
            "not_yet_treated" | "never_treated" => panel.d_treat[i] == 1,
            "first_entry" => {
                let newly = panel.d_treat[i] == 1;
                let first_time = panel.first_treat[i] == panel.time[i];
                let stays = if h >= 0 {
                    panel.lead_treat(i, h_us) == Some(1)
                } else {
                    true
                };
                if fixed_comp {
                    newly && first_time && stays && not_late.unwrap_or(true)
                } else {
                    newly && first_time && stays
                }
            }
            "stabilized" => {
                if effect_stabilization.is_none() {
                    return Err(GreenersError::InvalidOperation(
                        "clean_control='stabilized' requires effect_stabilization".into(),
                    ));
                }
                let ccs0 = panel.ccs_at(0, i) == 1;
                let mut tmask = panel.d_treat[i] == 1 && ccs0 && panel.treat_obs[i] == 1;
                if tmask {
                    if let Some(li) = panel.lag_treat(i, 1) {
                        let lo = panel.lag_treat_obs(i, 1);
                        tmask = li == 0 && lo == Some(1);
                    } else {
                        tmask = false;
                    }
                }
                if tmask && switch_in == "sustained" && h >= 0 {
                    let mut stay = true;
                    for j in 0..=h_us {
                        match (panel.lead_treat(i, j), panel.lead_treat_obs(i, j)) {
                            (Some(1), Some(1)) => {}
                            _ => {
                                stay = false;
                                break;
                            }
                        }
                    }
                    tmask = tmask && stay;
                }
                if fixed_comp {
                    tmask = tmask && not_late.unwrap_or(true);
                }
                tmask
            }
            _ => false,
        };

        // Control mask.
        let ctrl_flag = match clean_control {
            "never_treated" => panel.never_treated[i],
            "not_yet_treated" | "first_entry" => {
                if fixed_comp {
                    ctrl_fixed.unwrap_or(false)
                } else if h >= 0 {
                    panel.lead_treat(i, h_us) == Some(0)
                } else {
                    panel.treat[i] == 0
                }
            }
            "stabilized" => {
                if effect_stabilization.is_none() {
                    return Err(GreenersError::InvalidOperation(
                        "clean_control='stabilized' requires effect_stabilization".into(),
                    ));
                }
                let col_h = if fixed_comp {
                    fixed_composition_h.unwrap() as i64
                } else if h >= 0 {
                    h
                } else if h == -1 {
                    -1
                } else {
                    h
                };
                let stable = panel.ccs_at(col_h, i) == 1;
                let mut cmask = panel.d_treat[i] == 0 && stable && panel.treat_obs[i] == 1;
                if control_pool == "untreated_only" {
                    if let (Some(lt), Some(lo)) = (panel.lag_treat(i, 1), panel.lag_treat_obs(i, 1))
                    {
                        cmask = cmask && panel.treat[i] == 0 && lt == 0 && lo == 1;
                    } else {
                        cmask = false;
                    }
                }
                cmask
            }
            _ => false,
        };

        if !treat_flag && !ctrl_flag {
            continue;
        }
        if treat_flag && ctrl_flag {
            continue;
        }

        // Check controls are finite.
        let mut ok = true;
        for (col_idx, _) in controls.iter().enumerate() {
            let v = control_matrix[col_idx][i];
            if !v.is_finite() {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }

        if treat_flag {
            n_treated += 1;
        } else {
            n_controls += 1;
        }

        indices.push(i);
        d_local.push(if treat_flag { 1 } else { 0 });
        outcome_local.push(ol);
        time.push(panel.time[i]);
        unit_id.push(panel.unit_id[i]);
    }

    // Exact-replication convention: keep treated-only time cells but require
    // at least one control observation overall.
    if n_controls == 0 && n_treated > 0 {
        indices.clear();
        d_local.clear();
        outcome_local.clear();
        time.clear();
        unit_id.clear();
        n_treated = 0;
    }

    Ok(LocalSample {
        indices,
        d_local,
        outcome_local,
        time,
        unit_id,
        n_treated,
        n_controls,
    })
}

fn build_pooled_local(
    panel: &Panel,
    post_window: usize,
    clean_control: &str,
    controls: &[String],
    control_matrix: &[Vec<f64>],
    control_pool: &str,
    switch_in: &str,
) -> Option<LocalSample> {
    let mut indices = Vec::new();
    let mut d_local = Vec::new();
    let mut outcome_local = Vec::new();
    let mut time = Vec::new();
    let mut unit_id = Vec::new();
    let mut n_treated = 0usize;
    let mut n_controls = 0usize;

    for (i, _) in panel.outcome.iter().enumerate().take(panel.n) {
        if !panel.base[i].is_finite() {
            continue;
        }
        let mut pooled = 0.0;
        let mut count = 0usize;
        for h in 0..=post_window {
            if let Some(v) = panel.outcome_at(i, h as i64) {
                if v.is_finite() {
                    pooled += v;
                    count += 1;
                }
            }
        }
        if count == 0 {
            continue;
        }
        let ol = (pooled / count as f64) - panel.base[i];
        if !ol.is_finite() {
            continue;
        }

        // Treat/control masks at the pooled horizon (no fixed composition).
        let treat_flag = match clean_control {
            "not_yet_treated" | "never_treated" => panel.d_treat[i] == 1,
            "first_entry" => {
                let newly = panel.d_treat[i] == 1;
                let first_time = panel.first_treat[i] == panel.time[i];
                let stays = panel.lead_treat(i, post_window) == Some(1);
                newly && first_time && stays
            }
            "stabilized" => {
                let ccs0 = panel.ccs_at(0, i) == 1;
                let mut tmask = panel.d_treat[i] == 1 && ccs0 && panel.treat_obs[i] == 1;
                if tmask {
                    if let (Some(lt), Some(lo)) = (panel.lag_treat(i, 1), panel.lag_treat_obs(i, 1))
                    {
                        tmask = lt == 0 && lo == 1;
                    } else {
                        tmask = false;
                    }
                }
                if tmask && switch_in == "sustained" {
                    let mut stay = true;
                    for j in 0..=post_window {
                        match (panel.lead_treat(i, j), panel.lead_treat_obs(i, j)) {
                            (Some(1), Some(1)) => {}
                            _ => {
                                stay = false;
                                break;
                            }
                        }
                    }
                    tmask = tmask && stay;
                }
                tmask
            }
            _ => false,
        };

        let ctrl_flag = match clean_control {
            "never_treated" => panel.never_treated[i],
            "not_yet_treated" | "first_entry" => panel.lead_treat(i, post_window) == Some(0),
            "stabilized" => {
                let stable = panel.ccs_at(post_window as i64, i) == 1;
                let mut cmask = panel.d_treat[i] == 0 && stable && panel.treat_obs[i] == 1;
                if control_pool == "untreated_only" {
                    if let (Some(lt), Some(lo)) = (panel.lag_treat(i, 1), panel.lag_treat_obs(i, 1))
                    {
                        cmask = cmask && panel.treat[i] == 0 && lt == 0 && lo == 1;
                    } else {
                        cmask = false;
                    }
                }
                cmask
            }
            _ => false,
        };

        if !treat_flag && !ctrl_flag {
            continue;
        }
        if treat_flag && ctrl_flag {
            continue;
        }

        let mut ok = true;
        for (col_idx, _) in controls.iter().enumerate() {
            let v = control_matrix[col_idx][i];
            if !v.is_finite() {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }

        if treat_flag {
            n_treated += 1;
        } else {
            n_controls += 1;
        }

        indices.push(i);
        d_local.push(if treat_flag { 1 } else { 0 });
        outcome_local.push(ol);
        time.push(panel.time[i]);
        unit_id.push(panel.unit_id[i]);
    }

    if n_controls == 0 {
        return None;
    }

    Some(LocalSample {
        indices,
        d_local,
        outcome_local,
        time,
        unit_id,
        n_treated,
        n_controls,
    })
}

// -----------------------------------------------------------------------------
// Estimation helpers
// -----------------------------------------------------------------------------

fn compute_rw_weights(local: &LocalSample) -> Result<Vec<f64>, GreenersError> {
    let mut sum_d: HashMap<i64, f64> = HashMap::new();
    let mut count: HashMap<i64, usize> = HashMap::new();
    for (idx, &t) in local.time.iter().enumerate() {
        *sum_d.entry(t).or_insert(0.0) += local.d_local[idx] as f64;
        *count.entry(t).or_insert(0) += 1;
    }
    let mut mean_d: HashMap<i64, f64> = HashMap::new();
    for (&t, &s) in sum_d.iter() {
        let c = count[&t];
        mean_d.insert(t, s / c as f64);
    }

    let mut num = vec![f64::NAN; local.len()];
    let mut den = 0.0;
    for (idx, (&t, &d)) in local.time.iter().zip(&local.d_local).enumerate() {
        let p_t = mean_d[&t];
        let resid = d as f64 - p_t;
        if d == 1 {
            num[idx] = resid;
            den += resid;
        }
    }

    if den.abs() <= 1e-12 || !den.is_finite() {
        return Err(GreenersError::InvalidOperation(
            "RW weights: zero or invalid denominator".into(),
        ));
    }

    // Group max weight per time.
    let mut gw: HashMap<i64, f64> = HashMap::new();
    for (idx, (&t, &d)) in local.time.iter().zip(&local.d_local).enumerate() {
        if d == 1 {
            let w = num[idx] / den;
            let entry = gw.entry(t).or_insert(w);
            if w > *entry {
                *entry = w;
            }
        }
    }

    let mut out = vec![f64::NAN; local.len()];
    for (idx, &t) in local.time.iter().enumerate() {
        if let Some(&gwt) = gw.get(&t) {
            if gwt.is_finite() && gwt > 0.0 {
                out[idx] = 1.0 / gwt;
            }
        }
    }
    Ok(out)
}

fn fit_linear(
    local: &LocalSample,
    controls: &[String],
    control_matrix: &[Vec<f64>],
    estimand: &str,
    pre_rw: Option<&HashMap<i64, f64>>,
    z: f64,
) -> Result<(Estimate, HashMap<usize, f64>), GreenersError> {
    // Compute weights.
    let weights: Vec<Option<f64>> = if estimand == "rw" {
        if let Some(tmap) = pre_rw {
            local
                .time
                .iter()
                .map(|&t| tmap.get(&t).copied().filter(|&w| w.is_finite() && w > 0.0))
                .collect()
        } else {
            let w = compute_rw_weights(local)?;
            w.into_iter()
                .map(|v| {
                    if v.is_finite() && v > 0.0 {
                        Some(v)
                    } else {
                        None
                    }
                })
                .collect()
        }
    } else {
        vec![Some(1.0); local.len()]
    };

    let mut y_vec = Vec::new();
    let mut x_mat = Vec::new();
    let mut cluster_ids = Vec::new();
    let mut times_set: Vec<i64> = Vec::new();
    let mut time_map: HashMap<i64, usize> = HashMap::new();

    // Two-pass: first determine included times.
    for (idx, &t) in local.time.iter().enumerate() {
        if weights[idx].is_none() {
            continue;
        }
        *time_map.entry(t).or_insert_with(|| {
            let new_id = times_set.len();
            times_set.push(t);
            new_id
        });
    }

    let n_cols = 1 + times_set.len() + controls.len();
    let mut names = vec!["D_local".to_string()];
    for t in &times_set {
        names.push(format!("time_{t}"));
    }
    names.extend(controls.iter().cloned());

    for (idx, &t) in local.time.iter().enumerate() {
        let w = match weights[idx] {
            Some(w) => w,
            None => continue,
        };
        let sqrt_w = w.sqrt();
        y_vec.push(local.outcome_local[idx] * sqrt_w);

        let mut row = vec![0.0; n_cols];
        row[0] = local.d_local[idx] as f64 * sqrt_w;
        if let Some(&tid) = time_map.get(&t) {
            row[1 + tid] = sqrt_w;
        }
        let mut col = 1 + times_set.len();
        for (c, _) in controls.iter().enumerate() {
            let original_i = local.indices[idx];
            row[col] = control_matrix[c][original_i] * sqrt_w;
            col += 1;
        }
        x_mat.extend(row);
        cluster_ids.push(local.unit_id[idx]);
    }

    if y_vec.is_empty() {
        return Ok((Estimate::nan(), HashMap::new()));
    }

    let k = y_vec.len();
    let y = Array1::from(y_vec);
    let x = Array2::from_shape_vec((k, n_cols), x_mat)
        .map_err(|e| GreenersError::ShapeMismatch(e.to_string()))?;

    let cr = crate::linalg::drop_collinear(&x, &names, 1e-10);
    let x_clean = cr.x_clean;

    let d_idx = cr.keep_indices.iter().position(|&i| i == 0);
    if d_idx.is_none() || x_clean.nrows() <= x_clean.ncols() {
        return Ok((Estimate::nan(), HashMap::new()));
    }
    let d_idx = d_idx.unwrap();

    let fit = OLS::fit(&y, &x_clean, CovarianceType::Clustered(cluster_ids.clone()))
        .map_err(|_| GreenersError::OptimizationFailed)?;

    let est = fit.params[d_idx];
    let se = fit.std_errors[d_idx];
    let estimate = inference_stats(est, se, z)?;

    // Cluster-level influence for this horizon (used to compute SE of ATT avg).
    let psi = influence_linear(&x_clean, &y, &fit.params, d_idx, &cluster_ids)?;

    Ok((estimate, psi))
}

fn influence_linear(
    x: &Array2<f64>,
    y: &Array1<f64>,
    params: &Array1<f64>,
    d_idx: usize,
    cluster_ids: &[usize],
) -> Result<HashMap<usize, f64>, GreenersError> {
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || p == 0 || cluster_ids.len() != n {
        return Ok(HashMap::new());
    }

    let xt_x = x.t().dot(x);
    let xt_x_inv = xt_x.inv()?;
    let row_d = xt_x_inv.row(d_idx);

    let predicted = x.dot(params);
    let residuals = y - &predicted;

    let mut v_map: HashMap<usize, Vec<f64>> = HashMap::new();
    for i in 0..n {
        let cid = cluster_ids[i];
        let entry = v_map.entry(cid).or_default();
        if entry.is_empty() {
            entry.resize(p, 0.0);
        }
        let r = residuals[i];
        let xi = x.row(i);
        for j in 0..p {
            entry[j] += r * xi[j];
        }
    }

    let g = v_map.len();
    let correction = if g <= 1 {
        1.0
    } else {
        (g as f64 / (g as f64 - 1.0)) * ((n as f64 - 1.0) / ((n - p) as f64).max(1.0))
    };
    let sqrt_corr = correction.sqrt();

    let mut psi = HashMap::new();
    for (cid, v) in v_map {
        let v_arr = Array1::from(v);
        let psi_g = sqrt_corr * row_d.dot(&v_arr);
        psi.insert(cid, psi_g);
    }
    Ok(psi)
}

fn fit_ra(
    local: &LocalSample,
    controls: &[String],
    control_matrix: &[Vec<f64>],
    z: f64,
) -> Result<(Estimate, HashMap<usize, f64>), GreenersError> {
    // Restrict to time cells with at least one control.
    let mut ctrl_count: HashMap<i64, usize> = HashMap::new();
    for (idx, &t) in local.time.iter().enumerate() {
        if local.d_local[idx] == 0 {
            *ctrl_count.entry(t).or_insert(0) += 1;
        }
    }

    let mut keep_idx = Vec::new();
    for (idx, &t) in local.time.iter().enumerate() {
        if ctrl_count.get(&t).copied().unwrap_or(0) > 0 {
            keep_idx.push(idx);
        }
    }
    if keep_idx.is_empty() {
        return Ok((Estimate::nan(), HashMap::new()));
    }

    let mut times_set: Vec<i64> = Vec::new();
    let mut time_map: HashMap<i64, usize> = HashMap::new();
    for &idx in &keep_idx {
        let t = local.time[idx];
        let _ = time_map.entry(t).or_insert_with(|| {
            let id = times_set.len();
            times_set.push(t);
            id
        });
    }

    let n_cols = times_set.len() + controls.len();
    let mut names: Vec<String> = times_set.iter().map(|t| format!("time_{t}")).collect();
    names.extend(controls.iter().cloned());

    // Build full and control-only design matrices.
    let mut y_full = Vec::new();
    let mut x_full = Vec::new();
    let mut d_full = Vec::new();
    let mut y_ctrl = Vec::new();
    let mut x_ctrl = Vec::new();
    let mut cluster_ctrl = Vec::new();

    for &idx in &keep_idx {
        let i = local.indices[idx];
        let ol = local.outcome_local[idx];
        y_full.push(ol);
        d_full.push(local.d_local[idx]);

        let mut row = vec![0.0; n_cols];
        if let Some(&tid) = time_map.get(&local.time[idx]) {
            row[tid] = 1.0;
        }
        let mut col = times_set.len();
        for (c, _) in controls.iter().enumerate() {
            row[col] = control_matrix[c][i];
            col += 1;
        }
        x_full.extend(row.clone());

        if local.d_local[idx] == 0 {
            y_ctrl.push(ol);
            x_ctrl.extend(row);
            cluster_ctrl.push(local.unit_id[idx]);
        }
    }

    if y_ctrl.is_empty() || x_ctrl.is_empty() {
        return Ok((Estimate::nan(), HashMap::new()));
    }

    let k_full = keep_idx.len();
    let x_full_arr = Array2::from_shape_vec((k_full, n_cols), x_full)
        .map_err(|e| GreenersError::ShapeMismatch(e.to_string()))?;

    let k_ctrl = y_ctrl.len();
    let x_ctrl_arr = Array2::from_shape_vec((k_ctrl, n_cols), x_ctrl)
        .map_err(|e| GreenersError::ShapeMismatch(e.to_string()))?;

    let cr = crate::linalg::drop_collinear(&x_ctrl_arr, &names, 1e-10);
    let x_ctrl_clean = cr.x_clean;
    let keep = cr.keep_indices;

    if x_ctrl_clean.nrows() <= x_ctrl_clean.ncols() {
        return Ok((Estimate::nan(), HashMap::new()));
    }

    let fit = OLS::fit(
        &Array1::from(y_ctrl),
        &x_ctrl_clean,
        CovarianceType::Clustered(cluster_ctrl),
    )
    .map_err(|_| GreenersError::OptimizationFailed)?;

    let x_full_clean = x_full_arr.select(Axis(1), &keep);
    let pred = fit.predict(&x_full_clean);
    let mut sum_resid = 0.0;
    let mut count = 0usize;
    for (idx, &d) in d_full.iter().enumerate() {
        if d == 1 {
            let resid = y_full[idx] - pred[idx];
            sum_resid += resid;
            count += 1;
        }
    }
    if count == 0 {
        return Ok((Estimate::nan(), HashMap::new()));
    }
    let est = sum_resid / count as f64;

    // RA standard error via stacked GMM influence functions.
    let p = fit.params.len();
    let mut theta = Vec::with_capacity(p + 1);
    theta.extend(fit.params.iter());
    theta.push(est);

    let cluster_full: Vec<usize> = keep_idx.iter().map(|&k| local.unit_id[k]).collect();
    let x = x_full_clean.clone();
    let y = y_full.clone();
    let d = d_full.clone();

    let moments = move |th: &[f64]| -> Array2<f64> {
        let mu = th[p];
        let beta_arr = Array1::from(th[..p].to_vec());
        let n = x.nrows();
        let mut m = Array2::<f64>::zeros((n, p + 1));
        for i in 0..n {
            let yi = y[i];
            let zi = x.row(i);
            let r = yi - zi.dot(&beta_arr);
            let di = d[i] as f64;
            let d_complement = 1.0 - di;
            for j in 0..p {
                m[[i, j]] = d_complement * zi[j] * r;
            }
            m[[i, p]] = di * (r - mu);
        }
        m
    };

    let mut target_grad = vec![0.0; p + 1];
    target_grad[p] = 1.0;
    let psi_vec =
        stacked_influence(&theta, moments, &cluster_full, &target_grad, 1e-6).unwrap_or_default();
    let se = se_from_influence(&psi_vec.iter().map(|(_, v)| *v).collect::<Vec<_>>());
    let estimate = inference_stats(est, se, z)?;
    let psi = psi_vec.into_iter().collect();

    Ok((estimate, psi))
}

fn scalar_avg_se(post_rows: &[&EventRow]) -> f64 {
    let h = post_rows.len() as f64;
    if h == 0.0 {
        return f64::NAN;
    }
    let mut psi_avg: HashMap<usize, f64> = HashMap::new();
    for row in post_rows {
        for (&cid, &psi) in &row.psi_by_cluster {
            *psi_avg.entry(cid).or_insert(0.0) += psi;
        }
    }
    for psi in psi_avg.values_mut() {
        *psi /= h;
    }
    let values: Vec<f64> = psi_avg.values().copied().collect();
    se_from_influence(&values)
}

fn inference_to_scalar(term: String, est: f64, se: f64, z: f64) -> ScalarRow {
    let e = inference_stats(est, se, z).unwrap_or(Estimate::nan());
    ScalarRow {
        term,
        estimate: e.estimate,
        se: e.se,
        t_stat: e.t_stat,
        p_value: e.p_value,
        ci_lower: e.ci_lower,
        ci_upper: e.ci_upper,
    }
}

fn inference_stats(est: f64, se: f64, z: f64) -> Result<Estimate, GreenersError> {
    if !est.is_finite() || !se.is_finite() || se <= 0.0 {
        return Ok(Estimate {
            estimate: est,
            se,
            t_stat: f64::NAN,
            p_value: f64::NAN,
            ci_lower: f64::NAN,
            ci_upper: f64::NAN,
        });
    }
    let t = est / se;
    let p = p_value_two_sided(t);
    Ok(Estimate {
        estimate: est,
        se,
        t_stat: t,
        p_value: p,
        ci_lower: est - z * se,
        ci_upper: est + z * se,
    })
}

// -----------------------------------------------------------------------------
// Stacked GMM influence functions (matches pylpdid _inference.py)
// -----------------------------------------------------------------------------

fn stacked_influence<F>(
    theta: &[f64],
    moments: F,
    cluster_ids: &[usize],
    target_grad: &[f64],
    eps: f64,
) -> Result<Vec<(usize, f64)>, GreenersError>
where
    F: Fn(&[f64]) -> Array2<f64>,
{
    let n = cluster_ids.len();
    let p = theta.len();
    if n == 0 || p == 0 {
        return Ok(Vec::new());
    }

    let m0 = moments(theta);
    let q = m0.ncols();
    if m0.nrows() != n {
        return Err(GreenersError::ShapeMismatch(
            "stacked_influence: moment matrix row count mismatch".into(),
        ));
    }

    // Mean moment function for numeric Jacobian.
    let mean_moment = |th: &[f64]| -> Array1<f64> {
        let m = moments(th);
        m.mean_axis(Axis(0))
            .unwrap_or_else(|| Array1::zeros(m.ncols()))
    };

    // A = dm / dtheta (q x p) via central differences.
    let mut a = Array2::<f64>::zeros((q, p));
    for j in 0..p {
        let step = eps * theta[j].abs().max(1.0);
        if step == 0.0 {
            continue;
        }
        let mut tp = theta.to_vec();
        tp[j] += step;
        let mut tm = theta.to_vec();
        tm[j] -= step;
        let gp = mean_moment(&tp);
        let gm = mean_moment(&tm);
        for i in 0..q {
            a[[i, j]] = (gp[i] - gm[i]) / (2.0 * step);
        }
    }

    // S = sum of moments by cluster (G x q).
    let mut cluster_pos: HashMap<usize, usize> = HashMap::new();
    let mut cluster_labels: Vec<usize> = Vec::new();
    let mut s_vec: Vec<f64> = Vec::new();
    for i in 0..n {
        let cid = cluster_ids[i];
        let pos = *cluster_pos.entry(cid).or_insert_with(|| {
            cluster_labels.push(cid);
            cluster_labels.len() - 1
        });
        if s_vec.len() < (pos + 1) * q {
            s_vec.resize((pos + 1) * q, 0.0);
        }
        for j in 0..q {
            s_vec[pos * q + j] += m0[[i, j]];
        }
    }
    let g = cluster_labels.len();
    let s = Array2::from_shape_vec((g, q), s_vec)
        .map_err(|e| GreenersError::ShapeMismatch(e.to_string()))?;

    // Pseudo-inverse of A (q x p) -> p x q.
    let ainv = a.pinv()?;
    let ainv_t = ainv.t();

    // Small-sample correction (Cameron-Miller style).
    let correction = if g <= 1 {
        1.0
    } else {
        (g as f64 / (g as f64 - 1.0)) * ((n as f64 - 1.0) / ((n - p) as f64).max(1.0))
    };
    let sqrt_corr = correction.sqrt();

    // v = Ainv' * target_grad (length q).
    let grad = Array1::from(target_grad.to_vec());
    let v = ainv_t.dot(&grad);

    // psi_g = sqrt_corr * S * v / n for each cluster.
    let mut psi = Vec::with_capacity(g);
    for r in 0..g {
        let mut acc = 0.0;
        for c in 0..q {
            acc += s[[r, c]] * v[c];
        }
        psi.push((cluster_labels[r], sqrt_corr * acc / n as f64));
    }

    Ok(psi)
}

fn se_from_influence(psi: &[f64]) -> f64 {
    if psi.iter().any(|&x| !x.is_finite()) {
        return f64::NAN;
    }
    psi.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

// -----------------------------------------------------------------------------
// Utility helpers
// -----------------------------------------------------------------------------

fn z_crit(alpha: f64) -> f64 {
    let normal = Normal::new(0.0, 1.0).expect("standard normal");
    normal.inverse_cdf(1.0 - alpha / 2.0)
}

fn p_value_two_sided(t: f64) -> f64 {
    if !t.is_finite() {
        return f64::NAN;
    }
    let normal = Normal::new(0.0, 1.0).expect("standard normal");
    2.0 * (1.0 - normal.cdf(t.abs()))
}

fn column_to_f64(col: &Column, name: &str) -> Result<Vec<f64>, GreenersError> {
    match col {
        Column::Float(arr) => Ok(arr.to_vec()),
        Column::Int(arr) => Ok(arr.iter().map(|&v| v as f64).collect()),
        Column::Bool(arr) => Ok(arr.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect()),
        Column::String(arr) => Ok(arr
            .iter()
            .map(|s| s.parse::<f64>().unwrap_or(f64::NAN))
            .collect()),
        Column::Categorical(cat) => Ok(cat
            .to_strings()
            .iter()
            .map(|s| s.parse::<f64>().unwrap_or(f64::NAN))
            .collect()),
        Column::DateTime(_) => Err(GreenersError::InvalidOperation(format!(
            "column '{}' is DateTime and cannot be converted to float",
            name
        ))),
    }
}

fn column_to_strings(col: &Column, _name: &str) -> Result<Vec<String>, GreenersError> {
    match col {
        Column::String(arr) => Ok(arr.to_vec()),
        Column::Categorical(cat) => Ok(cat.to_strings()),
        Column::Int(arr) => Ok(arr.iter().map(|&v| v.to_string()).collect()),
        Column::Float(arr) => Ok(arr.iter().map(|&v| v.to_string()).collect()),
        Column::Bool(arr) => Ok(arr.iter().map(|&v| v.to_string()).collect()),
        Column::DateTime(_) => Err(GreenersError::InvalidOperation(
            "DateTime column cannot be converted to strings".into(),
        )),
    }
}
