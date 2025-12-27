use crate::{CovarianceType, GreenersError, OLS, DataFrame, Formula};
use ndarray::{Array1, Array2};
use std::fmt;

/// Result of the Difference-in-Differences estimator (Canonical 2x2)
#[derive(Debug)]
pub struct DidResult {
    pub att: f64,       // The treatment effect (Interaction coefficient)
    pub std_error: f64, // ATT standard error
    pub t_stat: f64,
    pub p_value: f64,
    pub n_obs: usize,
    pub r_squared: f64,
    pub control_pre_mean: f64,  // Control Mean (Pre)
    pub control_post_mean: f64, // Control Mean (Post)
    pub treated_pre_mean: f64,  // Treated Mean (Pre)
    pub treated_post_mean: f64, // Treated Mean (Post - Counterfactual vs Real)
    pub cov_type: CovarianceType,
}

impl fmt::Display for DidResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            " Difference-in-Differences (2x2 Canonical) "
        )?;
        writeln!(
            f,
            "{:<20} {:>15.4} || {:<20} {:>15.4}",
            "ATT (Effect):", self.att, "R-squared:", self.r_squared
        )?;
        writeln!(
            f,
            "{:<20} {:>15.4} || {:<20} {:>15.4}",
            "Std. Error:", self.std_error, "P-value:", self.p_value
        )?;
        writeln!(
            f,
            "{:<20} {:>15.4} || {:<20} {:>15}",
            "t-statistic:", self.t_stat, "Observations:", self.n_obs
        )?;

        writeln!(f, "\n{:-^78}", " Group Means ")?;
        writeln!(
            f,
            "Control Group (Pre):  {:>10.4} | Control Group (Post): {:>10.4}",
            self.control_pre_mean, self.control_post_mean
        )?;
        writeln!(
            f,
            "Treated Group (Pre):  {:>10.4} | Treated Group (Post): {:>10.4}",
            self.treated_pre_mean, self.treated_post_mean
        )?;
        writeln!(
            f,
            "Parallel Trend Diff:  {:>10.4} (If > 0, Control grew more than Treated Pre-trend)",
            (self.control_post_mean - self.control_pre_mean)
                - (self.treated_post_mean - self.treated_pre_mean - self.att)
        )?;
        writeln!(f, "{:=^78}", "")
    }
}

pub struct DiffInDiff;

impl DiffInDiff {
    /// Estimates DiD model using a formula and DataFrame.
    ///
    /// The formula should specify the outcome variable and include 'treated' and 'post' variables.
    /// The interaction term is created automatically.
    ///
    /// # Examples
    /// ```no_run
    /// use greeners::{DiffInDiff, DataFrame, Formula, CovarianceType};
    /// use ndarray::Array1;
    /// use std::collections::HashMap;
    ///
    /// let mut data = HashMap::new();
    /// data.insert("y".to_string(), Array1::from(vec![1.0, 2.0, 3.0, 4.0]));
    /// data.insert("treated".to_string(), Array1::from(vec![0.0, 0.0, 1.0, 1.0]));
    /// data.insert("post".to_string(), Array1::from(vec![0.0, 1.0, 0.0, 1.0]));
    ///
    /// let df = DataFrame::new(data).unwrap();
    /// let formula = Formula::parse("y ~ treated + post").unwrap();
    ///
    /// let result = DiffInDiff::from_formula(&formula, &df, "treated", "post", CovarianceType::HC1).unwrap();
    /// ```
    pub fn from_formula(
        formula: &Formula,
        data: &DataFrame,
        treated_var: &str,
        post_var: &str,
        cov_type: CovarianceType,
    ) -> Result<DidResult, GreenersError> {
        // Extract y from formula
        let y = data.get(&formula.dependent)?;

        // Extract treated and post variables
        let treated = data.get(treated_var)?;
        let post = data.get(post_var)?;

        Self::fit(y, treated, post, cov_type)
    }

    /// Estimates the Canonical 2x2 DiD model.
    ///
    /// # Arguments
    /// * `y` - Outcome variable.
    /// * `treated` - Dummy: 1 if belongs to treatment group, 0 otherwise.
    /// * `post` - Dummy: 1 if in post-intervention period, 0 otherwise.
    /// * `cov_type` - Covariance type (Recommended: HC1 or Cluster if we had cluster ID).
    pub fn fit(
        y: &Array1<f64>,
        treated: &Array1<f64>,
        post: &Array1<f64>,
        cov_type: CovarianceType,
    ) -> Result<DidResult, GreenersError> {
        let n = y.len();
        if treated.len() != n || post.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "Input arrays must have same length".into(),
            ));
        }

        // 1. Build Matrix X [Intercept, Treated, Post, Interaction]
        let mut x_mat = Array2::<f64>::zeros((n, 4));
        let mut interaction = Array1::<f64>::zeros(n);

        // Means for display (Manual calculation for performance)
        let mut sum_c_pre = 0.0;
        let mut n_c_pre = 0.0;
        let mut sum_c_post = 0.0;
        let mut n_c_post = 0.0;
        let mut sum_t_pre = 0.0;
        let mut n_t_pre = 0.0;
        let mut sum_t_post = 0.0;
        let mut n_t_post = 0.0;

        for i in 0..n {
            let t = treated[i];
            let p = post[i];
            let inter = t * p; // D * T

            x_mat[[i, 0]] = 1.0; // Beta0
            x_mat[[i, 1]] = t; // Beta1 (Group Fixed Effect)
            x_mat[[i, 2]] = p; // Beta2 (Time Fixed Effect)
            x_mat[[i, 3]] = inter; // Delta (ATT)

            interaction[i] = inter;

            // Accumulate means
            let val = y[i];
            if t == 0.0 && p == 0.0 {
                sum_c_pre += val;
                n_c_pre += 1.0;
            } else if t == 0.0 && p == 1.0 {
                sum_c_post += val;
                n_c_post += 1.0;
            } else if t == 1.0 && p == 0.0 {
                sum_t_pre += val;
                n_t_pre += 1.0;
            } else if t == 1.0 && p == 1.0 {
                sum_t_post += val;
                n_t_post += 1.0;
            }
        }

        // 2. Run OLS
        let ols = OLS::fit(y, &x_mat, cov_type.clone())?;

        // The ATT is the interaction coefficient (index 3)
        let att = ols.params[3];
        let std_error = ols.std_errors[3];
        let t_stat = ols.t_values[3];
        let p_value = ols.p_values[3];

        Ok(DidResult {
            att,
            std_error,
            t_stat,
            p_value,
            n_obs: n,
            r_squared: ols.r_squared,
            control_pre_mean: if n_c_pre > 0.0 {
                sum_c_pre / n_c_pre
            } else {
                0.0
            },
            control_post_mean: if n_c_post > 0.0 {
                sum_c_post / n_c_post
            } else {
                0.0
            },
            treated_pre_mean: if n_t_pre > 0.0 {
                sum_t_pre / n_t_pre
            } else {
                0.0
            },
            treated_post_mean: if n_t_post > 0.0 {
                sum_t_post / n_t_post
            } else {
                0.0
            },
            cov_type,
        })
    }
}
