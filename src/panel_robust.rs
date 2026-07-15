//! Robust panel inference tests:
//!   1. Robust Hausman test (Cameron-Trivedi 2005; Wooldridge 2010)
//!   2. Robust F-test for panel significance (Wooldridge 2010)
//!
//! The standard Hausman test assumes RE variance estimator is
//! efficient under H0. When errors are heteroskedastic or
//! clustered, this fails. The robust version uses a cluster-
//! robust variance of (b_fe - b_re) instead of the simple
//! difference of diagonal variances.
//!
//! The robust F-test uses Wald statistics with cluster-robust
//! covariance to test joint significance of coefficients in
//! panel models.

use crate::linalg::LinalgInverse as _;
use crate::panel::{PanelResult, RandomEffectsResult};
use crate::GreenersError;
use ndarray::{Array1, Array2};
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor};
use std::fmt;

/// Result of robust Hausman test.
#[derive(Debug)]
pub struct RobustHausmanResult {
    /// Chi-squared statistic
    pub chi2: f64,
    /// Degrees of freedom
    pub df: usize,
    /// P-value
    pub p_value: f64,
    /// Difference of coefficients (b_fe - b_re)
    pub beta_diff: Array1<f64>,
    /// Robust covariance of the difference
    pub cov_diff: Array2<f64>,
    /// Whether to reject H0 (RE inconsistent)
    pub reject_h0: bool,
    /// Recommendation
    pub recommendation: String,
    /// Number of coefficients
    pub n_coef: usize,
    /// Method: "robust" or "classical"
    pub method: String,
}

impl fmt::Display for RobustHausmanResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Robust Hausman Test ")?;
        writeln!(f, "Cameron-Trivedi (2005); Wooldridge (2010)")?;
        writeln!(f, "H0: Random Effects is consistent")?;
        writeln!(f, "H1: Random Effects is inconsistent (use FE)")?;
        writeln!(f, "{:<20} {:>12}", "Method:", self.method)?;
        writeln!(f, "{:<20} {:>12}", "Coefficients:", self.n_coef)?;
        writeln!(f, "{:<20} {:>12}", "df:", self.df)?;
        writeln!(f, "{:<20} {:>12.4}", "Chi2:", self.chi2)?;
        writeln!(f, "{:<20} {:>12.4}", "P-value:", self.p_value)?;

        // Beta differences
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Coefficient differences (FE - RE):")?;
        writeln!(f, "  {:<10} {:>12}", "Coef", "FE - RE")?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..self.n_coef {
            writeln!(
                f,
                "  {:<10} {:>12.6}",
                format!("b{}", i + 1),
                self.beta_diff[i]
            )?;
        }

        writeln!(f, "\n  Result: {}", self.recommendation)?;

        write!(f, "{:=^78}", "")
    }
}

/// Result of robust F-test for panel.
#[derive(Debug)]
pub struct RobustFTestResult {
    /// Wald statistic (chi2 version)
    pub wald_chi2: f64,
    /// F-statistic (Wald / q)
    pub f_stat: f64,
    /// Numerator df
    pub df_num: usize,
    /// Denominator df
    pub df_denom: usize,
    /// P-value (from F distribution)
    pub p_value: f64,
    /// P-value (from chi2 distribution)
    pub p_value_chi2: f64,
    /// Whether to reject H0
    pub reject_h0: bool,
    /// Null hypothesis description
    pub h0: String,
    /// Coefficients being tested
    pub tested_coefs: Vec<String>,
    /// Number of restrictions
    pub n_restrictions: usize,
}

impl fmt::Display for RobustFTestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Robust F-Test (Panel) ")?;
        writeln!(f, "Wooldridge (2010)")?;
        writeln!(f, "H0: {}", self.h0)?;
        writeln!(f, "{:<20} {:>12}", "Restrictions:", self.n_restrictions)?;
        writeln!(f, "{:<20} {:>12}", "df (num):", self.df_num)?;
        writeln!(f, "{:<20} {:>12}", "df (denom):", self.df_denom)?;
        writeln!(f, "{:<20} {:>12.4}", "Wald Chi2:", self.wald_chi2)?;
        writeln!(f, "{:<20} {:>12.4}", "F-stat:", self.f_stat)?;
        writeln!(f, "{:<20} {:>12.4}", "P-value (F):", self.p_value)?;
        writeln!(f, "{:<20} {:>12.4}", "P-value (Chi2):", self.p_value_chi2)?;

        // Tested coefficients
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Tested coefficients:")?;
        for name in &self.tested_coefs {
            writeln!(f, "  - {}", name)?;
        }

        let verdict = if self.reject_h0 {
            "Reject H0: coefficients are jointly significant."
        } else {
            "Fail to reject H0: coefficients are not jointly significant."
        };
        writeln!(f, "\n  Result: {}", verdict)?;

        write!(f, "{:=^78}", "")
    }
}

pub struct RobustHausman;

impl RobustHausman {
    /// Robust Hausman test comparing FE vs RE.
    ///
    /// Uses the robust covariance matrix of (b_fe - b_re) rather
    /// than the simple difference of variances. This is valid
    /// under heteroskedasticity and clustering.
    ///
    /// # Arguments
    /// * `fe` - Fixed Effects result
    /// * `re` - Random Effects result
    /// * `fe_vcov` - Robust covariance matrix of FE (k x k)
    /// * `re_vcov` - Robust covariance matrix of RE (k x k)
    pub fn compare(
        fe: &PanelResult,
        re: &RandomEffectsResult,
        fe_vcov: &Array2<f64>,
        re_vcov: &Array2<f64>,
    ) -> Result<RobustHausmanResult, GreenersError> {
        Self::compare_arrays(
            &fe.params,
            &re.params,
            fe_vcov,
            re_vcov,
            fe.variable_names.as_deref(),
        )
    }

    /// Compare using raw arrays (avoids needing to construct
    /// full PanelResult/RandomEffectsResult when aligning).
    pub fn compare_arrays(
        fe_beta: &Array1<f64>,
        re_beta: &Array1<f64>,
        fe_vcov: &Array2<f64>,
        re_vcov: &Array2<f64>,
        _var_names: Option<&[String]>,
    ) -> Result<RobustHausmanResult, GreenersError> {
        let k = fe_beta.len();
        if re_beta.len() != k {
            return Err(GreenersError::ShapeMismatch(
                "RobustHausman: FE and RE must have same number of params".into(),
            ));
        }
        if fe_vcov.nrows() != k || fe_vcov.ncols() != k {
            return Err(GreenersError::ShapeMismatch(
                "RobustHausman: fe_vcov must be k x k".into(),
            ));
        }
        if re_vcov.nrows() != k || re_vcov.ncols() != k {
            return Err(GreenersError::ShapeMismatch(
                "RobustHausman: re_vcov must be k x k".into(),
            ));
        }

        // Beta difference
        let beta_diff = fe_beta - re_beta;

        // Robust variance of difference: V(fe) - V(re)
        let cov_diff = fe_vcov - re_vcov;

        // Wald statistic: b_diff' * V_diff^{-1} * b_diff
        let cov_diff_inv = cov_diff.inv()?;
        let wald = beta_diff.dot(&cov_diff_inv.dot(&beta_diff));

        // Handle negative chi2 (can happen if cov_diff is not PSD)
        let chi2 = wald.max(0.0);

        // P-value
        let dist = ChiSquared::new(k as f64)
            .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_value = 1.0 - dist.cdf(chi2);

        let reject_h0 = p_value < 0.05;
        let recommendation = if reject_h0 {
            "Reject H0. Use FIXED EFFECTS (RE is inconsistent)."
        } else {
            "Fail to reject H0. Use RANDOM EFFECTS (it is efficient)."
        };

        Ok(RobustHausmanResult {
            chi2,
            df: k,
            p_value,
            beta_diff,
            cov_diff,
            reject_h0,
            recommendation: recommendation.to_string(),
            n_coef: k,
            method: "robust".to_string(),
        })
    }

    /// Classical Hausman test (using diagonal variances only).
    /// Falls back to the simple version when full covariance
    /// matrices are not available.
    pub fn classical(
        fe: &PanelResult,
        re: &RandomEffectsResult,
    ) -> Result<RobustHausmanResult, GreenersError> {
        let k = fe.params.len();
        let beta_diff = &fe.params - &re.params;

        let var_fe = fe.std_errors.mapv(|s| s.powi(2));
        let var_re = re.std_errors.mapv(|s| s.powi(2));
        let diff_var = &var_fe - &var_re;

        let mut chi2 = 0.0;
        for i in 0..k {
            if diff_var[i] > 0.0 {
                chi2 += beta_diff[i].powi(2) / diff_var[i];
            }
        }

        let dist = ChiSquared::new(k as f64)
            .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_value = 1.0 - dist.cdf(chi2);

        // Build diagonal cov_diff for output
        let mut cov_diff = Array2::zeros((k, k));
        for i in 0..k {
            cov_diff[(i, i)] = diff_var[i];
        }

        let reject_h0 = p_value < 0.05;
        let recommendation = if reject_h0 {
            "Reject H0. Use FIXED EFFECTS (RE is inconsistent)."
        } else {
            "Fail to reject H0. Use RANDOM EFFECTS (it is efficient)."
        };

        Ok(RobustHausmanResult {
            chi2,
            df: k,
            p_value,
            beta_diff,
            cov_diff,
            reject_h0,
            recommendation: recommendation.to_string(),
            n_coef: k,
            method: "classical".to_string(),
        })
    }
}

pub struct RobustFTest;

impl RobustFTest {
    /// Robust Wald F-test for joint significance of coefficients.
    ///
    /// Tests H0: beta_{i1} = beta_{i2} = ... = beta_{iq} = 0
    /// using a robust covariance matrix.
    ///
    /// # Arguments
    /// * `beta` - Full coefficient vector (p)
    /// * `vcov` - Robust covariance matrix (p x p)
    /// * `indices` - Indices of coefficients to test (0-based)
    /// * `coef_names` - Optional names of all coefficients
    /// * `n` - Number of observations (for F-test df)
    pub fn test(
        beta: &Array1<f64>,
        vcov: &Array2<f64>,
        indices: &[usize],
        coef_names: Option<&[String]>,
        n: usize,
    ) -> Result<RobustFTestResult, GreenersError> {
        let p = beta.len();
        let q = indices.len();
        if q == 0 {
            return Err(GreenersError::InvalidOperation(
                "RobustFTest: need at least 1 coefficient to test".into(),
            ));
        }
        if vcov.nrows() != p || vcov.ncols() != p {
            return Err(GreenersError::ShapeMismatch(
                "RobustFTest: vcov must be p x p".into(),
            ));
        }
        for &idx in indices {
            if idx >= p {
                return Err(GreenersError::InvalidOperation(format!(
                    "RobustFTest: index {idx} out of range (p={p})"
                )));
            }
        }

        // Extract subvector and submatrix
        let mut beta_r = Array1::zeros(q);
        let mut vcov_r = Array2::zeros((q, q));
        for (i, &ri) in indices.iter().enumerate() {
            beta_r[i] = beta[ri];
            for (j, &rj) in indices.iter().enumerate() {
                vcov_r[(i, j)] = vcov[(ri, rj)];
            }
        }

        // Wald statistic: beta_r' * V_r^{-1} * beta_r
        let vcov_r_inv = vcov_r.inv()?;
        let wald = beta_r.dot(&vcov_r_inv.dot(&beta_r));

        // F-statistic: Wald / q
        let f_stat = wald / q as f64;

        // Degrees of freedom
        let df_num = q;
        let df_denom = n.saturating_sub(p);

        // P-value from F distribution
        let f_dist = FisherSnedecor::new(df_num as f64, df_denom as f64)
            .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_value = 1.0 - f_dist.cdf(f_stat);

        // P-value from chi2 (asymptotic)
        let chi2_dist = ChiSquared::new(q as f64)
            .map_err(|e| GreenersError::InvalidOperation(e.to_string()))?;
        let p_value_chi2 = 1.0 - chi2_dist.cdf(wald);

        let reject_h0 = p_value < 0.05;

        let tested_coefs: Vec<String> = match coef_names {
            Some(names) => indices.iter().map(|&i| names[i].clone()).collect(),
            None => indices.iter().map(|&i| format!("x{}", i + 1)).collect(),
        };

        let h0 = format!(
            "beta_{} = 0 (jointly)",
            tested_coefs.join(" = beta_") + " = 0"
        );

        Ok(RobustFTestResult {
            wald_chi2: wald,
            f_stat,
            df_num,
            df_denom,
            p_value,
            p_value_chi2,
            reject_h0,
            h0,
            tested_coefs,
            n_restrictions: q,
        })
    }

    /// Test joint significance of ALL slope coefficients
    /// (excluding intercept).
    pub fn test_all_slopes(
        beta: &Array1<f64>,
        vcov: &Array2<f64>,
        coef_names: Option<&[String]>,
        n: usize,
    ) -> Result<RobustFTestResult, GreenersError> {
        let p = beta.len();
        if p < 2 {
            return Err(GreenersError::InvalidOperation(
                "RobustFTest: need at least 2 coefficients".into(),
            ));
        }
        // Test all except first (intercept)
        let indices: Vec<usize> = (1..p).collect();
        Self::test(beta, vcov, &indices, coef_names, n)
    }
}
