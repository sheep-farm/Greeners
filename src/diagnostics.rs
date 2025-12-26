use crate::error::GreenersError;
use crate::CovarianceType; // Needed to call OLS fit
use crate::OLS; // We reuse OLS for the Breusch-Pagan auxiliary regression
use ndarray::{Array1, Array2};
use statrs::distribution::{ChiSquared, ContinuousCDF};

pub struct Diagnostics;

impl Diagnostics {
    /// Jarque-Bera test for Normality of Residuals.
    /// H0: Residuals are normally distributed.
    ///
    /// Returns: (JB-Statistic, p-value)
    pub fn jarque_bera(residuals: &Array1<f64>) -> Result<(f64, f64), GreenersError> {
        let n = residuals.len() as f64;
        let mean = residuals.mean().unwrap_or(0.0);

        // Calculate Central Moments
        let m2 = residuals.mapv(|r| (r - mean).powi(2)).sum() / n;
        let m3 = residuals.mapv(|r| (r - mean).powi(3)).sum() / n;
        let m4 = residuals.mapv(|r| (r - mean).powi(4)).sum() / n;

        // Skewness (S) and Kurtosis (K)
        let skewness = m3 / m2.powf(1.5);
        let kurtosis = m4 / m2.powi(2);

        // JB = (n/6) * (S^2 + (K - 3)^2 / 4)
        let jb_stat = (n / 6.0) * (skewness.powi(2) + (kurtosis - 3.0).powi(2) / 4.0);

        // Chi-Square Distribution with 2 degrees of freedom
        let chi2 = ChiSquared::new(2.0).map_err(|_| GreenersError::OptimizationFailed)?;
        let p_value = 1.0 - chi2.cdf(jb_stat);

        Ok((jb_stat, p_value))
    }

    /// Breusch-Pagan test for Heteroskedasticity.
    /// H0: Homoskedasticity (Variance is constant).
    ///
    /// Steps:
    /// 1. Get squared residuals (u^2).
    /// 2. Run auxiliary regression: u^2 = alpha + delta*X + error.
    /// 3. LM Statistic = n * R_squared_aux.
    ///
    /// Returns: (LM-Statistic, p-value)
    pub fn breusch_pagan(
        residuals: &Array1<f64>,
        x: &Array2<f64>,
    ) -> Result<(f64, f64), GreenersError> {
        let n = residuals.len() as f64;

        // 1. Auxiliary dependent variable: squared residuals
        let u_sq = residuals.mapv(|x| x.powi(2));

        // 2. Auxiliary Regression: u^2 against X
        // We use CovarianceType::NonRobust because we only want the R2
        let aux_model = OLS::fit(&u_sq, x, CovarianceType::NonRobust)?;

        // 3. Lagrange Multiplier Statistic = n * R2
        let lm_stat = n * aux_model.r_squared;

        // Degrees of freedom = k (number of regressors in auxiliary, excluding constant if any, but here we simplify to k-1 if intercept)
        // The correct BP is df = number of exogenous variables causing variance.
        // Assuming X has intercept and we want to test the variables:
        let df = (x.ncols() - 1) as f64;

        // Protection for case X has only intercept or df <= 0
        let df_safe = if df <= 0.0 { 1.0 } else { df };

        let chi2 = ChiSquared::new(df_safe).map_err(|_| GreenersError::OptimizationFailed)?;
        let p_value = 1.0 - chi2.cdf(lm_stat);

        Ok((lm_stat, p_value))
    }

    /// Durbin-Watson Test for Autocorrelation of Residuals.
    /// Range: [0, 4].
    /// - 2.0: No autocorrelation.
    /// - 0 to <2: Positive autocorrelation (Common in time series).
    /// - >2 to 4: Negative autocorrelation.
    pub fn durbin_watson(residuals: &Array1<f64>) -> f64 {
        let n = residuals.len();
        if n < 2 {
            return 0.0;
        }

        let mut numerator = 0.0;
        // Sum of squared differences: sum((e_t - e_{t-1})^2)
        for t in 1..n {
            let diff = residuals[t] - residuals[t - 1];
            numerator += diff.powi(2);
        }

        // Sum of squared residuals: sum(e_t^2)
        let denominator = residuals.mapv(|x| x.powi(2)).sum();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}
