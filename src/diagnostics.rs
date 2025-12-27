use crate::error::GreenersError;
use crate::CovarianceType; // Needed to call OLS fit
use crate::OLS; // We reuse OLS for the Breusch-Pagan auxiliary regression
use ndarray::{Array1, Array2};
use ndarray_linalg::{Inverse, SVD};
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

    /// Variance Inflation Factor (VIF) for each predictor.
    ///
    /// VIF measures how much the variance of an estimated regression coefficient
    /// increases due to multicollinearity. For variable j:
    ///
    /// VIF_j = 1 / (1 - R²_j)
    ///
    /// where R²_j is the R-squared from regressing X_j on all other predictors.
    ///
    /// **Interpretation:**
    /// - VIF = 1: No correlation with other predictors
    /// - VIF < 5: Acceptable multicollinearity
    /// - VIF 5-10: Moderate multicollinearity (caution needed)
    /// - VIF > 10: High multicollinearity (problematic)
    ///
    /// **Note:** If X includes an intercept column (all 1s), VIF is undefined
    /// for that column. This function returns NaN for constant columns.
    ///
    /// # Arguments
    /// * `x` - Design matrix (n × k), typically including intercept
    ///
    /// # Returns
    /// Array of VIF values for each column. Intercept column will have VIF = NaN.
    pub fn vif(x: &Array2<f64>) -> Result<Array1<f64>, GreenersError> {
        let k = x.ncols();
        let mut vif_values = Array1::<f64>::zeros(k);

        for j in 0..k {
            // Check if column is constant (e.g., intercept)
            let col_j = x.column(j);
            let col_mean = col_j.mean().unwrap_or(0.0);
            let col_var = col_j.mapv(|v| (v - col_mean).powi(2)).sum();

            if col_var < 1e-12 {
                // Constant column (likely intercept) - VIF is undefined
                vif_values[j] = f64::NAN;
                continue;
            }

            // Regress X_j on all other X columns
            // Build X_{-j} (all columns except j)
            let mut x_minus_j_cols = Vec::new();
            for i in 0..k {
                if i != j {
                    x_minus_j_cols.push(x.column(i).to_owned());
                }
            }

            if x_minus_j_cols.is_empty() {
                // Only one predictor - VIF = 1
                vif_values[j] = 1.0;
                continue;
            }

            // Stack columns to create X_{-j}
            let n = x.nrows();
            let mut x_minus_j = Array2::<f64>::zeros((n, x_minus_j_cols.len()));
            for (col_idx, col_data) in x_minus_j_cols.iter().enumerate() {
                x_minus_j.column_mut(col_idx).assign(col_data);
            }

            // Run auxiliary regression: X_j = X_{-j} * beta + error
            let y_j = col_j.to_owned();
            match OLS::fit(&y_j, &x_minus_j, CovarianceType::NonRobust) {
                Ok(aux_result) => {
                    let r_squared = aux_result.r_squared;

                    // VIF = 1 / (1 - R²)
                    // Protection: If R² ≈ 1, VIF → ∞
                    if r_squared >= 0.9999 {
                        vif_values[j] = f64::INFINITY;
                    } else {
                        vif_values[j] = 1.0 / (1.0 - r_squared);
                    }
                }
                Err(_) => {
                    // If regression fails (e.g., perfect collinearity), set VIF to infinity
                    vif_values[j] = f64::INFINITY;
                }
            }
        }

        Ok(vif_values)
    }

    /// Leverage values (diagonal elements of hat matrix H = X(X'X)^-1X').
    ///
    /// Leverage measures how far an observation's predictor values are from
    /// the mean of the predictor values. High leverage points have the potential
    /// to be influential.
    ///
    /// **Interpretation:**
    /// - Average leverage: h̄ = k/n (where k = number of parameters, n = observations)
    /// - High leverage threshold: h_i > 2k/n or h_i > 3k/n
    /// - Range: 0 ≤ h_i ≤ 1
    ///
    /// **Note:** High leverage alone doesn't mean the point is influential.
    /// Use Cook's distance to identify truly influential observations.
    ///
    /// # Arguments
    /// * `x` - Design matrix (n × k)
    ///
    /// # Returns
    /// Array of leverage values (one per observation)
    pub fn leverage(x: &Array2<f64>) -> Result<Array1<f64>, GreenersError> {
        let x_t = x.t();
        let xtx = x_t.dot(x);
        let xtx_inv = xtx.inv()?;

        // H = X(X'X)^-1X'
        // We only need diagonal elements: h_i = x_i' (X'X)^-1 x_i
        let n = x.nrows();
        let mut h_values = Array1::<f64>::zeros(n);

        for i in 0..n {
            let x_i = x.row(i);
            // h_i = x_i' * (X'X)^-1 * x_i
            let temp = xtx_inv.dot(&x_i);
            h_values[i] = x_i.dot(&temp);
        }

        Ok(h_values)
    }

    /// Cook's Distance for detecting influential observations.
    ///
    /// Cook's D measures the influence of each observation on the fitted values.
    /// It combines leverage and residual size to identify observations that
    /// significantly affect the regression results.
    ///
    /// Formula: D_i = (e_i² / (k * MSE)) * (h_i / (1 - h_i)²)
    ///
    /// where:
    /// - e_i = residual for observation i
    /// - k = number of parameters (including intercept)
    /// - MSE = mean squared error
    /// - h_i = leverage for observation i
    ///
    /// **Interpretation:**
    /// - D_i > 1: Highly influential (investigate!)
    /// - D_i > 4/n: Potentially influential (common threshold)
    /// - D_i > 0.5: Worth examining
    ///
    /// **Rule of thumb:** D_i > 4/(n-k-1) suggests influence
    ///
    /// # Arguments
    /// * `residuals` - Residuals from the regression
    /// * `x` - Design matrix (n × k)
    /// * `mse` - Mean squared error (σ²)
    ///
    /// # Returns
    /// Array of Cook's distances (one per observation)
    pub fn cooks_distance(
        residuals: &Array1<f64>,
        x: &Array2<f64>,
        mse: f64,
    ) -> Result<Array1<f64>, GreenersError> {
        let n = residuals.len();
        let k = x.ncols();

        // Get leverage values
        let h_values = Self::leverage(x)?;

        let mut cook_d = Array1::<f64>::zeros(n);

        for i in 0..n {
            let e_i = residuals[i];
            let h_i = h_values[i];

            // Protect against h_i = 1 (would cause division by zero)
            if h_i >= 0.9999 {
                cook_d[i] = f64::INFINITY;
                continue;
            }

            // D_i = (e_i² / (k * MSE)) * (h_i / (1 - h_i)²)
            let numerator = e_i.powi(2) * h_i;
            let denominator = (k as f64) * mse * (1.0 - h_i).powi(2);

            cook_d[i] = numerator / denominator;
        }

        Ok(cook_d)
    }

    /// Condition Number of the design matrix.
    ///
    /// The condition number measures multicollinearity by computing the ratio
    /// of the largest to smallest singular value of X:
    ///
    /// κ(X) = σ_max / σ_min
    ///
    /// **Interpretation:**
    /// - κ < 10: No multicollinearity
    /// - κ 10-30: Moderate multicollinearity
    /// - κ 30-100: Strong multicollinearity (caution)
    /// - κ > 100: Severe multicollinearity (problematic)
    ///
    /// **Advantage over VIF:** Single number summarizing overall collinearity
    ///
    /// **Note:** Automatically handles intercept and scaling issues via SVD.
    ///
    /// # Arguments
    /// * `x` - Design matrix (n × k)
    ///
    /// # Returns
    /// Condition number (scalar)
    pub fn condition_number(x: &Array2<f64>) -> Result<f64, GreenersError> {
        // Use Singular Value Decomposition to get singular values
        let (_u, s, _vt) = x.svd(false, false)?;

        // Condition number = max(σ) / min(σ)
        let sigma_max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sigma_min = s.iter().cloned().fold(f64::INFINITY, f64::min);

        if sigma_min < 1e-12 {
            // Near-singular matrix
            Ok(f64::INFINITY)
        } else {
            Ok(sigma_max / sigma_min)
        }
    }
}
