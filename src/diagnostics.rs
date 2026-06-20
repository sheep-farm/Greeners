use crate::error::GreenersError;
use crate::linalg::{LinalgInverse as _, LinalgSVD as _};
use crate::CovarianceType; // Needed to call OLS fit
use crate::OLS; // We reuse OLS for the Breusch-Pagan auxiliary regression
use ndarray::{Array1, Array2};
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Result of Ljung-Box portmanteau test.
#[derive(Debug)]
pub struct LjungBoxResult {
    pub q_stat: f64,
    pub p_value: f64,
    pub lags: usize,
    pub n_obs: usize,
    /// Sample autocorrelations at each lag (1..=lags)
    pub acf: Vec<f64>,
}

/// Result of Engle's ARCH LM test.
#[derive(Debug)]
pub struct ArchTestResult {
    pub lm_stat: f64,
    pub lm_pvalue: f64,
    pub f_stat: f64,
    pub f_pvalue: f64,
    pub lags: usize,
    pub n_obs: usize,
    pub r_squared: f64,
}

/// Result of Anderson-Darling normality test.
#[derive(Debug)]
pub struct AndersonDarlingResult {
    pub statistic: f64,
    /// Critical values at [15%, 10%, 5%, 2.5%, 1%]
    pub critical_values: [f64; 5],
    pub significance_levels: [f64; 5],
    pub n_obs: usize,
}

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
    /// D'Agostino-Pearson omnibus test for normality.
    ///
    /// Combines skewness and kurtosis z-scores: K^2 = Z1^2 + Z2^2 ~ chi2(2).
    /// More powerful than Jarque-Bera for small samples.
    ///
    /// Returns: (omnibus-statistic, p-value)
    pub fn omnibus(residuals: &Array1<f64>) -> Result<(f64, f64), GreenersError> {
        let n = residuals.len() as f64;
        if n < 20.0 {
            return Err(GreenersError::ShapeMismatch(
                "Omnibus test requires at least 20 observations".into(),
            ));
        }

        let mean = residuals.mean().unwrap_or(0.0);
        let m2 = residuals.mapv(|r| (r - mean).powi(2)).sum() / n;
        let m3 = residuals.mapv(|r| (r - mean).powi(3)).sum() / n;
        let m4 = residuals.mapv(|r| (r - mean).powi(4)).sum() / n;

        let skewness = m3 / m2.powf(1.5);
        let kurtosis = m4 / m2.powi(2);

        // D'Agostino skewness z-score
        let y = skewness * ((n + 1.0) * (n + 3.0) / (6.0 * (n - 2.0))).sqrt();
        let beta2_s = 3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0)
            / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0));
        let w2 = (2.0 * (beta2_s - 1.0)).sqrt() - 1.0;
        let _w = w2.max(1e-10).sqrt();
        let delta = 1.0 / (0.5 * w2.max(1e-10).ln()).sqrt();
        let alpha_s = (2.0 / (w2 - 1.0)).max(1e-10).sqrt();
        let z1 = delta * (y / alpha_s + ((y / alpha_s).powi(2) + 1.0).sqrt()).ln();

        // D'Agostino kurtosis z-score
        let e_k = 3.0 * (n - 1.0) / (n + 1.0);
        let var_k = 24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0).powi(2) * (n + 3.0) * (n + 5.0));
        let x_k = (kurtosis - e_k) / var_k.max(1e-10).sqrt();

        let beta1 = 6.0 * (n * n - 5.0 * n + 2.0) / ((n + 7.0) * (n + 9.0))
            * (6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0))).sqrt();
        let a = 6.0 + 8.0 / beta1 * (2.0 / beta1 + (1.0 + 4.0 / (beta1 * beta1)).sqrt());
        let z2 = ((1.0 - 2.0 / (9.0 * a))
            - ((1.0 - 2.0 / a) / (1.0 + x_k * (2.0 / (a - 4.0)).max(1e-10).sqrt()))
                .powf(1.0 / 3.0))
            / (2.0 / (9.0 * a)).sqrt();

        let k2 = z1 * z1 + z2 * z2;

        let chi2 = ChiSquared::new(2.0).map_err(|_| GreenersError::OptimizationFailed)?;
        let p_value = 1.0 - chi2.cdf(k2);

        Ok((k2, p_value))
    }

    /// Harvey-Collier test for linearity.
    ///
    /// Performs a t-test on the mean of recursive residuals.
    /// H0: Linear specification is correct.
    /// Returns (t_statistic, p_value).
    pub fn harvey_collier(y: &Array1<f64>, x: &Array2<f64>) -> Result<(f64, f64), GreenersError> {
        let n = y.len();
        let k = x.ncols();
        if n <= k + 1 {
            return Err(GreenersError::ShapeMismatch(
                "Not enough observations for Harvey-Collier test".into(),
            ));
        }

        // Compute recursive residuals using expanding window OLS
        let mut rec_resids = Vec::new();
        for t in k..n {
            let x_t = x.slice(ndarray::s![..t, ..]).to_owned();
            let y_t = y.slice(ndarray::s![..t]).to_owned();

            if let Ok(ols_res) = OLS::fit(&y_t, &x_t, CovarianceType::NonRobust) {
                // One-step ahead forecast error
                let x_new = x.row(t);
                let y_hat = x_new.dot(&ols_res.params);
                let resid = y[t] - y_hat;

                // Scaling factor: 1 + x_{t+1}' (X'X)^-1 x_{t+1}
                let xtx = x_t.t().dot(&x_t);
                if let Ok(xtx_inv) = xtx.inv() {
                    let h = 1.0 + x_new.dot(&xtx_inv.dot(&x_new));
                    let scaled = resid / h.sqrt().max(1e-10);
                    rec_resids.push(scaled);
                }
            }
        }

        if rec_resids.len() < 3 {
            return Err(GreenersError::InvalidOperation(
                "Not enough recursive residuals".into(),
            ));
        }

        let m = rec_resids.len();
        let mean = rec_resids.iter().sum::<f64>() / m as f64;
        let var = rec_resids.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (m - 1) as f64;
        let se = (var / m as f64).sqrt();

        if se < 1e-15 {
            return Ok((0.0, 1.0));
        }

        let t_stat = mean / se;
        let df = (m - 1) as f64;
        let dist = statrs::distribution::StudentsT::new(0.0, 1.0, df)
            .map_err(|_| GreenersError::OptimizationFailed)?;
        let p_value = 2.0 * (1.0 - dist.cdf(t_stat.abs()));

        Ok((t_stat, p_value))
    }

    /// Anderson-Darling test for normality.
    ///
    /// Returns `AndersonDarlingResult` with test statistic and critical values
    /// at 15%, 10%, 5%, 2.5%, 1% significance levels.
    pub fn anderson_darling(data: &Array1<f64>) -> Result<AndersonDarlingResult, GreenersError> {
        let n = data.len();
        if n < 8 {
            return Err(GreenersError::ShapeMismatch(
                "Need at least 8 observations for Anderson-Darling test".into(),
            ));
        }

        let mean = data.mean().unwrap_or(0.0);
        let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std = var.sqrt();

        if std < 1e-15 {
            return Err(GreenersError::InvalidOperation("Zero variance data".into()));
        }

        // Standardize and sort
        let mut z: Vec<f64> = data.iter().map(|&x| (x - mean) / std).collect();
        z.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let normal = statrs::distribution::Normal::new(0.0, 1.0)
            .map_err(|_| GreenersError::OptimizationFailed)?;

        // A² = -n - (1/n) Σ (2i-1)[ln(Φ(z_i)) + ln(1-Φ(z_{n+1-i}))]
        let nf = n as f64;
        let mut sum = 0.0;
        for i in 0..n {
            let phi_i = normal.cdf(z[i]).clamp(1e-15, 1.0 - 1e-15);
            let phi_ni = normal.cdf(z[n - 1 - i]).clamp(1e-15, 1.0 - 1e-15);
            sum += (2 * i + 1) as f64 * (phi_i.ln() + (1.0 - phi_ni).ln());
        }
        let a2 = -nf - sum / nf;

        // Adjusted statistic for finite sample
        let a2_adj = a2 * (1.0 + 0.75 / nf + 2.25 / (nf * nf));

        // Critical values for normal distribution: 15%, 10%, 5%, 2.5%, 1%
        let critical_values = [0.576, 0.656, 0.787, 0.918, 1.092];

        Ok(AndersonDarlingResult {
            statistic: a2_adj,
            critical_values,
            significance_levels: [0.15, 0.10, 0.05, 0.025, 0.01],
            n_obs: n,
        })
    }

    /// Lilliefors test for normality.
    ///
    /// Kolmogorov-Smirnov test with estimated mean and variance.
    /// Returns (statistic, p_value).
    pub fn lilliefors(data: &Array1<f64>) -> Result<(f64, f64), GreenersError> {
        let n = data.len();
        if n < 4 {
            return Err(GreenersError::ShapeMismatch(
                "Need at least 4 observations for Lilliefors test".into(),
            ));
        }

        let mean = data.mean().unwrap_or(0.0);
        let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std = var.sqrt();

        if std < 1e-15 {
            return Ok((0.0, 1.0));
        }

        // Sort data
        let mut sorted: Vec<f64> = data.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let normal = statrs::distribution::Normal::new(0.0, 1.0)
            .map_err(|_| GreenersError::OptimizationFailed)?;

        // KS statistic: max |F_n(x) - Φ((x-mean)/std)|
        let nf = n as f64;
        let mut d_stat = 0.0_f64;

        for (i, &x) in sorted.iter().enumerate() {
            let z = (x - mean) / std;
            let f_n = (i + 1) as f64 / nf;
            let f_n_prev = i as f64 / nf;
            let phi = normal.cdf(z);
            d_stat = d_stat.max((f_n - phi).abs()).max((f_n_prev - phi).abs());
        }

        // Approximate p-value using Lilliefors table approximation
        // Based on Dallal & Wilkinson (1986) formula
        let sqrt_n = nf.sqrt();
        let d_adj = d_stat * (sqrt_n - 0.01 + 0.85 / sqrt_n);
        let p_value = if d_adj <= 0.302 {
            1.0
        } else if d_adj <= 0.5 {
            2.76773 - 19.828315 * d_adj + 80.709644 * d_adj.powi(2) - 138.55152 * d_adj.powi(3)
                + 81.218052 * d_adj.powi(4)
        } else if d_adj <= 1.8 {
            (-0.7514 + 1.3076 * d_adj).exp().clamp(0.0, 1.0) * (-8.318 * d_adj * d_adj).exp()
        } else {
            0.0
        }
        .clamp(0.0, 1.0);

        Ok((d_stat, p_value))
    }

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

    /// Engle's ARCH LM test for conditional heteroskedasticity.
    ///
    /// H₀: no ARCH effects of order `lags` in `series`.
    ///
    /// Procedure:
    ///   1. Demean the series and compute squared residuals e_t².
    ///   2. Regress e_t² on a constant and `lags` of itself.
    ///   3. LM = n_eff · R²  ~  χ²(lags) under H₀.
    ///   4. F  = (R²/p) / ((1−R²)/(n_eff−p−1))  ~  F(p, n_eff−p−1).
    ///
    /// Returns `ArchTestResult`.
    pub fn arch_test(series: &Array1<f64>, lags: usize) -> Result<ArchTestResult, GreenersError> {
        // drop NaN/Inf before any computation
        let clean: Vec<f64> = series.iter()
            .cloned()
            .filter(|x| x.is_finite())
            .collect();
        let series = Array1::from_vec(clean);

        let n = series.len();
        if n <= lags + 2 {
            return Err(GreenersError::ShapeMismatch(format!(
                "ARCH test needs > {} observations, got {}",
                lags + 2,
                n
            )));
        }

        // demean and square
        let mean = series.mean().unwrap_or(0.0);
        let e2: Vec<f64> = series.iter().map(|&x| (x - mean).powi(2)).collect();

        let n_eff = n - lags;

        // y_aux = e_t²  for t = lags..n
        let y_aux = Array1::from_vec(e2[lags..].to_vec());

        // X_aux = [1, e_{t-1}², ..., e_{t-lags}²]
        let mut x_data = Vec::with_capacity(n_eff * (lags + 1));
        for t in lags..n {
            x_data.push(1.0); // intercept
            for k in 1..=lags {
                x_data.push(e2[t - k]);
            }
        }
        let x_aux = Array2::from_shape_vec((n_eff, lags + 1), x_data)
            .map_err(|_| GreenersError::ShapeMismatch("ARCH: matrix build failed".into()))?;

        let aux = OLS::fit(&y_aux, &x_aux, CovarianceType::NonRobust)?;
        let r2 = aux.r_squared.clamp(0.0, 1.0);

        let lm_stat = n_eff as f64 * r2;
        let df_lm = lags as f64;
        let chi2 = ChiSquared::new(df_lm)
            .map_err(|_| GreenersError::OptimizationFailed)?;
        let lm_pvalue = 1.0 - chi2.cdf(lm_stat);

        let df1 = lags as f64;
        let df2 = (n_eff - lags - 1) as f64;
        let f_stat = if df2 > 0.0 && r2 < 1.0 {
            (r2 / df1) / ((1.0 - r2) / df2)
        } else {
            f64::INFINITY
        };
        let f_pvalue = if df2 > 0.0 {
            use statrs::distribution::FisherSnedecor;
            let f_dist = FisherSnedecor::new(df1, df2)
                .map_err(|_| GreenersError::OptimizationFailed)?;
            1.0 - ContinuousCDF::cdf(&f_dist, f_stat)
        } else {
            0.0
        };

        Ok(ArchTestResult {
            lm_stat,
            lm_pvalue,
            f_stat,
            f_pvalue,
            lags,
            n_obs: n_eff,
            r_squared: r2,
        })
    }

    /// Ljung-Box portmanteau test for serial autocorrelation.
    ///
    /// H₀: the first `lags` autocorrelations are jointly zero.
    ///
    /// Q = n(n+2) Σ_{k=1}^{m} ρ̂_k² / (n−k)  ~  χ²(m) under H₀.
    ///
    /// NaN/Inf values are removed before computation.
    pub fn ljung_box(series: &Array1<f64>, lags: usize) -> Result<LjungBoxResult, GreenersError> {
        let clean: Vec<f64> = series.iter().cloned().filter(|x| x.is_finite()).collect();
        let n = clean.len();
        if lags == 0 {
            return Err(GreenersError::InvalidOperation("lags must be >= 1".into()));
        }
        if n <= lags + 1 {
            return Err(GreenersError::ShapeMismatch(format!(
                "Ljung-Box needs > {} observations, got {}",
                lags + 1,
                n
            )));
        }

        let nf = n as f64;
        let mean = clean.iter().sum::<f64>() / nf;
        let denom: f64 = clean.iter().map(|&x| (x - mean).powi(2)).sum();

        if denom < 1e-15 {
            return Err(GreenersError::InvalidOperation("zero-variance series".into()));
        }

        // sample ACF at lags 1..=lags
        let mut acf = Vec::with_capacity(lags);
        for k in 1..=lags {
            let num: f64 = (k..n).map(|t| (clean[t] - mean) * (clean[t - k] - mean)).sum();
            acf.push(num / denom);
        }

        // Q = n(n+2) Σ ρ̂_k² / (n-k)
        let q_stat = nf * (nf + 2.0)
            * acf.iter().enumerate()
                .map(|(i, &r)| r * r / (nf - (i + 1) as f64))
                .sum::<f64>();

        let chi2 = ChiSquared::new(lags as f64)
            .map_err(|_| GreenersError::OptimizationFailed)?;
        let p_value = 1.0 - chi2.cdf(q_stat);

        Ok(LjungBoxResult { q_stat, p_value, lags, n_obs: n, acf })
    }
}
