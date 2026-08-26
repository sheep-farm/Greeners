#[inline]
pub fn array1_slice(arr: &ndarray::Array1<f64>) -> &[f64] {
    arr.as_slice().unwrap_or(&[])
}

#[derive(Debug, Clone, PartialEq)]
pub enum CovarianceType {
    /// Standard OLS (Homoscedastic)
    NonRobust,
    /// White's Robust Errors (HC1) - Only Heteroscedasticity
    /// Uses small-sample correction: n/(n-k)
    HC1,
    /// HC2 - Leverage-adjusted heteroscedasticity-robust SE
    /// Adjusts for leverage: σ²_i / (1 - h_i)
    /// More efficient than HC1 with small samples
    HC2,
    /// HC3 - Jackknife heteroscedasticity-robust SE
    /// Uses squared leverage adjustment: σ²_i / (1 - h_i)²
    /// Most robust for small samples (MacKinnon & White, 1985)
    /// Recommended default robust SE estimator
    HC3,
    /// HC4 - Refined jackknife (Cribari-Neto, 2004)
    /// Uses power adjustment: σ²_i / (1 - h_i)^δᵢ where δᵢ = min(4, n·h_i/k)
    /// Best small-sample performance, especially with influential observations
    /// More refined than HC3 for datasets with high-leverage points
    HC4,
    /// Newey-West (HAC) - Heteroscedasticity + Autocorrelation
    /// The 'usize' parameter is the number of lags (L).
    /// Common rule of thumb: L = n^0.25
    NeweyWest(usize),
    /// Clustered Standard Errors (One-Way)
    /// Critical for panel data, experiments, and grouped observations
    /// The `Vec<usize>` contains cluster IDs for each observation
    Clustered(Vec<usize>),
    /// Two-Way Clustered Standard Errors (Cameron-Gelbach-Miller, 2011)
    /// For panel data with clustering along two dimensions (e.g., firm + time)
    /// First Vec: cluster IDs for dimension 1 (e.g., firm IDs)
    /// Second Vec: cluster IDs for dimension 2 (e.g., time periods)
    /// Formula: V = V₁ + V₂ - V₁₂ where V₁₂ is intersection clustering
    /// Essential for panel data with both cross-sectional and time correlation
    ClusteredTwoWay(Vec<usize>, Vec<usize>),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum InferenceType {
    /// Student's t-distribution (default for finite samples)
    ///
    /// Uses t(df) distribution for hypothesis testing and confidence intervals.
    /// This is the exact finite-sample distribution under normality assumptions.
    ///
    /// **Recommended for:**
    /// - Small to medium samples (n < 100)
    /// - When exact finite-sample inference is desired
    /// - Conservative hypothesis testing
    ///
    /// **Used by:** OLS, IV/2SLS, Panel models (default)
    #[default]
    StudentT,

    /// Standard Normal distribution (z-distribution)
    ///
    /// Uses N(0,1) distribution for hypothesis testing and confidence intervals.
    /// This is the asymptotic distribution (as n → ∞).
    ///
    /// **Recommended for:**
    /// - Large samples (n > 1000)
    /// - Asymptotic theory contexts (MLE, GMM)
    /// - Compatibility with statsmodels/Python
    ///
    /// **Used by:** Logit, Probit, GMM, Quantile Regression (always)
    ///
    /// **Note:** For large samples (df > 30), t and z distributions are nearly identical.
    Normal,
}
