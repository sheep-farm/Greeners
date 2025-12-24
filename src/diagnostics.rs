use crate::error::GreenersError;
use crate::CovarianceType; // Necessário para chamar o fit do OLS
use crate::OLS; // Reusamos o OLS para a regressão auxiliar do Breusch-Pagan
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

        // Calcular Momentos Centrais
        let m2 = residuals.mapv(|r| (r - mean).powi(2)).sum() / n;
        let m3 = residuals.mapv(|r| (r - mean).powi(3)).sum() / n;
        let m4 = residuals.mapv(|r| (r - mean).powi(4)).sum() / n;

        // Skewness (S) e Kurtosis (K)
        let skewness = m3 / m2.powf(1.5);
        let kurtosis = m4 / m2.powi(2);

        // JB = (n/6) * (S^2 + (K - 3)^2 / 4)
        let jb_stat = (n / 6.0) * (skewness.powi(2) + (kurtosis - 3.0).powi(2) / 4.0);

        // Distribuição Chi-Quadrado com 2 graus de liberdade
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

        // 1. Variável dependente auxiliar: resíduos ao quadrado
        let u_sq = residuals.mapv(|x| x.powi(2));

        // 2. Regressão Auxiliar: u^2 contra X
        // Usamos CovarianceType::NonRobust porque só queremos o R2
        let aux_model = OLS::fit(&u_sq, x, CovarianceType::NonRobust)?;

        // 3. Lagrange Multiplier Statistic = n * R2
        let lm_stat = n * aux_model.r_squared;

        // Graus de liberdade = k (número de regressores na auxiliar, excluindo constante se houver, mas aqui simplificamos para k-1 se tiver intercepto)
        // O correto do BP é df = numero de variaveis exogenas que causam a variancia.
        // Assumindo que X tem intercepto e queremos testar as variáveis:
        let df = (x.ncols() - 1) as f64;

        // Proteção para caso X tenha só intercepto ou df <= 0
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
