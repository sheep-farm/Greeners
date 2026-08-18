use crate::panel::{PanelResult, RandomEffectsResult};
use statrs::distribution::{ChiSquared, ContinuousCDF};

pub struct HausmanTest;

impl HausmanTest {
    /// Compara Fixed Effects vs Random Effects.
    /// H0: Random Effects is consistent (preferable).
    /// H1: Random Effects is inconsistent (use Fixed Effects).
    pub fn compare(fe: &PanelResult, re: &RandomEffectsResult) -> String {
        let k = fe.params.len();

        // 1. Diferença dos Betas (b_fe - b_re)
        let diff_beta = &fe.params - &re.params;

        // 2. Diferença das Variâncias (Var_fe - Var_re)
        //Note: Simplification using only the diagonal (subsequent cross covariance null for simple test)
        //The complete test would require the complete covariance matrices, but the diagonal is a good proxy.
        let var_fe = fe.std_errors.mapv(|s| s.powi(2));
        let var_re = re.std_errors.mapv(|s| s.powi(2));
        let diff_var = &var_fe - &var_re;

        //3. Chi2 Statistics (Quadratic Form)
        // H = (b_diff)' * (Var_diff)^-1 * (b_diff)
        //As we are using diagonal, simplifies for weighted sum
        let mut chi2_stat = 0.0;
        for i in 0..k {
            if diff_var[i] > 0.0 {
                chi2_stat += (diff_beta[i].powi(2)) / diff_var[i];
            }
        }

        //4. P-Value
        let Ok(dist) = ChiSquared::new(k as f64) else {
            return "Invalid degrees of freedom".to_string();
        };
        let p_value = 1.0 - dist.cdf(chi2_stat);

        //Format Output
        let recommendation = if p_value < 0.05 {
            "Reject H0. Use FIXED EFFECTS (RE is inconsistent)."
        } else {
            "Fail to reject H0. Use RANDOM EFFECTS (it is efficient)."
        };

        format!(
            "\n=== Hausman Test ===\nChi2 Statistic: {:.4}\nP-Value: {:.4}\nResult: {}",
            chi2_stat, p_value, recommendation
        )
    }
}
