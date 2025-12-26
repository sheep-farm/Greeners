use crate::panel::{PanelResult, RandomEffectsResult};
use statrs::distribution::{ChiSquared, ContinuousCDF};

pub struct HausmanTest;

impl HausmanTest {
    /// Compara Fixed Effects vs Random Effects.
    /// H0: Random Effects é consistente (preferível).
    /// H1: Random Effects é inconsistente (use Fixed Effects).
    pub fn compare(fe: &PanelResult, re: &RandomEffectsResult) -> String {
        let k = fe.params.len();
        
        // 1. Diferença dos Betas (b_fe - b_re)
        let diff_beta = &fe.params - &re.params;
        
        // 2. Diferença das Variâncias (Var_fe - Var_re)
        // Nota: Simplificação usando apenas a diagonal (assume covariância cruzada nula para o teste simples)
        // O teste completo exigiria as matrizes de covariância completas, mas a diagonal é um bom proxy.
        let var_fe = fe.std_errors.mapv(|s| s.powi(2));
        let var_re = re.std_errors.mapv(|s| s.powi(2));
        let diff_var = &var_fe - &var_re;
        
        // 3. Estatística Chi2 (Forma Quadrática)
        // H = (b_diff)' * (Var_diff)^-1 * (b_diff)
        // Como estamos usando diagonal, simplifica para soma ponderada
        let mut chi2_stat = 0.0;
        for i in 0..k {
            if diff_var[i] > 0.0 {
               chi2_stat += (diff_beta[i].powi(2)) / diff_var[i];
            }
        }
        
        // 4. P-Valor
        let dist = ChiSquared::new(k as f64).unwrap();
        let p_value = 1.0 - dist.cdf(chi2_stat);
        
        // Formatar Saída
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