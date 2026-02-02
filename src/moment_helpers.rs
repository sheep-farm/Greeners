use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Moment helper functions analogous to `statsmodels.stats.moment_helpers`.
pub struct MomentHelpers;

impl MomentHelpers {
    /// Sample skewness (Fisher's definition, bias-corrected).
    /// G1 = (n / ((n-1)*(n-2))) * sum(((x - mean) / std)^3)
    pub fn skewness(data: &[f64]) -> f64 {
        let n = data.len() as f64;
        if n < 3.0 {
            return f64::NAN;
        }
        let mean = data.iter().sum::<f64>() / n;
        let m2: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = m2.sqrt();
        if std == 0.0 {
            return f64::NAN;
        }
        let sum3: f64 = data.iter().map(|x| ((x - mean) / std).powi(3)).sum();
        (n / ((n - 1.0) * (n - 2.0))) * sum3
    }

    /// Sample excess kurtosis (Fisher's definition, bias-corrected).
    /// Returns kurtosis - 3 (Normal = 0).
    pub fn kurtosis(data: &[f64]) -> f64 {
        let n = data.len() as f64;
        if n < 4.0 {
            return f64::NAN;
        }
        let mean = data.iter().sum::<f64>() / n;
        let m2: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = m2.sqrt();
        if std == 0.0 {
            return f64::NAN;
        }
        let sum4: f64 = data.iter().map(|x| ((x - mean) / std).powi(4)).sum();
        let g2 = (n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0)) * sum4
            - 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0));
        g2
    }

    /// Jarque-Bera test statistic and p-value for normality.
    /// JB = (n/6) * (S^2 + K^2/4), compared to chi2(2).
    pub fn jarque_bera(data: &[f64]) -> (f64, f64) {
        let n = data.len() as f64;
        let s = Self::skewness(data);
        let k = Self::kurtosis(data);
        let jb = (n / 6.0) * (s.powi(2) + k.powi(2) / 4.0);
        let chi2 = ChiSquared::new(2.0).unwrap();
        let p = 1.0 - chi2.cdf(jb);
        (jb, p)
    }

    /// D'Agostino's K² test for normality.
    /// Uses skewness Z1 and kurtosis Z2, then K² = Z1² + Z2², chi2(2).
    pub fn dagostino(data: &[f64]) -> (f64, f64) {
        let n = data.len() as f64;
        if n < 20.0 {
            return (f64::NAN, f64::NAN);
        }

        // Skewness test (Z1)
        let mean = data.iter().sum::<f64>() / n;
        let m2: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let m3: f64 = data.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
        let sqrt_b1 = m3 / m2.powf(1.5);

        let y = sqrt_b1 * ((n + 1.0) * (n + 3.0) / (6.0 * (n - 2.0))).sqrt();
        let beta2 = 3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0)
            / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0));
        let w2 = -1.0 + (2.0 * (beta2 - 1.0)).sqrt();
        let delta = 1.0 / (0.5 * w2.ln()).sqrt();
        let alpha = (2.0 / (w2 - 1.0)).sqrt();
        let z1 = delta * (y / alpha + ((y / alpha).powi(2) + 1.0).sqrt()).ln();

        // Kurtosis test (Z2)
        let m4: f64 = data.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;
        let b2 = m4 / (m2 * m2);
        let e_b2 = 3.0 * (n - 1.0) / (n + 1.0);
        let var_b2 =
            24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0).powi(2) * (n + 3.0) * (n + 5.0));
        let x = (b2 - e_b2) / var_b2.sqrt();
        let sqrt_beta1 = 6.0 * (n * n - 5.0 * n + 2.0) / ((n + 7.0) * (n + 9.0))
            * (6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0))).sqrt();
        let a = 6.0 + 8.0 / sqrt_beta1 * (2.0 / sqrt_beta1 + (1.0 + 4.0 / (sqrt_beta1.powi(2))).sqrt());
        let z2 = ((1.0 - 2.0 / (9.0 * a))
            - ((1.0 - 2.0 / a) / (1.0 + x * (2.0 / (a - 4.0)).sqrt())).cbrt())
            / (2.0 / (9.0 * a)).sqrt();

        let k2 = z1.powi(2) + z2.powi(2);
        let chi2 = ChiSquared::new(2.0).unwrap();
        let p = 1.0 - chi2.cdf(k2);
        (k2, p)
    }

    /// Convert central moments to cumulants.
    /// Input: \[mu1, mu2, mu3, mu4\] (central moments).
    /// Output: \[kappa1, kappa2, kappa3, kappa4\] (cumulants).
    pub fn central_moments_to_cumulants(moments: &[f64; 4]) -> [f64; 4] {
        let [mu1, mu2, mu3, mu4] = *moments;
        // kappa1 = mu1 (mean)
        // kappa2 = mu2 (variance, since these are central moments)
        // kappa3 = mu3
        // kappa4 = mu4 - 3*mu2^2
        [mu1, mu2, mu3, mu4 - 3.0 * mu2.powi(2)]
    }

    /// Convert cumulants to central moments.
    pub fn cumulants_to_central_moments(cumulants: &[f64; 4]) -> [f64; 4] {
        let [k1, k2, k3, k4] = *cumulants;
        // mu1 = k1
        // mu2 = k2
        // mu3 = k3
        // mu4 = k4 + 3*k2^2
        [k1, k2, k3, k4 + 3.0 * k2.powi(2)]
    }

    /// Compute the first n raw moments from data.
    pub fn raw_moments(data: &[f64], n: usize) -> Vec<f64> {
        let len = data.len() as f64;
        if len == 0.0 {
            return vec![f64::NAN; n];
        }
        (1..=n)
            .map(|k| data.iter().map(|x| x.powi(k as i32)).sum::<f64>() / len)
            .collect()
    }

    /// Compute the first n central moments from data.
    pub fn central_moments(data: &[f64], n: usize) -> Vec<f64> {
        let len = data.len() as f64;
        if len == 0.0 {
            return vec![f64::NAN; n];
        }
        let mean = data.iter().sum::<f64>() / len;
        (1..=n)
            .map(|k| {
                data.iter()
                    .map(|x| (x - mean).powi(k as i32))
                    .sum::<f64>()
                    / len
            })
            .collect()
    }
}
