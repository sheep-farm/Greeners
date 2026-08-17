use statrs::distribution::{
    ChiSquared, Continuous, ContinuousCDF, FisherSnedecor, Normal, StudentsT,
};

/// bilateral p-value of Student t distribution.
pub fn t_pvalue_two(t: f64, df: f64) -> f64 {
    let Ok(dist) = StudentsT::new(0.0, 1.0, df) else {
        return f64::NAN;
    };
    2.0 * (1.0 - dist.cdf(t.abs()))
}

/// Quantil of Student t distribution (inverse to CDF).
/// Ex.: `t_quantile(0.975, df)` devolve o t crítico para IC 95%.
pub fn t_quantile(p: f64, df: f64) -> f64 {
    let Ok(dist) = StudentsT::new(0.0, 1.0, df) else {
        return f64::NAN;
    };
    dist.inverse_cdf(p)
}

/// p-value of the chi-square distribution.
pub fn chi2_pvalue(stat: f64, df: f64) -> f64 {
    let Ok(dist) = ChiSquared::new(df) else {
        return f64::NAN;
    };
    1.0 - dist.cdf(stat)
}

/// PDF of standard normal distribution N(0.1).
pub fn norm_pdf(x: f64) -> f64 {
    let Ok(dist) = Normal::new(0.0, 1.0) else {
        return f64::NAN;
    };
    dist.pdf(x)
}

/// Função logística (sigmoid): 1 / (1 + e^{-x}).
pub fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// p-value of Fisher-Snedecor F distribution (upper tail).
pub fn f_pvalue(f: f64, df1: f64, df2: f64) -> f64 {
    let Ok(dist) = FisherSnedecor::new(df1, df2) else {
        return f64::NAN;
    };
    1.0 - dist.cdf(f)
}
