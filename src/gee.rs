use crate::error::GreenersError;
use crate::glm::{Family, Link};
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

/// Correlation structure for GEE.
#[derive(Debug, Clone)]
pub enum CorrStructure {
    /// Independent working correlation
    Independence,
    /// Exchangeable (compound symmetry)
    Exchangeable,
    /// First-order autoregressive
    AR1,
    /// Unstructured
    Unstructured,
}

/// Result of GEE estimation.
#[derive(Debug)]
pub struct GeeResult {
    pub params: Array1<f64>,
    pub robust_se: Array1<f64>,
    pub naive_se: Array1<f64>,
    pub z_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub working_correlation: Array2<f64>,
    pub scale: f64,
    pub qic: f64,
    pub n_obs: usize,
    pub n_groups: usize,
    pub n_iter: usize,
    pub converged: bool,
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for GeeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " Generalized Estimating Equations ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10}", "Groups:", self.n_groups)?;
        writeln!(f, "{:<20} {:>10.4}", "Scale:", self.scale)?;
        writeln!(f, "{:<20} {:>10.4}", "QIC:", self.qic)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} | {:>10} | {:>10} | {:>10} | {:>8} | {:>8}",
            "Variable", "coef", "robust SE", "naive SE", "z", "P>|z|"
        )?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.params.len() {
            let name = self
                .variable_names
                .as_ref()
                .and_then(|n| n.get(i).cloned())
                .unwrap_or_else(|| format!("x{}", i));
            writeln!(
                f,
                "{:<12} | {:>10.4} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3}",
                name,
                self.params[i],
                self.robust_se[i],
                self.naive_se[i],
                self.z_values[i],
                self.p_values[i]
            )?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

/// Generalized Estimating Equations.
pub struct GEE;

impl GEE {
    pub fn fit(
        y: &Array1<f64>,
        x: &Array2<f64>,
        groups: &Array1<usize>,
        family: &Family,
        link: &Link,
        corr_structure: &CorrStructure,
    ) -> Result<GeeResult, GreenersError> {
        Self::fit_with_names(y, x, groups, family, link, corr_structure, None)
    }

    pub fn fit_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        groups: &Array1<usize>,
        family: &Family,
        link: &Link,
        corr_structure: &CorrStructure,
        variable_names: Option<Vec<String>>,
    ) -> Result<GeeResult, GreenersError> {
        let n = y.len();
        let k = x.ncols();

        if n != x.nrows() || n != groups.len() {
            return Err(GreenersError::ShapeMismatch(
                "Dimension mismatch in GEE inputs".into(),
            ));
        }

        // Identify groups
        let mut unique_groups: Vec<usize> = groups.iter().cloned().collect();
        unique_groups.sort();
        unique_groups.dedup();
        let g = unique_groups.len();

        let group_indices: Vec<Vec<usize>> = unique_groups
            .iter()
            .map(|&grp| (0..n).filter(|&i| groups[i] == grp).collect())
            .collect();

        let max_ni = group_indices.iter().map(|idx| idx.len()).max().unwrap_or(1);

        // Initialize with identity working correlation
        let mut beta = Array1::<f64>::zeros(k);
        // Simple initialization: use mean of y
        let max_iter = 50;
        let tol = 1e-6;
        let mut converged = false;
        let mut n_iter = 0;
        let mut scale = 1.0;
        let mut work_corr = Array2::<f64>::eye(max_ni);

        for iter in 0..max_iter {
            n_iter = iter + 1;

            let eta = x.dot(&beta);
            let mu = eta.mapv(|e| apply_inv_link(link, e));

            // Pearson residuals
            let resid: Array1<f64> = Array1::from(
                (0..n)
                    .map(|i| {
                        let v = variance(family, mu[i]);
                        (y[i] - mu[i]) / v.sqrt()
                    })
                    .collect::<Vec<_>>(),
            );

            // Estimate scale
            let df = (n - k) as f64;
            scale = resid.iter().map(|r| r * r).sum::<f64>() / df;

            // Update working correlation
            work_corr = estimate_correlation(corr_structure, &resid, &group_indices, max_ni);

            // IRLS update
            let mut bread = Array2::<f64>::zeros((k, k));
            let mut meat_sum = Array1::<f64>::zeros(k);

            for idx in &group_indices {
                let ni = idx.len();
                let xi = stack_rows(x, idx);
                let yi: Array1<f64> = idx.iter().map(|&i| y[i]).collect::<Vec<_>>().into();
                let mu_i: Array1<f64> = idx.iter().map(|&i| mu[i]).collect::<Vec<_>>().into();

                // D_i = diag(dmu/deta)
                let d_i: Array1<f64> = idx
                    .iter()
                    .map(|&i| apply_dinv_link(link, eta[i]))
                    .collect::<Vec<_>>()
                    .into();

                // A_i = diag(V(mu_i))
                let a_i: Array1<f64> = idx
                    .iter()
                    .map(|&i| variance(family, mu[i]))
                    .collect::<Vec<_>>()
                    .into();

                // Working covariance: V_i = A_i^{1/2} R A_i^{1/2} * scale
                let a_sqrt: Array1<f64> = a_i.mapv(|a| a.sqrt());
                let mut v_i = Array2::<f64>::zeros((ni, ni));
                for a in 0..ni {
                    for b in 0..ni {
                        let r = if a < work_corr.nrows() && b < work_corr.ncols() {
                            work_corr[[a, b]]
                        } else if a == b {
                            1.0
                        } else {
                            0.0
                        };
                        v_i[[a, b]] = a_sqrt[a] * r * a_sqrt[b] * scale;
                    }
                }

                let v_inv = match v_i.inv() {
                    Ok(inv) => inv,
                    Err(_) => {
                        // Fallback to diagonal
                        let mut diag = Array2::<f64>::zeros((ni, ni));
                        for j in 0..ni {
                            diag[[j, j]] = 1.0 / (a_i[j] * scale).max(1e-10);
                        }
                        diag
                    }
                };

                // D_i' V_i^{-1}
                let mut di_mat = Array2::<f64>::zeros((ni, ni));
                for j in 0..ni {
                    di_mat[[j, j]] = d_i[j];
                }

                let dt_vinv = di_mat.t().dot(&v_inv);
                bread = &bread + &xi.t().dot(&dt_vinv.dot(&xi));
                let ri = &yi - &mu_i;
                meat_sum = &meat_sum + &xi.t().dot(&dt_vinv.dot(&ri));
            }

            let bread_inv = match bread.inv() {
                Ok(inv) => inv,
                Err(_) => break,
            };

            let new_beta = &beta + &bread_inv.dot(&meat_sum);

            let diff = (&new_beta - &beta)
                .iter()
                .map(|d| d.abs())
                .fold(0.0_f64, f64::max);

            beta = new_beta;

            if diff < tol {
                converged = true;
                break;
            }
        }

        // Naive covariance (model-based)
        let eta = x.dot(&beta);
        let mu = eta.mapv(|e| apply_inv_link(link, e));

        let mut bread = Array2::<f64>::zeros((k, k));
        let mut sandwich_meat = Array2::<f64>::zeros((k, k));

        for idx in &group_indices {
            let ni = idx.len();
            let xi = stack_rows(x, idx);
            let yi: Array1<f64> = idx.iter().map(|&i| y[i]).collect::<Vec<_>>().into();
            let mu_i: Array1<f64> = idx.iter().map(|&i| mu[i]).collect::<Vec<_>>().into();

            let d_i: Array1<f64> = idx
                .iter()
                .map(|&i| apply_dinv_link(link, eta[i]))
                .collect::<Vec<_>>()
                .into();
            let a_i: Array1<f64> = idx
                .iter()
                .map(|&i| variance(family, mu[i]))
                .collect::<Vec<_>>()
                .into();

            let a_sqrt: Array1<f64> = a_i.mapv(|a| a.sqrt());
            let mut v_i = Array2::<f64>::zeros((ni, ni));
            for a in 0..ni {
                for b in 0..ni {
                    let r = if a < work_corr.nrows() && b < work_corr.ncols() {
                        work_corr[[a, b]]
                    } else if a == b {
                        1.0
                    } else {
                        0.0
                    };
                    v_i[[a, b]] = a_sqrt[a] * r * a_sqrt[b] * scale;
                }
            }

            let v_inv = match v_i.inv() {
                Ok(inv) => inv,
                Err(_) => {
                    let mut diag = Array2::<f64>::zeros((ni, ni));
                    for j in 0..ni {
                        diag[[j, j]] = 1.0 / (a_i[j] * scale).max(1e-10);
                    }
                    diag
                }
            };

            let mut di_mat = Array2::<f64>::zeros((ni, ni));
            for j in 0..ni {
                di_mat[[j, j]] = d_i[j];
            }

            let dt_vinv = di_mat.t().dot(&v_inv);
            bread = &bread + &xi.t().dot(&dt_vinv.dot(&xi));

            // Meat: sum of u_i u_i' where u_i = X_i' D_i V_i^{-1} (y_i - mu_i)
            let ri = &yi - &mu_i;
            let ui = xi.t().dot(&dt_vinv.dot(&ri));
            for a in 0..k {
                for b in 0..k {
                    sandwich_meat[[a, b]] += ui[a] * ui[b];
                }
            }
        }

        let bread_inv = bread.inv()?;
        let naive_cov = bread_inv.clone();
        let robust_cov = bread_inv.dot(&sandwich_meat).dot(&bread_inv);

        let naive_se: Array1<f64> = (0..k)
            .map(|j| naive_cov[[j, j]].abs().sqrt())
            .collect::<Vec<_>>()
            .into();

        let robust_se: Array1<f64> = (0..k)
            .map(|j| robust_cov[[j, j]].abs().sqrt())
            .collect::<Vec<_>>()
            .into();

        let z_values = &beta / &robust_se;
        let normal = Normal::new(0.0, 1.0).map_err(|_| GreenersError::OptimizationFailed)?;
        let p_values = z_values.mapv(|z| 2.0 * (1.0 - normal.cdf(z.abs())));

        // QIC (Pan, 2001) - simplified
        let mut quasi_ll = 0.0;
        for i in 0..n {
            quasi_ll -= 0.5 * (y[i] - mu[i]).powi(2) / variance(family, mu[i]).max(1e-10);
        }
        let qic = -2.0 * quasi_ll + 2.0 * k as f64;

        Ok(GeeResult {
            params: beta,
            robust_se,
            naive_se,
            z_values,
            p_values,
            working_correlation: work_corr,
            scale,
            qic,
            n_obs: n,
            n_groups: g,
            n_iter,
            converged,
            variable_names,
        })
    }
}

fn stack_rows(mat: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let k = mat.ncols();
    let mut result = Array2::<f64>::zeros((indices.len(), k));
    for (i, &idx) in indices.iter().enumerate() {
        result.row_mut(i).assign(&mat.row(idx));
    }
    result
}

fn apply_inv_link(link: &Link, eta: f64) -> f64 {
    match link {
        Link::Identity => eta,
        Link::Log => eta.exp(),
        Link::Logit => 1.0 / (1.0 + (-eta).exp()),
        Link::Probit => {
            let n = Normal::new(0.0, 1.0).unwrap();
            n.cdf(eta)
        }
        Link::InversePower => 1.0 / eta.max(1e-10),
        Link::InverseSquared => 1.0 / eta.max(1e-10).sqrt(),
        Link::CLogLog => 1.0 - (-eta.exp()).exp(),
        Link::Power(p) => eta.powf(1.0 / p),
        Link::NegativeBinomial(alpha) => {
            let e = eta.exp();
            e / (1.0 - alpha * e).max(1e-10)
        }
        Link::Cauchy => 0.5 + (eta).atan() / std::f64::consts::PI,
    }
}

fn apply_dinv_link(link: &Link, eta: f64) -> f64 {
    match link {
        Link::Identity => 1.0,
        Link::Log => eta.exp(),
        Link::Logit => {
            let p = 1.0 / (1.0 + (-eta).exp());
            p * (1.0 - p)
        }
        Link::Probit => {
            use statrs::distribution::Continuous;
            let n = Normal::new(0.0, 1.0).unwrap();
            n.pdf(eta)
        }
        Link::InversePower => -1.0 / (eta * eta).max(1e-10),
        Link::InverseSquared => -0.5 / eta.max(1e-10).powf(1.5),
        Link::CLogLog => {
            let e = eta.exp();
            e * (-e).exp()
        }
        _ => 1.0, // fallback
    }
}

fn variance(family: &Family, mu: f64) -> f64 {
    match family {
        Family::Gaussian => 1.0,
        Family::Binomial => (mu * (1.0 - mu)).max(1e-10),
        Family::Poisson => mu.max(1e-10),
        Family::Gamma => (mu * mu).max(1e-10),
        Family::InverseGaussian => (mu * mu * mu).max(1e-10),
        Family::Tweedie(p) => mu.powf(*p).max(1e-10),
        Family::NegativeBinomial(alpha) => (mu + alpha * mu * mu).max(1e-10),
    }
}

fn estimate_correlation(
    structure: &CorrStructure,
    resid: &Array1<f64>,
    group_indices: &[Vec<usize>],
    max_ni: usize,
) -> Array2<f64> {
    match structure {
        CorrStructure::Independence => Array2::eye(max_ni),
        CorrStructure::Exchangeable => {
            let mut sum_rr = 0.0;
            let mut n_pairs = 0;
            for idx in group_indices {
                let ni = idx.len();
                for a in 0..ni {
                    for b in (a + 1)..ni {
                        sum_rr += resid[idx[a]] * resid[idx[b]];
                        n_pairs += 1;
                    }
                }
            }
            let alpha = if n_pairs > 0 {
                (sum_rr / n_pairs as f64).clamp(-0.99, 0.99)
            } else {
                0.0
            };
            let mut r = Array2::<f64>::eye(max_ni);
            for a in 0..max_ni {
                for b in 0..max_ni {
                    if a != b {
                        r[[a, b]] = alpha;
                    }
                }
            }
            r
        }
        CorrStructure::AR1 => {
            let mut sum_lag1 = 0.0;
            let mut n_lag1 = 0;
            for idx in group_indices {
                let ni = idx.len();
                for a in 0..(ni.saturating_sub(1)) {
                    sum_lag1 += resid[idx[a]] * resid[idx[a + 1]];
                    n_lag1 += 1;
                }
            }
            let rho = if n_lag1 > 0 {
                (sum_lag1 / n_lag1 as f64).clamp(-0.99, 0.99)
            } else {
                0.0
            };
            let mut r = Array2::<f64>::eye(max_ni);
            for a in 0..max_ni {
                for b in 0..max_ni {
                    r[[a, b]] = rho.powi((a as i32 - b as i32).unsigned_abs() as i32);
                }
            }
            r
        }
        CorrStructure::Unstructured => {
            let mut r = Array2::<f64>::eye(max_ni);
            let mut counts = Array2::<f64>::zeros((max_ni, max_ni));
            for idx in group_indices {
                let ni = idx.len();
                for a in 0..ni {
                    for b in 0..ni {
                        if a < max_ni && b < max_ni {
                            r[[a, b]] += resid[idx[a]] * resid[idx[b]];
                            counts[[a, b]] += 1.0;
                        }
                    }
                }
            }
            for a in 0..max_ni {
                for b in 0..max_ni {
                    if counts[[a, b]] > 0.0 {
                        r[[a, b]] /= counts[[a, b]];
                    }
                }
            }
            // Normalize
            let diag: Vec<f64> = (0..max_ni).map(|i| r[[i, i]].sqrt().max(1e-10)).collect();
            for a in 0..max_ni {
                for b in 0..max_ni {
                    r[[a, b]] /= diag[a] * diag[b];
                }
            }
            r
        }
    }
}
