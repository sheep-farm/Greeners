use crate::linalg::{LinalgCholesky as _, LinalgEig as _, LinalgInverse as _, UPLO};
use crate::GreenersError;
use ndarray::{s, Array1, Array2, Axis};
use num_complex::Complex64;
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

#[derive(Debug)]
pub struct VecmResult {
    pub alpha: Array2<f64>,
    pub beta: Array2<f64>,
    pub gamma: Array2<f64>,
    pub residuals: Array2<f64>,
    pub std_errors_alpha: Array2<f64>,
    pub std_errors_beta: Array2<f64>,
    pub std_errors_gamma: Array2<f64>,
    pub variable_names: Vec<String>,
    pub rank: usize,
    pub n_vars: usize,
    pub n_obs: usize,
    pub lags: usize,
    pub eigenvalues: Array1<f64>,
    // Original series used for bootstrap initial conditions.
    pub data: Array2<f64>,
}

impl fmt::Display for VecmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            format!(" VECM (Johansen ML) - Rank {} ", self.rank)
        )?;
        writeln!(f, "{:<20} {:>10}", "No. Variables:", self.n_vars)?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;

        writeln!(f, "\n{:-^78}", " Cointegration Vectors (Beta) ")?;
        writeln!(
            f,
            "Interpret: Long-run equilibrium relationships (The 'Leash')"
        )?;
        for row in self.beta.rows() {
            write!(f, "[ ")?;
            for val in row {
                write!(f, "{:>10.4} ", val)?;
            }
            writeln!(f, "]")?;
        }

        writeln!(f, "\n{:-^78}", " Adjustment Coefficients (Alpha) ")?;
        writeln!(f, "Interpret: Speed of correction towards equilibrium")?;
        for row in self.alpha.rows() {
            write!(f, "[ ")?;
            for val in row {
                write!(f, "{:>10.4} ", val)?;
            }
            writeln!(f, "]")?;
        }

        writeln!(f, "\n{:-^78}", " Johansen Eigenvalues (Lambda) ")?;
        for val in &self.eigenvalues {
            write!(f, "{:>10.4} ", val)?;
        }
        writeln!(f, "\n{:=^78}", "")?;

        // Parsable coefficient table for validation tooling.
        writeln!(f, "\n{:-^78}", " Parameters ")?;
        writeln!(
            f,
            "{:<20} {:>10} {:>10} {:>8} {:>8} {:>10} {:>10}",
            "", "coef", "std err", "z", "P>|z|", "[0.025", "0.975]"
        )?;
        writeln!(f, "{:-^78}", "")?;

        let normal = Normal::new(0.0, 1.0).map_err(|_| fmt::Error)?;
        let mut print_row = |name: String, coef: f64, se: f64| -> fmt::Result {
            let z = if se > 0.0 { coef / se } else { 0.0 };
            let p = 2.0 * (1.0 - normal.cdf(z.abs()));
            let ci_lower = coef - 1.96 * se;
            let ci_upper = coef + 1.96 * se;
            writeln!(
                f,
                "{:<20} {:>10.4} {:>10.4} {:>8.3} {:>8.3} {:>10.4} {:>10.4}",
                name, coef, se, z, p, ci_lower, ci_upper
            )
        };

        for r in 0..self.rank {
            for j in 0..self.n_vars {
                let name = format!("beta_{}_y{}", r + 1, j + 1);
                let coef = self.beta[[j, r]];
                let se = self.std_errors_beta[[j, r]];
                print_row(name, coef, se)?;
            }
        }
        for j in 0..self.n_vars {
            for r in 0..self.rank {
                let name = format!("alpha_{}_y{}", r + 1, j + 1);
                let coef = self.alpha[[j, r]];
                let se = self.std_errors_alpha[[j, r]];
                print_row(name, coef, se)?;
            }
        }

        let k = self.n_vars;
        let p_vecm = self.lags.saturating_sub(1);
        for j in 0..k {
            for l in 1..=p_vecm {
                for i in 0..k {
                    let name = format!("gamma_{}_y{}_y{}", l, j + 1, i + 1);
                    let col = 1 + (l - 1) * k + i;
                    let coef = self.gamma[[j, col]];
                    let se = self.std_errors_gamma[[j, (l - 1) * k + i]];
                    print_row(name, coef, se)?;
                }
            }
        }

        writeln!(f, "{:=^78}", "")
    }
}

impl VecmResult {
    /// Run a parametric residual bootstrap and return a new `VecmResult` with
    /// `std_errors_alpha`, `std_errors_beta` and `std_errors_gamma` populated.
    pub fn bootstrap_standard_errors(&self, n_boot: usize) -> Result<VecmResult, GreenersError> {
        if n_boot == 0 {
            return Err(GreenersError::InvalidOperation(
                "n_boot must be positive".into(),
            ));
        }

        let k = self.n_vars;
        let rank = self.rank;
        let p_vecm = self.lags.saturating_sub(1);
        let n_eff = self.n_obs;
        let n_gamma_short_cols = k * p_vecm;
        let t_total = n_eff + self.lags;

        let mut alpha_sum = Array2::<f64>::zeros((k, rank));
        let mut alpha_sum_sq = Array2::<f64>::zeros((k, rank));
        let mut beta_sum = Array2::<f64>::zeros((k, rank));
        let mut beta_sum_sq = Array2::<f64>::zeros((k, rank));
        let mut gamma_sum = Array2::<f64>::zeros((k, n_gamma_short_cols));
        let mut gamma_sum_sq = Array2::<f64>::zeros((k, n_gamma_short_cols));

        let mut rng = thread_rng();
        let idx_dist = Uniform::new(0, n_eff);

        let mut successes = 0usize;
        let max_failures = n_boot / 2 + 1;
        let mut failures = 0usize;

        while successes < n_boot && failures < max_failures {
            let mut y_boot = Array2::<f64>::zeros((t_total, k));
            for t in 0..self.lags {
                y_boot.row_mut(t).assign(&self.data.row(t));
            }

            for i in 0..n_eff {
                let t = self.lags + i;
                let y_lag = y_boot.row(t - 1).insert_axis(Axis(1)); // K x 1

                // Cointegration contribution: alpha * beta' * y_{t-1}
                let beta_ty = self.beta.t().dot(&y_lag); // rank x 1
                let mut dy_t = self.alpha.dot(&beta_ty); // K x 1

                // Short-run dynamics (excluding intercept)
                for l in 1..=p_vecm {
                    let dy_lag = &y_boot.row(t - l) - &y_boot.row(t - l - 1);
                    let gamma_l = self.gamma.slice(s![.., 1 + (l - 1) * k..1 + l * k]);
                    dy_t += &gamma_l.dot(&dy_lag.insert_axis(Axis(1)));
                }

                // Intercept
                dy_t += &self.gamma.column(0).insert_axis(Axis(1));

                // Bootstrap residual
                let idx = idx_dist.sample(&mut rng);
                dy_t += &self.residuals.row(idx).insert_axis(Axis(1));

                let y_t = &y_boot.row(t - 1).insert_axis(Axis(1)) + &dy_t;
                y_boot.row_mut(t).assign(&y_t.slice(s![.., 0]));
            }

            match VECM::fit(&y_boot, self.lags, self.rank) {
                Ok(mut boot) => {
                    // Align sign: make the first non-zero element of each beta column positive.
                    for r in 0..rank {
                        let col = boot.beta.column(r);
                        if let Some(first) = col.iter().find(|&&x| x.abs() > 1e-10) {
                            if *first < 0.0 {
                                boot.beta.column_mut(r).mapv_inplace(|x| -x);
                                boot.alpha.column_mut(r).mapv_inplace(|x| -x);
                            }
                        }
                    }

                    alpha_sum += &boot.alpha;
                    alpha_sum_sq += &boot.alpha.mapv(|x| x * x);
                    beta_sum += &boot.beta;
                    beta_sum_sq += &boot.beta.mapv(|x| x * x);

                    if n_gamma_short_cols > 0 {
                        let gamma_short = boot.gamma.slice(s![.., 1..]).to_owned();
                        gamma_sum += &gamma_short;
                        gamma_sum_sq += &gamma_short.mapv(|x| x * x);
                    }

                    successes += 1;
                }
                Err(_) => {
                    failures += 1;
                }
            }
        }

        if successes < n_boot / 2 + 1 {
            return Err(GreenersError::OptimizationFailed);
        }

        let n = successes as f64;
        let mean_alpha = &alpha_sum / n;
        let mean_beta = &beta_sum / n;
        let mean_gamma = if n_gamma_short_cols > 0 {
            &gamma_sum / n
        } else {
            Array2::zeros((k, 0))
        };

        let var_alpha = (&alpha_sum_sq / n) - &mean_alpha.mapv(|x| x * x);
        let var_beta = (&beta_sum_sq / n) - &mean_beta.mapv(|x| x * x);
        let var_gamma = if n_gamma_short_cols > 0 {
            (&gamma_sum_sq / n) - &mean_gamma.mapv(|x| x * x)
        } else {
            Array2::zeros((k, 0))
        };

        let se_alpha = var_alpha.mapv(|x| x.max(0.0).sqrt());
        let se_beta = var_beta.mapv(|x| x.max(0.0).sqrt());
        let se_gamma = var_gamma.mapv(|x| x.max(0.0).sqrt());

        Ok(VecmResult {
            alpha: self.alpha.clone(),
            beta: self.beta.clone(),
            gamma: self.gamma.clone(),
            residuals: self.residuals.clone(),
            std_errors_alpha: se_alpha,
            std_errors_beta: se_beta,
            std_errors_gamma: se_gamma,
            variable_names: self.variable_names.clone(),
            rank: self.rank,
            n_vars: self.n_vars,
            n_obs: self.n_obs,
            lags: self.lags,
            eigenvalues: self.eigenvalues.clone(),
            data: self.data.clone(),
        })
    }

    /// Convenience alias for `bootstrap_standard_errors`.
    pub fn with_inference(&self, n_boot: usize) -> Result<VecmResult, GreenersError> {
        self.bootstrap_standard_errors(n_boot)
    }
}

pub struct VECM;

impl VECM {
    pub fn fit(data: &Array2<f64>, lags: usize, rank: usize) -> Result<VecmResult, GreenersError> {
        let t_total = data.nrows();
        let k = data.ncols();

        if rank == 0 || rank >= k {
            return Err(GreenersError::ShapeMismatch(
                "Rank must be between 1 and k-1".into(),
            ));
        }

        let _n_obs = t_total - lags;

        // 1. Preparar Dados (Delta Y_t)
        let mut dy = Array2::<f64>::zeros((t_total - 1, k));
        for i in 1..t_total {
            let diff = &data.row(i) - &data.row(i - 1);
            dy.row_mut(i - 1).assign(&diff);
        }

        let n_eff = t_total - lags;
        let p_vecm = lags - 1;
        let n_z_cols = k * p_vecm + 1;

        let mut z_mat = Array2::<f64>::zeros((n_eff, n_z_cols));
        let mut dy_target = Array2::<f64>::zeros((n_eff, k));
        let mut y_lag_level = Array2::<f64>::zeros((n_eff, k));

        for i in 0..n_eff {
            let t_original = lags + i;

            let dy_row = &data.row(t_original) - &data.row(t_original - 1);
            dy_target.row_mut(i).assign(&dy_row);

            y_lag_level.row_mut(i).assign(&data.row(t_original - 1));

            z_mat[[i, 0]] = 1.0; // Intercepto

            for l in 1..=p_vecm {
                let lag_time = t_original - l;
                let dy_lag = &data.row(lag_time) - &data.row(lag_time - 1);

                let start_col = 1 + (l - 1) * k;
                for j in 0..k {
                    z_mat[[i, start_col + j]] = dy_lag[j];
                }
            }
        }

        // 2. Regressões Auxiliares
        let ztz = z_mat.t().dot(&z_mat);
        let ztz_inv = ztz.inv().map_err(|_| GreenersError::SingularMatrix)?;

        let beta_0 = ztz_inv.dot(&z_mat.t()).dot(&dy_target);
        let r0 = &dy_target - &z_mat.dot(&beta_0);

        let beta_1 = ztz_inv.dot(&z_mat.t()).dot(&y_lag_level);
        let r1 = &y_lag_level - &z_mat.dot(&beta_1);

        // 3. Matrizes de Momento
        let t_float = n_eff as f64;
        let s00 = r0.t().dot(&r0) / t_float;
        let s11 = r1.t().dot(&r1) / t_float;
        let s01 = r0.t().dot(&r1) / t_float;
        let s10 = s01.t();

        // Resolver autovalores generalizados
        let s11_chol = s11
            .cholesky(UPLO::Lower)
            .map_err(|_| GreenersError::SingularMatrix)?;

        let s11_inv_chol = s11_chol.inv().map_err(|_| GreenersError::SingularMatrix)?;

        let s00_inv = s00.inv().map_err(|_| GreenersError::SingularMatrix)?;

        let temp = s11_inv_chol
            .dot(&s10)
            .dot(&s00_inv)
            .dot(&s01)
            .dot(&s11_inv_chol.t());

        let (eigvals_complex, eigvecs_complex) =
            temp.eig().map_err(|_| GreenersError::OptimizationFailed)?;

        // 4. Filtrar e Ordenar (CORREÇÃO DE TIPOS AQUI)
        let mut pairs: Vec<(f64, Array1<f64>)> = eigvals_complex
            .iter()
            .enumerate()
            .map(|(i, v)| {
                // Explicitamos que 'c' é Complex64 para ajudar o compilador
                let vec_real = eigvecs_complex.column(i).mapv(|c: Complex64| c.re);
                (v.re, vec_real)
            })
            .collect();

        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let sorted_eigenvalues: Array1<f64> = Array1::from_vec(pairs.iter().map(|p| p.0).collect());

        // 5. Estimar Beta e Alpha
        let mut beta_est = Array2::<f64>::zeros((k, rank));
        // for r in 0..rank {
        for (r, _pair) in pairs.iter().enumerate().take(rank) {
            let v = &pairs[r].1;
            let beta_vec = s11_inv_chol.t().dot(v);
            beta_est.column_mut(r).assign(&beta_vec);
        }

        let cointegration_term = r1.dot(&beta_est);

        let alpha_est = r0.t().dot(&cointegration_term).dot(
            &cointegration_term
                .t()
                .dot(&cointegration_term)
                .inv()
                .unwrap(),
        );

        let error_correction = y_lag_level.dot(&beta_est).dot(&alpha_est.t());
        let dy_clean = &dy_target - &error_correction;
        let gamma_full = ztz_inv.dot(&z_mat.t()).dot(&dy_clean);

        // Residuals of the VECM equation: dy - Z*gamma' - y_lag_level*beta*alpha'
        let residuals = &dy_target - &z_mat.dot(&gamma_full) - &error_correction;

        let variable_names: Vec<String> = (0..k).map(|i| format!("y{}", i + 1)).collect();

        let p_vecm = lags - 1;
        let n_gamma_short_cols = k * p_vecm;

        Ok(VecmResult {
            alpha: alpha_est,
            beta: beta_est,
            gamma: gamma_full.t().to_owned(),
            residuals,
            std_errors_alpha: Array2::zeros((k, rank)),
            std_errors_beta: Array2::zeros((k, rank)),
            std_errors_gamma: Array2::zeros((k, n_gamma_short_cols)),
            variable_names,
            rank,
            n_vars: k,
            n_obs: n_eff,
            lags,
            eigenvalues: sorted_eigenvalues,
            data: data.clone(),
        })
    }
}
