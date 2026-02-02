use crate::error::GreenersError;
use ndarray::{s, Array1, Array2, Axis};
use ndarray_linalg::{Eigh, Inverse, UPLO};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};
use std::fmt;

// ─── PCA ───────────────────────────────────────────────────────────────────────

/// Result of Principal Component Analysis.
#[derive(Debug)]
pub struct PCAResult {
    /// Principal components (eigenvectors as columns, k x n_components)
    pub components: Array2<f64>,
    /// Explained variance per component
    pub explained_variance: Array1<f64>,
    /// Proportion of variance explained
    pub explained_variance_ratio: Array1<f64>,
    /// Loadings (components scaled by sqrt of eigenvalue)
    pub loadings: Array2<f64>,
    /// Scores (data projected onto components)
    pub scores: Array2<f64>,
    /// Column means (for centering)
    pub mean: Array1<f64>,
    /// Column standard deviations (for standardizing)
    pub std: Array1<f64>,
    pub n_obs: usize,
    pub n_components: usize,
}

impl PCAResult {
    /// Project new data onto principal components.
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let centered = self.standardize(data);
        centered.dot(&self.components)
    }

    /// Reconstruct data from scores.
    pub fn inverse_transform(&self, scores: &Array2<f64>) -> Array2<f64> {
        let recon = scores.dot(&self.components.t());
        // Unstandardize
        let mut result = recon;
        for (j, mut col) in result.axis_iter_mut(Axis(1)).enumerate() {
            col *= self.std[j];
            col += self.mean[j];
        }
        result
    }

    fn standardize(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut centered = data.clone();
        for (j, mut col) in centered.axis_iter_mut(Axis(1)).enumerate() {
            col -= self.mean[j];
            if self.std[j] > 1e-15 {
                col /= self.std[j];
            }
        }
        centered
    }
}

impl fmt::Display for PCAResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^60}", " Principal Component Analysis ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10}", "Components:", self.n_components)?;
        writeln!(
            f,
            "\n{:<12} {:>12} {:>12}",
            "Component", "Var Expl", "Cumulative"
        )?;
        writeln!(f, "{:-^40}", "")?;
        let mut cum = 0.0;
        for i in 0..self.n_components {
            cum += self.explained_variance_ratio[i];
            writeln!(
                f,
                "PC{:<10} {:>12.4} {:>12.4}",
                i + 1,
                self.explained_variance_ratio[i],
                cum
            )?;
        }
        writeln!(f, "{:=^60}", "")
    }
}

/// Principal Component Analysis via eigendecomposition of correlation matrix.
pub struct PCA;

impl PCA {
    pub fn fit(data: &Array2<f64>, n_components: usize) -> Result<PCAResult, GreenersError> {
        let (n, p) = (data.nrows(), data.ncols());
        if n < 2 {
            return Err(GreenersError::InvalidOperation(
                "Need at least 2 observations for PCA".into(),
            ));
        }
        let nc = n_components.min(p);

        // Standardize
        let mut mean = Array1::<f64>::zeros(p);
        let mut std = Array1::<f64>::zeros(p);
        for j in 0..p {
            let col = data.column(j);
            mean[j] = col.mean().unwrap_or(0.0);
            let var = col.iter().map(|x| (x - mean[j]).powi(2)).sum::<f64>() / (n - 1) as f64;
            std[j] = var.sqrt().max(1e-15);
        }

        let mut z = data.clone();
        for (j, mut col) in z.axis_iter_mut(Axis(1)).enumerate() {
            col -= mean[j];
            col /= std[j];
        }

        // Correlation matrix = Z'Z / (n-1)
        let corr = z.t().dot(&z) / (n - 1) as f64;

        // Eigendecomposition (returns ascending order)
        let (eigenvalues, eigenvectors) = corr.eigh(UPLO::Upper)?;

        // Reverse to descending order
        let total_var: f64 = eigenvalues.iter().sum();
        let ev: Array1<f64> = eigenvalues.slice(s![..;-1]).to_owned();
        let evec: Array2<f64> = eigenvectors.slice(s![.., ..;-1]).to_owned();

        // Take top n_components
        let explained_variance = ev.slice(s![..nc]).to_owned();
        let explained_variance_ratio = explained_variance.mapv(|v| v / total_var.max(1e-15));
        let components = evec.slice(s![.., ..nc]).to_owned();

        // Loadings = components * sqrt(eigenvalue)
        let mut loadings = components.clone();
        for (j, mut col) in loadings.axis_iter_mut(Axis(1)).enumerate() {
            col *= explained_variance[j].sqrt();
        }

        // Scores
        let scores = z.dot(&components);

        Ok(PCAResult {
            components,
            explained_variance,
            explained_variance_ratio,
            loadings,
            scores,
            mean,
            std,
            n_obs: n,
            n_components: nc,
        })
    }
}

// ─── Factor Analysis ───────────────────────────────────────────────────────────

/// Rotation method for Factor Analysis.
#[derive(Debug, Clone)]
pub enum Rotation {
    None,
    Varimax,
}

/// Result of Factor Analysis.
#[derive(Debug)]
pub struct FactorResult {
    pub loadings: Array2<f64>,
    pub communalities: Array1<f64>,
    pub uniquenesses: Array1<f64>,
    pub eigenvalues: Array1<f64>,
    pub n_factors: usize,
    pub n_obs: usize,
}

impl fmt::Display for FactorResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^60}", " Factor Analysis ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10}", "Factors:", self.n_factors)?;
        writeln!(f, "\nCommunalities:")?;
        for (i, &c) in self.communalities.iter().enumerate() {
            writeln!(f, "  Var{}: {:.4}", i + 1, c)?;
        }
        writeln!(f, "{:=^60}", "")
    }
}

/// Factor Analysis via principal axis factoring.
pub struct FactorAnalysis;

impl FactorAnalysis {
    pub fn fit(
        data: &Array2<f64>,
        n_factors: usize,
        rotation: Rotation,
    ) -> Result<FactorResult, GreenersError> {
        let (n, p) = (data.nrows(), data.ncols());
        if n < 2 || n_factors > p {
            return Err(GreenersError::InvalidOperation(
                "Invalid dimensions for factor analysis".into(),
            ));
        }

        // Standardize
        let mut z = data.clone();
        for mut col in z.axis_iter_mut(Axis(1)) {
            let m = col.mean().unwrap_or(0.0);
            col -= m;
            let s = col.iter().map(|x| x * x).sum::<f64>() / (n - 1) as f64;
            let s = s.sqrt().max(1e-15);
            col /= s;
        }

        let corr = z.t().dot(&z) / (n - 1) as f64;
        let (eigenvalues_all, eigenvectors_all) = corr.eigh(UPLO::Upper)?;

        // Reverse to descending
        let eigenvalues: Array1<f64> = eigenvalues_all.slice(s![..;-1]).to_owned();
        let eigenvectors: Array2<f64> = eigenvectors_all.slice(s![.., ..;-1]).to_owned();

        // Initial loadings: L = V * sqrt(Lambda)
        let mut loadings = Array2::<f64>::zeros((p, n_factors));
        for j in 0..n_factors {
            let sqrt_ev = eigenvalues[j].max(0.0).sqrt();
            for i in 0..p {
                loadings[[i, j]] = eigenvectors[[i, j]] * sqrt_ev;
            }
        }

        // Varimax rotation
        if matches!(rotation, Rotation::Varimax) {
            loadings = varimax_rotation(&loadings, 100);
        }

        // Communalities: sum of squared loadings per variable
        let communalities: Array1<f64> = (0..p)
            .map(|i| (0..n_factors).map(|j| loadings[[i, j]].powi(2)).sum())
            .collect::<Vec<_>>()
            .into();

        let uniquenesses = communalities.mapv(|c| (1.0 - c).max(0.0));

        Ok(FactorResult {
            loadings,
            communalities,
            uniquenesses,
            eigenvalues: eigenvalues.slice(s![..n_factors]).to_owned(),
            n_factors,
            n_obs: n,
        })
    }
}

fn varimax_rotation(loadings: &Array2<f64>, max_iter: usize) -> Array2<f64> {
    let (p, k) = (loadings.nrows(), loadings.ncols());
    if k < 2 {
        return loadings.clone();
    }

    let mut rotated = loadings.clone();

    for _ in 0..max_iter {
        let mut changed = false;
        for i in 0..k {
            for j in (i + 1)..k {
                // Compute rotation angle for columns i and j
                let mut a = 0.0;
                let mut b = 0.0;
                let mut c = 0.0;
                let mut d = 0.0;

                for r in 0..p {
                    let li = rotated[[r, i]];
                    let lj = rotated[[r, j]];
                    let u = li * li - lj * lj;
                    let v = 2.0 * li * lj;
                    a += u;
                    b += v;
                    c += u * u - v * v;
                    d += 2.0 * u * v;
                }

                let num = d - 2.0 * a * b / p as f64;
                let den = c - (a * a - b * b) / p as f64;
                let angle = 0.25 * num.atan2(den);

                if angle.abs() < 1e-10 {
                    continue;
                }
                changed = true;

                let cos_a = angle.cos();
                let sin_a = angle.sin();

                for r in 0..p {
                    let li = rotated[[r, i]];
                    let lj = rotated[[r, j]];
                    rotated[[r, i]] = cos_a * li + sin_a * lj;
                    rotated[[r, j]] = -sin_a * li + cos_a * lj;
                }
            }
        }
        if !changed {
            break;
        }
    }

    rotated
}

// ─── MANOVA ────────────────────────────────────────────────────────────────────

/// Result of MANOVA test.
#[derive(Debug)]
pub struct ManovaResult {
    /// Wilks' Lambda
    pub wilks_lambda: f64,
    /// Pillai's trace
    pub pillai_trace: f64,
    /// Hotelling-Lawley trace
    pub hotelling_lawley: f64,
    /// Roy's largest root
    pub roys_largest_root: f64,
    /// Approximate F-values for each test
    pub f_values: [f64; 4],
    /// P-values for each test
    pub p_values: [f64; 4],
    pub n_obs: usize,
    pub n_groups: usize,
    pub n_vars: usize,
}

impl fmt::Display for ManovaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^70}", " One-Way MANOVA ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10}", "Groups:", self.n_groups)?;
        writeln!(f, "{:<20} {:>10}", "Variables:", self.n_vars)?;
        writeln!(
            f,
            "\n{:<24} {:>10} {:>10} {:>10}",
            "Test", "Statistic", "F", "P-value"
        )?;
        writeln!(f, "{:-^60}", "")?;
        let names = [
            "Wilks' Lambda",
            "Pillai's trace",
            "Hotelling-Lawley",
            "Roy's largest root",
        ];
        let stats = [
            self.wilks_lambda,
            self.pillai_trace,
            self.hotelling_lawley,
            self.roys_largest_root,
        ];
        for i in 0..4 {
            writeln!(
                f,
                "{:<24} {:>10.4} {:>10.4} {:>10.4}",
                names[i], stats[i], self.f_values[i], self.p_values[i]
            )?;
        }
        writeln!(f, "{:=^70}", "")
    }
}

/// One-way MANOVA.
pub struct MANOVA;

impl MANOVA {
    /// Fit one-way MANOVA.
    /// y_matrix: n x p matrix of dependent variables
    /// groups: group assignment for each observation (length n)
    pub fn fit(
        y_matrix: &Array2<f64>,
        groups: &Array1<usize>,
    ) -> Result<ManovaResult, GreenersError> {
        let (n, p) = (y_matrix.nrows(), y_matrix.ncols());
        if n != groups.len() {
            return Err(GreenersError::ShapeMismatch(
                "y_matrix rows must match groups length".into(),
            ));
        }

        // Find unique groups
        let mut unique_groups: Vec<usize> = groups.iter().cloned().collect();
        unique_groups.sort();
        unique_groups.dedup();
        let g = unique_groups.len();

        if g < 2 {
            return Err(GreenersError::InvalidOperation(
                "Need at least 2 groups for MANOVA".into(),
            ));
        }

        // Grand mean
        let grand_mean: Array1<f64> = y_matrix.mean_axis(Axis(0)).unwrap();

        // Between-groups (H) and within-groups (E) matrices
        let mut h_matrix = Array2::<f64>::zeros((p, p));
        let mut e_matrix = Array2::<f64>::zeros((p, p));

        for &grp in &unique_groups {
            // Indices for this group
            let idx: Vec<usize> = (0..n).filter(|&i| groups[i] == grp).collect();
            let ni = idx.len();
            if ni == 0 {
                continue;
            }

            // Group mean
            let mut group_mean = Array1::<f64>::zeros(p);
            for &i in &idx {
                group_mean = &group_mean + &y_matrix.row(i).to_owned();
            }
            group_mean /= ni as f64;

            // H += n_i * (mean_i - grand_mean)(mean_i - grand_mean)'
            let diff = &group_mean - &grand_mean;
            for a in 0..p {
                for b in 0..p {
                    h_matrix[[a, b]] += ni as f64 * diff[a] * diff[b];
                }
            }

            // E += sum_j (y_ij - mean_i)(y_ij - mean_i)'
            for &i in &idx {
                let d = &y_matrix.row(i).to_owned() - &group_mean;
                for a in 0..p {
                    for b in 0..p {
                        e_matrix[[a, b]] += d[a] * d[b];
                    }
                }
            }
        }

        // Eigenvalues of E^{-1} H
        let e_inv = e_matrix.inv()?;
        let m = e_inv.dot(&h_matrix);
        // Use symmetric eigendecomposition on (E^-1 H + H E^-1)/2 for stability
        let m_sym = (&m + &m.t()) * 0.5;
        let (eig_vals, _) = m_sym.eigh(UPLO::Upper)?;
        let mut lambdas: Vec<f64> = eig_vals.iter().cloned().collect();
        lambdas.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let s = p.min(g - 1);

        // Statistics
        let wilks_lambda: f64 = lambdas.iter().take(s).map(|&l| 1.0 / (1.0 + l)).product();
        let pillai_trace: f64 = lambdas.iter().take(s).map(|&l| l / (1.0 + l)).sum();
        let hotelling_lawley: f64 = lambdas.iter().take(s).sum();
        let roys_largest_root = lambdas.first().cloned().unwrap_or(0.0);

        // Approximate F-statistics
        let df_h = (g - 1) as f64;
        let df_e = (n - g) as f64;
        let pf = p as f64;

        // Wilks' Lambda F-approximation (Rao's F)
        let t = if pf * pf + df_h * df_h - 5.0 > 0.0 {
            ((pf * pf * df_h * df_h - 4.0) / (pf * pf + df_h * df_h - 5.0)).sqrt()
        } else {
            1.0
        };
        let df1_wilks = pf * df_h;
        let df2_wilks = (df_e + df_h - 0.5 * (pf + df_h + 1.0)) * t - 0.5 * (df1_wilks) + 1.0;
        let lambda_t = if t > 0.0 {
            wilks_lambda.powf(1.0 / t)
        } else {
            wilks_lambda
        };
        let f_wilks = if lambda_t < 1.0 {
            ((1.0 - lambda_t) / lambda_t) * (df2_wilks / df1_wilks)
        } else {
            0.0
        };

        // Pillai F-approximation
        let s_f = s as f64;
        let f_pillai = (pillai_trace / s_f)
            * ((df_e + s_f - pf + s_f * df_h) / ((s_f.max(1.0)) * (s_f * df_h)));
        let df1_pillai = s_f * pf * df_h / s_f.max(1.0);
        let df2_pillai = s_f * (df_e + s_f - pf);

        // Hotelling-Lawley F-approximation
        let f_hl = hotelling_lawley * df_e / (s_f * df_h * pf);
        let df1_hl = s_f * pf * df_h / s_f.max(1.0);
        let df2_hl = s_f * df_e;

        // Roy's largest root F-approximation
        let f_roy = roys_largest_root * df_e / pf.max(df_h);
        let df1_roy = pf.max(df_h);
        let df2_roy = df_e;

        let f_values = [f_wilks, f_pillai.max(0.0), f_hl.max(0.0), f_roy.max(0.0)];

        // P-values
        let p_values = [
            f_pvalue(f_wilks, df1_wilks, df2_wilks),
            f_pvalue(f_pillai, df1_pillai, df2_pillai),
            f_pvalue(f_hl, df1_hl, df2_hl),
            f_pvalue(f_roy, df1_roy, df2_roy),
        ];

        Ok(ManovaResult {
            wilks_lambda,
            pillai_trace,
            hotelling_lawley,
            roys_largest_root,
            f_values,
            p_values,
            n_obs: n,
            n_groups: g,
            n_vars: p,
        })
    }
}

// ─── Canonical Correlation Analysis ──────────────────────────────────────────

/// Result of Canonical Correlation Analysis.
#[derive(Debug)]
pub struct CanCorrResult {
    /// Canonical correlations (descending)
    pub cancorr: Array1<f64>,
    /// Weights for X variables
    pub x_weights: Array2<f64>,
    /// Weights for Y variables
    pub y_weights: Array2<f64>,
    /// Wilks' Lambda
    pub wilks_lambda: f64,
    /// Approximate F-statistic
    pub f_stat: f64,
    /// P-value
    pub p_value: f64,
    pub n_obs: usize,
}

impl fmt::Display for CanCorrResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^60}", " Canonical Correlation Analysis ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10.4}", "Wilks' Lambda:", self.wilks_lambda)?;
        writeln!(f, "{:<20} {:>10.4}", "F-statistic:", self.f_stat)?;
        writeln!(f, "{:<20} {:>10.4}", "P-value:", self.p_value)?;
        writeln!(f, "\nCanonical Correlations:")?;
        for (i, &c) in self.cancorr.iter().enumerate() {
            writeln!(f, "  CC{}: {:.4}", i + 1, c)?;
        }
        writeln!(f, "{:=^60}", "")
    }
}

/// Canonical Correlation Analysis.
pub struct CanCorr;

impl CanCorr {
    /// Fit CCA.
    ///
    /// - `x`: n x p matrix
    /// - `y`: n x q matrix
    pub fn fit(x: &Array2<f64>, y: &Array2<f64>) -> Result<CanCorrResult, GreenersError> {
        let (n, p) = (x.nrows(), x.ncols());
        let q = y.ncols();

        if n != y.nrows() {
            return Err(GreenersError::ShapeMismatch(
                "x and y must have same number of rows".into(),
            ));
        }
        if n < p + q + 1 {
            return Err(GreenersError::ShapeMismatch(
                "Not enough observations for CCA".into(),
            ));
        }

        let s = p.min(q);

        // Center
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        let y_mean = y.mean_axis(Axis(0)).unwrap();
        let mut xc = x.clone();
        let mut yc = y.clone();
        for (j, mut col) in xc.axis_iter_mut(Axis(1)).enumerate() {
            col -= x_mean[j];
        }
        for (j, mut col) in yc.axis_iter_mut(Axis(1)).enumerate() {
            col -= y_mean[j];
        }

        let nf = (n - 1) as f64;
        let sxx = xc.t().dot(&xc) / nf;
        let syy = yc.t().dot(&yc) / nf;
        let sxy = xc.t().dot(&yc) / nf;

        // Compute Sxx^{-1/2} via eigendecomposition
        let sxx_inv = sxx.inv()?;
        let syy_inv = syy.inv()?;

        // Eigenvalue problem: Sxx^{-1} Sxy Syy^{-1} Syx a = lambda^2 a
        let m = sxx_inv.dot(&sxy).dot(&syy_inv).dot(&sxy.t());
        let m_sym = (&m + &m.t()) * 0.5;
        let (eig_vals, eig_vecs) = m_sym.eigh(UPLO::Upper)?;

        // Sort descending
        let mut idx: Vec<usize> = (0..p).collect();
        idx.sort_by(|&a, &b| eig_vals[b].partial_cmp(&eig_vals[a]).unwrap());

        let cancorr: Array1<f64> = idx
            .iter()
            .take(s)
            .map(|&i| eig_vals[i].max(0.0).sqrt().min(1.0))
            .collect();

        let mut x_weights = Array2::<f64>::zeros((p, s));
        for (new_col, &old_col) in idx.iter().take(s).enumerate() {
            x_weights
                .column_mut(new_col)
                .assign(&eig_vecs.column(old_col));
        }

        // Y weights: Syy^{-1} Syx * x_weights, normalized
        let y_weights_raw = syy_inv.dot(&sxy.t()).dot(&x_weights);
        let mut y_weights = Array2::<f64>::zeros((q, s));
        for j in 0..s {
            let col = y_weights_raw.column(j);
            let norm = col.dot(&col).sqrt().max(1e-15);
            y_weights.column_mut(j).assign(&(&col / norm));
        }

        // Wilks' Lambda = product(1 - r_i^2)
        let wilks_lambda: f64 = cancorr.iter().map(|&r| 1.0 - r * r).product();

        // Approximate F-test (Rao's F)
        let pf = p as f64;
        let qf = q as f64;
        let nf_obs = n as f64;
        let t = if pf * pf * qf * qf - 4.0 > 0.0 {
            ((pf * pf * qf * qf - 4.0) / (pf * pf + qf * qf - 5.0)).sqrt()
        } else {
            1.0
        };
        let df1 = pf * qf;
        let df2 = ((nf_obs - 1.0 - 0.5 * (pf + qf + 1.0)) * t - 0.5 * df1 + 1.0).max(1.0);
        let lambda_t = if t > 0.0 {
            wilks_lambda.powf(1.0 / t)
        } else {
            wilks_lambda
        };
        let f_stat = if lambda_t < 1.0 {
            ((1.0 - lambda_t) / lambda_t) * (df2 / df1)
        } else {
            0.0
        };

        let p_value = f_pvalue(f_stat, df1, df2);

        Ok(CanCorrResult {
            cancorr,
            x_weights,
            y_weights,
            wilks_lambda,
            f_stat,
            p_value,
            n_obs: n,
        })
    }
}

fn f_pvalue(f: f64, df1: f64, df2: f64) -> f64 {
    if df1 <= 0.0 || df2 <= 0.0 || !f.is_finite() || f <= 0.0 {
        return 1.0;
    }
    match FisherSnedecor::new(df1, df2) {
        Ok(dist) => 1.0 - dist.cdf(f),
        Err(_) => 1.0,
    }
}
