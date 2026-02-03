use crate::error::GreenersError;
use ndarray::{Array1, Array2};
use std::fmt;

/// Kernel function for density estimation and kernel regression.
#[derive(Debug, Clone)]
pub enum Kernel {
    Gaussian,
    Epanechnikov,
    Triangular,
    Uniform,
}

impl Kernel {
    fn evaluate(&self, u: f64) -> f64 {
        match self {
            Kernel::Gaussian => (-0.5 * u * u).exp() / (2.0 * std::f64::consts::PI).sqrt(),
            Kernel::Epanechnikov => {
                if u.abs() <= 1.0 {
                    0.75 * (1.0 - u * u)
                } else {
                    0.0
                }
            }
            Kernel::Triangular => {
                if u.abs() <= 1.0 {
                    1.0 - u.abs()
                } else {
                    0.0
                }
            }
            Kernel::Uniform => {
                if u.abs() <= 1.0 {
                    0.5
                } else {
                    0.0
                }
            }
        }
    }
}

// ─── KDE ───────────────────────────────────────────────────────────────────────

/// Result of Kernel Density Estimation.
#[derive(Debug)]
pub struct KDEResult {
    pub bandwidth: f64,
    pub support: Array1<f64>,
    pub density: Array1<f64>,
    pub n_obs: usize,
}

impl KDEResult {
    /// Evaluate density at given points.
    pub fn evaluate(&self, points: &Array1<f64>) -> Array1<f64> {
        // Interpolate from computed support/density
        points.mapv(|p| {
            // Find nearest support point
            let mut best_idx = 0;
            let mut best_dist = f64::MAX;
            for (i, &s) in self.support.iter().enumerate() {
                let d = (s - p).abs();
                if d < best_dist {
                    best_dist = d;
                    best_idx = i;
                }
            }
            self.density[best_idx]
        })
    }
}

impl fmt::Display for KDEResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^60}", " Kernel Density Estimation ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10.6}", "Bandwidth:", self.bandwidth)?;
        writeln!(
            f,
            "{:<20} {:>10.6}",
            "Max density:",
            self.density.iter().cloned().fold(0.0_f64, f64::max)
        )?;
        writeln!(f, "{:=^60}", "")
    }
}

/// Univariate Kernel Density Estimation.
pub struct KDEUnivariate;

impl KDEUnivariate {
    /// Fit KDE. If bandwidth is None, uses Silverman's rule.
    pub fn fit(
        data: &Array1<f64>,
        bandwidth: Option<f64>,
        kernel: Kernel,
    ) -> Result<KDEResult, GreenersError> {
        let n = data.len();
        if n < 2 {
            return Err(GreenersError::InvalidOperation(
                "Need at least 2 data points for KDE".into(),
            ));
        }

        // Silverman's rule of thumb
        let bw = bandwidth.unwrap_or_else(|| {
            let mean = data.mean().unwrap_or(0.0);
            let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            let std = var.sqrt();

            // IQR-based estimate
            let mut sorted: Vec<f64> = data.iter().cloned().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q1 = sorted[n / 4];
            let q3 = sorted[3 * n / 4];
            let iqr = q3 - q1;

            let a = std.min(iqr / 1.34);
            0.9 * a * (n as f64).powf(-0.2)
        });

        let bw = bw.max(1e-10);

        // Support: range +/- 3*bw
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let n_points = 512;
        let lo = min_val - 3.0 * bw;
        let hi = max_val + 3.0 * bw;
        let step = (hi - lo) / (n_points - 1) as f64;

        let support: Array1<f64> = Array1::from(
            (0..n_points)
                .map(|i| lo + i as f64 * step)
                .collect::<Vec<_>>(),
        );

        let density: Array1<f64> = support.mapv(|s| {
            let sum: f64 = data.iter().map(|&xi| kernel.evaluate((s - xi) / bw)).sum();
            sum / (n as f64 * bw)
        });

        Ok(KDEResult {
            bandwidth: bw,
            support,
            density,
            n_obs: n,
        })
    }
}

// ─── LOWESS ────────────────────────────────────────────────────────────────────

/// Result of LOWESS smoothing.
#[derive(Debug)]
pub struct LowessResult {
    pub smoothed: Array1<f64>,
    pub residuals: Array1<f64>,
    pub n_obs: usize,
    pub frac: f64,
}

impl fmt::Display for LowessResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^60}", " LOWESS Smoothing ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10.4}", "Fraction:", self.frac)?;
        writeln!(f, "{:=^60}", "")
    }
}

/// Locally Weighted Scatterplot Smoothing.
pub struct Lowess;

impl Lowess {
    /// Fit LOWESS smoother.
    /// frac: fraction of data used for each local fit (0 < frac <= 1)
    /// it: number of robustifying iterations (0 for no robustification)
    pub fn fit(
        y: &Array1<f64>,
        x: &Array1<f64>,
        frac: f64,
        it: usize,
    ) -> Result<LowessResult, GreenersError> {
        let n = y.len();
        if n != x.len() {
            return Err(GreenersError::ShapeMismatch(
                "y and x length mismatch".into(),
            ));
        }
        if n < 3 {
            return Err(GreenersError::InvalidOperation(
                "Need at least 3 observations".into(),
            ));
        }

        let span = ((frac * n as f64).ceil() as usize).max(2).min(n);

        // Sort by x
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());

        let x_sorted: Vec<f64> = indices.iter().map(|&i| x[i]).collect();
        let y_sorted: Vec<f64> = indices.iter().map(|&i| y[i]).collect();

        let mut weights = vec![1.0; n];
        let mut smoothed_sorted = vec![0.0; n];

        for _robustness_iter in 0..=(it) {
            smoothed_sorted = loess_smooth(&x_sorted, &y_sorted, &weights, span);

            if _robustness_iter < it {
                // Compute robustness weights
                let resid: Vec<f64> = (0..n)
                    .map(|i| (y_sorted[i] - smoothed_sorted[i]).abs())
                    .collect();
                let mut sorted_resid = resid.clone();
                sorted_resid.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median_resid = sorted_resid[n / 2];
                let u_scale = 6.0 * median_resid;

                weights = resid
                    .iter()
                    .map(|&r| {
                        let u = r / u_scale.max(1e-15);
                        if u < 1.0 {
                            (1.0 - u * u).powi(2)
                        } else {
                            0.0
                        }
                    })
                    .collect();
            }
        }

        // Unsort back to original order
        let mut smoothed = Array1::<f64>::zeros(n);
        for (sorted_idx, &orig_idx) in indices.iter().enumerate() {
            smoothed[orig_idx] = smoothed_sorted[sorted_idx];
        }

        let residuals = y - &smoothed;

        Ok(LowessResult {
            smoothed,
            residuals,
            n_obs: n,
            frac,
        })
    }
}

fn loess_smooth(x: &[f64], y: &[f64], w: &[f64], span: usize) -> Vec<f64> {
    let n = x.len();
    let h = span.min(n);

    (0..n)
        .map(|idx| {
            let xp = x[idx];

            let mut dists: Vec<(usize, f64)> = (0..n).map(|i| (i, (x[i] - xp).abs())).collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let max_dist = dists[h - 1].1.max(1e-15);

            let mut sum_w = 0.0;
            let mut sum_wx = 0.0;
            let mut sum_wy = 0.0;
            let mut sum_wxx = 0.0;
            let mut sum_wxy = 0.0;

            for &(i, d) in dists.iter().take(h) {
                if w[i] <= 0.0 {
                    continue;
                }
                let u = d / max_dist;
                let kernel = if u < 1.0 {
                    (1.0 - u.powi(3)).powi(3)
                } else {
                    0.0
                };
                let wi = kernel * w[i];
                let xi = x[i] - xp;

                sum_w += wi;
                sum_wx += wi * xi;
                sum_wy += wi * y[i];
                sum_wxx += wi * xi * xi;
                sum_wxy += wi * xi * y[i];
            }

            if sum_w < 1e-15 {
                return y.iter().sum::<f64>() / n as f64;
            }

            let det = sum_w * sum_wxx - sum_wx * sum_wx;
            if det.abs() < 1e-15 {
                sum_wy / sum_w
            } else {
                (sum_wxx * sum_wy - sum_wx * sum_wxy) / det
            }
        })
        .collect()
}

// ─── KernelReg ─────────────────────────────────────────────────────────────────

/// Result of Nadaraya-Watson kernel regression.
#[derive(Debug)]
pub struct KernelRegResult {
    pub fitted: Array1<f64>,
    pub residuals: Array1<f64>,
    pub bandwidth: f64,
    pub n_obs: usize,
}

impl fmt::Display for KernelRegResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^60}", " Kernel Regression (Nadaraya-Watson) ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10.6}", "Bandwidth:", self.bandwidth)?;
        writeln!(f, "{:=^60}", "")
    }
}

/// Nadaraya-Watson kernel regression estimator.
pub struct KernelReg;

impl KernelReg {
    /// Fit kernel regression. If bandwidth is None, uses rule-of-thumb.
    pub fn fit(
        y: &Array1<f64>,
        x: &Array1<f64>,
        bandwidth: Option<f64>,
        kernel: Kernel,
    ) -> Result<KernelRegResult, GreenersError> {
        let n = y.len();
        if n != x.len() {
            return Err(GreenersError::ShapeMismatch(
                "y and x length mismatch".into(),
            ));
        }
        if n < 2 {
            return Err(GreenersError::InvalidOperation(
                "Need at least 2 observations".into(),
            ));
        }

        let bw = bandwidth.unwrap_or_else(|| {
            let mean = x.mean().unwrap_or(0.0);
            let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            let std = var.sqrt();
            1.06 * std * (n as f64).powf(-0.2)
        });
        let bw = bw.max(1e-10);

        // Nadaraya-Watson: f(x0) = sum(K((x-x0)/h) * y) / sum(K((x-x0)/h))
        let fitted: Array1<f64> = x.mapv(|x0| {
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..n {
                let k = kernel.evaluate((x[i] - x0) / bw);
                num += k * y[i];
                den += k;
            }
            if den > 1e-15 {
                num / den
            } else {
                y.mean().unwrap_or(0.0)
            }
        });

        let residuals = y - &fitted;

        Ok(KernelRegResult {
            fitted,
            residuals,
            bandwidth: bw,
            n_obs: n,
        })
    }
}

// ─── KDEMultivariate ────────────────────────────────────────────────────────────

/// Result of multivariate kernel density estimation.
#[derive(Debug)]
pub struct KDEMultivariateResult {
    pub bandwidths: Array1<f64>,
    pub n_obs: usize,
    pub n_dims: usize,
    _data: Array2<f64>,
    _kernel: Kernel,
}

impl KDEMultivariateResult {
    /// Evaluate density at given points (m x d matrix).
    pub fn evaluate(&self, points: &Array2<f64>) -> Array1<f64> {
        let n = self._data.nrows();
        let d = self._data.ncols();
        let m = points.nrows();
        let mut density = Array1::<f64>::zeros(m);

        for i in 0..m {
            let mut sum = 0.0;
            for j in 0..n {
                let mut prod = 1.0;
                for dim in 0..d {
                    let u = (points[[i, dim]] - self._data[[j, dim]]) / self.bandwidths[dim];
                    prod *= self._kernel.evaluate(u) / self.bandwidths[dim];
                }
                sum += prod;
            }
            density[i] = sum / n as f64;
        }

        density
    }
}

impl fmt::Display for KDEMultivariateResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^60}", " Multivariate KDE ")?;
        writeln!(f, "{:<20} {:>10}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>10}", "Dimensions:", self.n_dims)?;
        writeln!(f, "Bandwidths: {:?}", self.bandwidths)?;
        writeln!(f, "{:=^60}", "")
    }
}

/// Multivariate Kernel Density Estimation via product kernel.
pub struct KDEMultivariate;

impl KDEMultivariate {
    /// Fit multivariate KDE.
    ///
    /// - `data`: n x d matrix
    /// - `bandwidths`: per-dimension bandwidths (None = Silverman rule per dim)
    /// - `kernel`: kernel function
    pub fn fit(
        data: &Array2<f64>,
        bandwidths: Option<&Array1<f64>>,
        kernel: Kernel,
    ) -> Result<KDEMultivariateResult, GreenersError> {
        let (n, d) = (data.nrows(), data.ncols());
        if n < 2 {
            return Err(GreenersError::InvalidOperation(
                "Need at least 2 data points for KDE".into(),
            ));
        }

        let bw = match bandwidths {
            Some(b) => {
                if b.len() != d {
                    return Err(GreenersError::ShapeMismatch(
                        "Bandwidth length must match data dimensions".into(),
                    ));
                }
                b.clone()
            }
            None => {
                // Silverman's rule per dimension
                let factor = (4.0 / ((d + 2) as f64)).powf(1.0 / (d as f64 + 4.0))
                    * (n as f64).powf(-1.0 / (d as f64 + 4.0));
                let mut bw = Array1::<f64>::zeros(d);
                for j in 0..d {
                    let col = data.column(j);
                    let mean = col.mean().unwrap_or(0.0);
                    let var = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
                    bw[j] = (var.sqrt() * factor).max(1e-10);
                }
                bw
            }
        };

        Ok(KDEMultivariateResult {
            bandwidths: bw,
            n_obs: n,
            n_dims: d,
            _data: data.clone(),
            _kernel: kernel,
        })
    }
}
