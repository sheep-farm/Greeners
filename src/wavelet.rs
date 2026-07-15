//! Maximal Overlap Discrete Wavelet Transform (MODWT).
//!
//! Decomposes a time series into multiple scales (time-frequency
//! analysis). Unlike the DWT, MODWT does not decimate, producing
//! coefficients at every time point for every scale.
//!
//! Uses the Haar wavelet (daubechies 1) for simplicity. The MODWT
//! produces:
//!   - W_j: wavelet coefficients at scale j (high-frequency detail)
//!   - V_j: scaling coefficients at scale j (low-frequency smooth)
//!
//! The original series can be reconstructed as:
//!   x_t = sum_j W_j + V_J (multiresolution decomposition)
//!
//! References: Percival & Walden (2000), "Wavelet Methods for Time
//! Series Analysis".

use crate::GreenersError;
use ndarray::Array1;
use std::fmt;

/// Result of MODWT decomposition.
#[derive(Debug)]
pub struct ModwtResult {
    /// Wavelet coefficients at each scale, shape (n_scales, T)
    pub wavelet_coeffs: Vec<Array1<f64>>,
    /// Scaling coefficients at the final scale (T)
    pub scaling_coeffs: Array1<f64>,
    /// Detail (reconstructed) components at each scale, shape (n_scales, T)
    pub details: Vec<Array1<f64>>,
    /// Smooth component (reconstructed from V_J)
    pub smooth: Array1<f64>,
    /// Energy at each scale
    pub energy: Vec<f64>,
    /// Number of scales
    pub n_scales: usize,
    /// Length of input series
    pub n_obs: usize,
    /// Wavelet name
    pub wavelet: String,
}

impl fmt::Display for ModwtResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " MODWT (Maximal Overlap DWT) ")?;
        writeln!(f, "Wavelet: {} (Haar)", self.wavelet)?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Scales:", self.n_scales)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<12} {:>12} {:>12} {:>12}",
            "Scale", "Level", "Energy", "% Total"
        )?;
        writeln!(f, "{:-^78}", "")?;
        let total_energy: f64 = self.energy.iter().sum();
        for j in 0..self.n_scales {
            let pct = if total_energy > 1e-15 {
                self.energy[j] / total_energy * 100.0
            } else {
                0.0
            };
            writeln!(
                f,
                "{:<12} {:>12} {:>12.6} {:>11.4}%",
                format!("W_{}", j + 1),
                j + 1,
                self.energy[j],
                pct
            )?;
        }
        let smooth_energy: f64 = self.scaling_coeffs.mapv(|v| v * v).sum();
        let pct = if total_energy + smooth_energy > 1e-15 {
            smooth_energy / (total_energy + smooth_energy) * 100.0
        } else {
            0.0
        };
        writeln!(
            f,
            "{:<12} {:>12} {:>12.6} {:>11.4}%",
            format!("V_{}", self.n_scales),
            self.n_scales,
            smooth_energy,
            pct
        )?;

        writeln!(f, "\n  Decomposition summary:")?;
        writeln!(f, "  W_1: highest frequency (period ~2)")?;
        for j in 1..self.n_scales {
            writeln!(f, "  W_{}: period ~{}", j + 1, 2_i32.pow(j as u32 + 1))?;
        }
        writeln!(
            f,
            "  V_{}: trend (period > {})",
            self.n_scales,
            2_i32.pow(self.n_scales as u32 + 1)
        )?;

        write!(f, "{:=^78}", "")
    }
}

pub struct MODWT;

impl MODWT {
    /// Compute MODWT using Haar wavelet.
    ///
    /// # Arguments
    /// * `x` - Input time series (T)
    /// * `n_scales` - Number of decomposition scales (J)
    pub fn fit(x: &Array1<f64>, n_scales: usize) -> Result<ModwtResult, GreenersError> {
        let t = x.len();
        if t < 4 {
            return Err(GreenersError::InvalidOperation(
                "MODWT: need at least 4 observations".into(),
            ));
        }
        let max_scales = (t as f64).log2().floor() as usize;
        let j = n_scales.min(max_scales);
        if j == 0 {
            return Err(GreenersError::InvalidOperation(
                "MODWT: n_scales must be >= 1".into(),
            ));
        }

        // Haar wavelet filters (MODWT version, normalized by 1/sqrt(2))
        let h = [1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()]; // scaling filter
        let g = [1.0 / 2.0_f64.sqrt(), -1.0 / 2.0_f64.sqrt()]; // wavelet filter

        let mut v_prev = x.clone();
        let mut wavelet_coeffs: Vec<Array1<f64>> = Vec::with_capacity(j);
        let mut all_v: Vec<Array1<f64>> = Vec::with_capacity(j + 1);
        all_v.push(x.clone());

        for level in 0..j {
            let scale = 2_i32.pow(level as u32) as usize;
            let mut w = Array1::zeros(t);
            let mut v = Array1::zeros(t);

            for n_i in 0..t {
                // MODWT: no decimation, circular convolution
                let idx_w = (n_i as i64 - scale as i64).rem_euclid(t as i64) as usize;
                let idx_v = (n_i as i64 - scale as i64).rem_euclid(t as i64) as usize;

                w[n_i] = g[0] * v_prev[n_i] + g[1] * v_prev[idx_w];
                v[n_i] = h[0] * v_prev[n_i] + h[1] * v_prev[idx_v];
            }

            wavelet_coeffs.push(w.clone());
            all_v.push(v.clone());
            v_prev = v;
        }

        let scaling_coeffs = v_prev.clone();

        // Reconstruct detail and smooth components via inverse MODWT
        // Detail_j = inverse of W_j with zeros elsewhere
        // Smooth = inverse of V_J
        let mut details: Vec<Array1<f64>> = Vec::with_capacity(j);
        for (level, w_coeff) in wavelet_coeffs.iter().enumerate().take(j) {
            let detail = Self::reconstruct_detail(w_coeff, level, j, t);
            details.push(detail);
        }
        let smooth = Self::reconstruct_smooth(&scaling_coeffs, j, t);

        // Energy at each scale
        let energy: Vec<f64> = wavelet_coeffs
            .iter()
            .map(|w| w.mapv(|v| v * v).sum())
            .collect();

        Ok(ModwtResult {
            wavelet_coeffs,
            scaling_coeffs,
            details,
            smooth,
            energy,
            n_scales: j,
            n_obs: t,
            wavelet: "haar".to_string(),
        })
    }

    /// Reconstruct detail component from wavelet coefficients at a given level.
    fn reconstruct_detail(
        w: &Array1<f64>,
        level: usize,
        _max_level: usize,
        t: usize,
    ) -> Array1<f64> {
        // Inverse MODWT: start from W_j (with V_j = 0) and reconstruct
        let h = [1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()];
        let g = [1.0 / 2.0_f64.sqrt(), -1.0 / 2.0_f64.sqrt()];

        // Start with W at level `level`, V = 0
        let mut w_curr = w.clone();
        let mut v_curr: Array1<f64> = Array1::zeros(t);

        // Reconstruct from level `level` up to level 0
        for lvl in (0..=level).rev() {
            let scale = 2_i32.pow(lvl as u32) as usize;
            let mut w_new = Array1::zeros(t);
            let mut v_new = Array1::zeros(t);

            for n_i in 0..t {
                let idx = (n_i as i64 + scale as i64).rem_euclid(t as i64) as usize;
                w_new[n_i] = h[0] * w_curr[n_i] - g[0] * v_curr[n_i] + h[1] * w_curr[idx]
                    - g[1] * v_curr[idx];
                v_new[n_i] = g[0] * w_curr[n_i]
                    + h[0] * v_curr[n_i]
                    + g[1] * w_curr[idx]
                    + h[1] * v_curr[idx];
            }
            // Simplified: just use w_new as detail
            w_curr = w_new;
            v_curr = v_new;
        }

        // The detail is the reconstructed signal from W_j alone
        w_curr
    }

    /// Reconstruct smooth component from scaling coefficients.
    fn reconstruct_smooth(v: &Array1<f64>, max_level: usize, t: usize) -> Array1<f64> {
        // Inverse MODWT from V_J (with W_j = 0 for all j)
        let h = [1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()];

        let mut v_curr = v.clone();

        for lvl in (0..max_level).rev() {
            let scale = 2_i32.pow(lvl as u32) as usize;
            let mut v_new = Array1::zeros(t);

            for n_i in 0..t {
                let idx = (n_i as i64 + scale as i64).rem_euclid(t as i64) as usize;
                v_new[n_i] = h[0] * v_curr[n_i] + h[1] * v_curr[idx];
            }
            v_curr = v_new;
        }

        v_curr
    }
}
