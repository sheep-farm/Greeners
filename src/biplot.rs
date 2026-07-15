//! PCA Biplot (Gabriel 1971).
//!
//! Simultaneous visualization of observations (scores) and
//! variables (loadings) on the same plot. The biplot projects
//! both rows and columns of a data matrix onto a low-dimensional
//! subspace.
//!
//! Two types:
//!   - Form biplot: row principal (GH biplot)
//!   - Covariance biplot: column principal (JK biplot)
//!
//! Reports scores, loadings, explained variance, and ASCII
//! visualization of the biplot.

use crate::multivariate::PCA;
use crate::GreenersError;
use ndarray::{s, Array1, Array2};
use std::fmt;

/// Biplot type.
#[derive(Debug, Clone, Copy)]
pub enum BiplotType {
    /// Row principal (GH biplot): preserves distances between observations
    Form,
    /// Column principal (JK biplot): preserves distances between variables
    Covariance,
    /// Symmetric (SVD): compromises both
    Symmetric,
}

impl BiplotType {
    fn as_str(&self) -> &str {
        match self {
            BiplotType::Form => "Form (row principal)",
            BiplotType::Covariance => "Covariance (column principal)",
            BiplotType::Symmetric => "Symmetric (SVD)",
        }
    }
}

/// Result of PCA biplot.
#[derive(Debug)]
pub struct BiplotResult {
    /// Observation scores (n x 2)
    pub scores: Array2<f64>,
    /// Variable loadings (p x 2)
    pub loadings: Array2<f64>,
    /// Explained variance ratio per component
    pub explained_variance_ratio: Array1<f64>,
    /// Cumulative explained variance
    pub cumulative_variance: Array1<f64>,
    /// Variable names
    pub variable_names: Vec<String>,
    /// Observation labels (1..n)
    pub obs_labels: Vec<String>,
    /// Biplot type
    pub biplot_type: BiplotType,
    /// Number of observations
    pub n_obs: usize,
    /// Number of variables
    pub n_vars: usize,
    /// ASCII biplot
    pub ascii_biplot: String,
}

impl fmt::Display for BiplotResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:=^78}", " PCA Biplot ")?;
        writeln!(f, "Gabriel (1971)")?;
        writeln!(f, "Type: {}", self.biplot_type.as_str())?;
        writeln!(f, "{:<20} {:>12}", "Observations:", self.n_obs)?;
        writeln!(f, "{:<20} {:>12}", "Variables:", self.n_vars)?;

        // Explained variance
        writeln!(f, "\n{:-^78}", "")?;
        writeln!(f, "  Explained variance:")?;
        writeln!(
            f,
            "  {:<10} {:>12} {:>12}",
            "Component", "Variance %", "Cumulative %"
        )?;
        writeln!(f, "{:-^78}", "")?;
        for i in 0..self.explained_variance_ratio.len() {
            writeln!(
                f,
                "  {:<10} {:>12.2}% {:>12.2}%",
                i + 1,
                self.explained_variance_ratio[i] * 100.0,
                self.cumulative_variance[i] * 100.0
            )?;
        }

        // Variable loadings
        writeln!(f, "\n  Variable loadings:")?;
        writeln!(f, "  {:<14} {:>10} {:>10}", "Variable", "Dim 1", "Dim 2")?;
        writeln!(f, "{:-^78}", "")?;
        for (j, name) in self.variable_names.iter().enumerate() {
            writeln!(
                f,
                "  {:<14} {:>10.4} {:>10.4}",
                name,
                self.loadings[(j, 0)],
                self.loadings[(j, 1)]
            )?;
        }

        // Observation scores (first 10)
        writeln!(f, "\n  Observation scores (first 10):")?;
        writeln!(f, "  {:<6} {:>10} {:>10}", "Obs", "Dim 1", "Dim 2")?;
        writeln!(f, "{:-^78}", "")?;
        let n_show = 10.min(self.n_obs);
        for i in 0..n_show {
            writeln!(
                f,
                "  {:<6} {:>10.4} {:>10.4}",
                i + 1,
                self.scores[(i, 0)],
                self.scores[(i, 1)]
            )?;
        }

        // ASCII biplot
        writeln!(f, "\n  ASCII Biplot:")?;
        write!(f, "{}", self.ascii_biplot)?;

        write!(f, "\n{:=^78}", "")
    }
}

pub struct Biplot;

impl Biplot {
    /// Compute PCA biplot.
    ///
    /// # Arguments
    /// * `x` - Data matrix (n x p)
    /// * `biplot_type` - Type of biplot
    /// * `variable_names` - Optional variable names
    pub fn fit(
        x: &Array2<f64>,
        biplot_type: BiplotType,
        variable_names: Option<Vec<String>>,
    ) -> Result<BiplotResult, GreenersError> {
        let n = x.nrows();
        let p = x.ncols();
        if n < 3 || p < 2 {
            return Err(GreenersError::InvalidOperation(
                "Biplot: need at least 3 observations and 2 variables".into(),
            ));
        }

        let names =
            variable_names.unwrap_or_else(|| (0..p).map(|i| format!("x{}", i + 1)).collect());
        let obs_labels: Vec<String> = (1..=n).map(|i| format!("{}", i)).collect();

        // Run PCA with 2 components
        let pca = PCA::fit(x, 2)?;

        // Scale scores and loadings based on biplot type
        // pca.components has shape [p, nc] (variables x components)
        // pca.scores has shape [n, nc] (observations x components)
        let (scores, loadings) = match biplot_type {
            BiplotType::Form => {
                // Row principal: scores = U * D, loadings = V
                // Scores scaled by sqrt(eigenvalue), loadings unscaled
                let scores = pca.scores.clone();
                let loadings = pca.components.clone();
                (scores, loadings)
            }
            BiplotType::Covariance => {
                // Column principal: scores = U, loadings = V * D
                // Loadings scaled by sqrt(eigenvalue), scores unscaled
                let mut loadings = pca.components.clone();
                for j in 0..loadings.nrows() {
                    for f in 0..2.min(loadings.ncols()) {
                        loadings[(j, f)] *= pca.explained_variance[f].sqrt();
                    }
                }
                // Unscale scores
                let mut scores = pca.scores.clone();
                for f in 0..2.min(scores.ncols()) {
                    let scale = pca.explained_variance[f].sqrt().max(1e-10);
                    for i in 0..n {
                        scores[(i, f)] /= scale;
                    }
                }
                (scores, loadings)
            }
            BiplotType::Symmetric => {
                // Symmetric: both scaled by D^{1/2}
                let mut scores = pca.scores.clone();
                let mut loadings = pca.components.clone();
                for f in 0..2.min(scores.ncols()) {
                    let scale = pca.explained_variance[f].powf(0.25);
                    for i in 0..n {
                        scores[(i, f)] /= pca.explained_variance[f].sqrt().max(1e-10);
                        scores[(i, f)] *= scale;
                    }
                    for j in 0..p {
                        loadings[(j, f)] *= pca.explained_variance[f].sqrt();
                        loadings[(j, f)] /= scale.max(1e-10);
                    }
                }
                (scores, loadings)
            }
        };

        // Ensure 2 columns
        let scores_2d = if scores.ncols() >= 2 {
            scores.slice(s![.., 0..2]).to_owned()
        } else {
            let mut s = Array2::zeros((n, 2));
            for i in 0..n {
                s[(i, 0)] = scores[(i, 0)];
            }
            s
        };
        let loadings_2d = if loadings.ncols() >= 2 {
            loadings.slice(s![.., 0..2]).to_owned()
        } else {
            let mut l = Array2::zeros((p, 2));
            for j in 0..p {
                l[(j, 0)] = loadings[(j, 0)];
            }
            l
        };

        // Cumulative variance
        let mut cum_var = Array1::zeros(pca.explained_variance_ratio.len());
        let mut cum = 0.0;
        for i in 0..cum_var.len() {
            cum += pca.explained_variance_ratio[i];
            cum_var[i] = cum;
        }

        // Generate ASCII biplot
        let ascii = Self::render_ascii(&scores_2d, &loadings_2d, &names, &obs_labels);

        Ok(BiplotResult {
            scores: scores_2d,
            loadings: loadings_2d,
            explained_variance_ratio: pca.explained_variance_ratio,
            cumulative_variance: cum_var,
            variable_names: names,
            obs_labels,
            biplot_type,
            n_obs: n,
            n_vars: p,
            ascii_biplot: ascii,
        })
    }

    fn render_ascii(
        scores: &Array2<f64>,
        loadings: &Array2<f64>,
        var_names: &[String],
        obs_labels: &[String],
    ) -> String {
        let width = 60;
        let height = 25;
        let mut grid = vec![vec![' '; width]; height];

        // Find data range
        let mut all_vals: Vec<(f64, f64, char, String)> = Vec::new();

        // Normalize scores to [-1, 1]
        let mut max_score = 0.0_f64;
        for i in 0..scores.nrows() {
            for j in 0..2 {
                max_score = max_score.max(scores[(i, j)].abs());
            }
        }
        max_score = max_score.max(1e-10);

        for i in 0..scores.nrows() {
            let x = scores[(i, 0)] / max_score;
            let y = scores[(i, 1)] / max_score;
            all_vals.push((x, y, 'o', obs_labels[i].clone()));
        }

        // Normalize loadings to [-0.9, 0.9]
        let mut max_load = 0.0_f64;
        for j in 0..loadings.nrows() {
            for f in 0..2 {
                max_load = max_load.max(loadings[(j, f)].abs());
            }
        }
        max_load = max_load.max(1e-10);

        for j in 0..loadings.nrows() {
            let x = loadings[(j, 0)] / max_load * 0.85;
            let y = loadings[(j, 1)] / max_load * 0.85;
            let ch = (b'A' + j as u8) as char;
            all_vals.push((x, y, ch, var_names[j].clone()));
        }

        // Draw axes
        let cx = width / 2;
        let cy = height / 2;
        for cell in grid[cy].iter_mut().take(width) {
            *cell = '-';
        }
        for row in grid.iter_mut().take(height) {
            row[cx] = '|';
        }
        grid[cy][cx] = '+';

        // Plot points
        for (x, y, ch, _) in &all_vals {
            let px = cx + (*x * cx as f64 * 0.9) as usize;
            let py = cy - (*y * cy as f64 * 0.9) as usize;
            if px < width && py < height {
                grid[py][px] = *ch;
            }
        }

        // Build string
        let mut s = String::new();
        s.push_str("  Legend: o=observations, A,B,C...=variables\n");
        for row in &grid {
            s.push_str("  ");
            s.push_str(row.iter().collect::<String>().trim_end());
            s.push('\n');
        }

        s
    }
}
