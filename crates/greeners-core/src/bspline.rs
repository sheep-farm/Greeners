use crate::error::GreenersError;
use ndarray::{Array1, Array2};

/// B-spline basis generator.
pub struct BSplineBasis;

impl BSplineBasis {
    /// Generate B-spline basis matrix.
    /// x: predictor variable
    /// df: degrees of freedom (number of basis functions)
    /// degree: spline degree (3 = cubic, default)
    pub fn generate(
        x: &Array1<f64>,
        df: usize,
        degree: usize,
    ) -> Result<Array2<f64>, GreenersError> {
        let n = x.len();
        if df < degree + 1 {
            return Err(GreenersError::InvalidOperation(
                "df must be >= degree + 1".into(),
            ));
        }

        let n_knots = df - degree + 1;
        let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (x_max - x_min).max(1e-10);

        // Interior knots (equally spaced)
        let n_interior = n_knots.saturating_sub(2);
        let mut knots = Vec::new();

        // Add boundary knots repeated (degree + 1) times
        for _ in 0..=degree {
            knots.push(x_min - 0.01 * range);
        }
        for i in 1..=n_interior {
            knots.push(x_min + i as f64 * range / (n_interior + 1) as f64);
        }
        for _ in 0..=degree {
            knots.push(x_max + 0.01 * range);
        }

        //Evaluate B-spline bass using Boor's algorithm
        let mut basis = Array2::<f64>::zeros((n, df));

        for (idx, &xi) in x.iter().enumerate() {
            for j in 0..df {
                basis[[idx, j]] = bspline_basis(j, degree, xi, &knots);
            }
        }

        Ok(basis)
    }

    /// Generate second-difference penalty matrix for a given df.
    pub fn penalty_matrix(df: usize) -> Array2<f64> {
        if df < 3 {
            return Array2::eye(df);
        }
        // Second-order difference matrix D
        let m = df - 2;
        let mut d = Array2::<f64>::zeros((m, df));
        for i in 0..m {
            d[[i, i]] = 1.0;
            d[[i, i + 1]] = -2.0;
            d[[i, i + 2]] = 1.0;
        }
        d.t().dot(&d)
    }
}

fn bspline_basis(j: usize, degree: usize, x: f64, knots: &[f64]) -> f64 {
    if degree == 0 {
        return if x >= knots[j] && x < knots[j + 1] {
            1.0
        } else {
            0.0
        };
    }

    let mut left = 0.0;
    let denom_left = knots[j + degree] - knots[j];
    if denom_left.abs() > 1e-15 {
        left = (x - knots[j]) / denom_left * bspline_basis(j, degree - 1, x, knots);
    }

    let mut right = 0.0;
    let denom_right = knots[j + degree + 1] - knots[j + 1];
    if denom_right.abs() > 1e-15 {
        right =
            (knots[j + degree + 1] - x) / denom_right * bspline_basis(j + 1, degree - 1, x, knots);
    }

    left + right
}
