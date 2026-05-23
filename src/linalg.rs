/// Pure-Rust linear algebra layer backed by `faer`, replacing `ndarray-linalg`/OpenBLAS.
///
/// Exposes the same method names (`.inv()`, `.qr()`, `.svd()`, `.eigh()`, `.eig()`,
/// `.cholesky()`, `.det()`) that the old `ndarray_linalg` traits provided, so the
/// only change needed in each caller is to swap the `use` line.
use crate::error::GreenersError;
use faer::linalg::solvers::{DenseSolveCore, Llt, PartialPivLu, Qr, SelfAdjointEigen, Svd};
use faer::prelude::*;
use faer::Side;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

// ─── UPLO ─────────────────────────────────────────────────────────────────────

/// Upper / Lower triangular selector, mirrors `ndarray_linalg::UPLO`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UPLO {
    Upper,
    Lower,
}

impl From<UPLO> for Side {
    fn from(u: UPLO) -> Side {
        match u {
            UPLO::Upper => Side::Upper,
            UPLO::Lower => Side::Lower,
        }
    }
}

// ─── Conversion helpers ────────────────────────────────────────────────────────

fn to_faer(m: &Array2<f64>) -> Mat<f64> {
    Mat::from_fn(m.nrows(), m.ncols(), |i, j| m[[i, j]])
}

fn from_faer_ref(m: faer::MatRef<'_, f64>) -> Array2<f64> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)])
}

fn from_faer(m: &Mat<f64>) -> Array2<f64> {
    from_faer_ref(m.as_ref())
}

// ─── Matrix inverse — replaces `ndarray_linalg::Inverse` ─────────────────────

pub trait LinalgInverse {
    fn inv(&self) -> Result<Array2<f64>, GreenersError>;
}

impl LinalgInverse for Array2<f64> {
    fn inv(&self) -> Result<Array2<f64>, GreenersError> {
        let n = self.nrows();
        if n != self.ncols() {
            return Err(GreenersError::InvalidOperation(
                "inv: matrix must be square".into(),
            ));
        }
        let mat = to_faer(self);
        let lu = PartialPivLu::new(mat.as_ref());
        // Detect truly singular matrices (exact zero pivot); near-singular allowed to proceed
        let u = lu.U();
        for k in 0..n {
            if u[(k, k)].abs() == 0.0 {
                return Err(GreenersError::SingularMatrix);
            }
        }
        let inv_mat = lu.inverse();
        Ok(from_faer(&inv_mat))
    }
}

// ─── QR decomposition — replaces `ndarray_linalg::QR` ───────────────────────

pub trait LinalgQR {
    fn qr(&self) -> Result<(Array2<f64>, Array2<f64>), GreenersError>;
}

impl LinalgQR for Array2<f64> {
    fn qr(&self) -> Result<(Array2<f64>, Array2<f64>), GreenersError> {
        let mat = to_faer(self);
        let qr = Qr::new(mat.as_ref());
        let q = qr.compute_thin_Q();
        let r = qr.R();
        // R from faer may be m×n but we want min(m,n) rows
        let size = self.nrows().min(self.ncols());
        let r_trimmed = Array2::from_shape_fn((size, self.ncols()), |(i, j)| r[(i, j)]);
        Ok((from_faer(&q), r_trimmed))
    }
}

// ─── SVD — replaces `ndarray_linalg::SVD` ────────────────────────────────────

pub trait LinalgSVD {
    #[allow(clippy::type_complexity)]
    fn svd(
        &self,
        compute_u: bool,
        compute_vt: bool,
    ) -> Result<(Option<Array2<f64>>, Array1<f64>, Option<Array2<f64>>), GreenersError>;
}

impl LinalgSVD for Array2<f64> {
    fn svd(
        &self,
        compute_u: bool,
        compute_vt: bool,
    ) -> Result<(Option<Array2<f64>>, Array1<f64>, Option<Array2<f64>>), GreenersError> {
        let mat = to_faer(self);
        let svd = Svd::new(mat.as_ref()).map_err(|_| GreenersError::OptimizationFailed)?;
        // S diagonal
        let s_col = svd.S().column_vector();
        let s: Array1<f64> = Array1::from_iter(s_col.iter().copied());
        let u = if compute_u {
            Some(from_faer_ref(svd.U()))
        } else {
            None
        };
        let vt = if compute_vt {
            // faer gives V, not Vt — transpose it
            let v = svd.V();
            let vt_arr = Array2::from_shape_fn((v.ncols(), v.nrows()), |(i, j)| v[(j, i)]);
            Some(vt_arr)
        } else {
            None
        };
        Ok((u, s, vt))
    }
}

// ─── Eigh — replaces `ndarray_linalg::Eigh` ──────────────────────────────────

pub trait LinalgEigh {
    fn eigh(&self, uplo: UPLO) -> Result<(Array1<f64>, Array2<f64>), GreenersError>;
}

impl LinalgEigh for Array2<f64> {
    fn eigh(&self, uplo: UPLO) -> Result<(Array1<f64>, Array2<f64>), GreenersError> {
        let mat = to_faer(self);
        let eig = SelfAdjointEigen::new(mat.as_ref(), uplo.into())
            .map_err(|_| GreenersError::OptimizationFailed)?;
        // S is a Diag<f64> with eigenvalues in ascending order
        let s_col = eig.S().column_vector();
        let evals: Array1<f64> = Array1::from_iter(s_col.iter().copied());
        let evecs = from_faer_ref(eig.U());
        Ok((evals, evecs))
    }
}

// ─── Eig — replaces `ndarray_linalg::Eig` ────────────────────────────────────

pub trait LinalgEig {
    fn eig(&self) -> Result<(Array1<Complex64>, Array2<Complex64>), GreenersError>;
}

impl LinalgEig for Array2<f64> {
    fn eig(&self) -> Result<(Array1<Complex64>, Array2<Complex64>), GreenersError> {
        use faer::linalg::solvers::Eigen;
        let mat = to_faer(self);
        let eig =
            Eigen::new_from_real(mat.as_ref()).map_err(|_| GreenersError::OptimizationFailed)?;
        // S diagonal: Diag<Complex<f64>>
        let s_col = eig.S().column_vector();
        let evals: Array1<Complex64> =
            Array1::from_iter(s_col.iter().map(|c| Complex64::new(c.re, c.im)));
        // U matrix: Mat<Complex<f64>>
        let u = eig.U();
        let evecs: Array2<Complex64> = Array2::from_shape_fn((u.nrows(), u.ncols()), |(i, j)| {
            let c = u[(i, j)];
            Complex64::new(c.re, c.im)
        });
        Ok((evals, evecs))
    }
}

// ─── Cholesky — replaces `ndarray_linalg::Cholesky` ──────────────────────────

pub trait LinalgCholesky {
    fn cholesky(&self, uplo: UPLO) -> Result<Array2<f64>, GreenersError>;
}

impl LinalgCholesky for Array2<f64> {
    fn cholesky(&self, uplo: UPLO) -> Result<Array2<f64>, GreenersError> {
        let mat = to_faer(self);
        let llt = Llt::new(mat.as_ref(), uplo.into()).map_err(|_| GreenersError::SingularMatrix)?;
        let l = llt.L();
        match uplo {
            UPLO::Lower => Ok(from_faer_ref(l)),
            UPLO::Upper => {
                // faer only stores L; transpose for U
                let arr = Array2::from_shape_fn((l.ncols(), l.nrows()), |(i, j)| l[(j, i)]);
                Ok(arr)
            }
        }
    }
}

// ─── Determinant — replaces `ndarray_linalg::Determinant` ────────────────────

pub trait LinalgDeterminant {
    fn det(&self) -> Result<f64, GreenersError>;
}

impl LinalgDeterminant for Array2<f64> {
    fn det(&self) -> Result<f64, GreenersError> {
        if self.nrows() != self.ncols() {
            return Err(GreenersError::InvalidOperation(
                "det: matrix must be square".into(),
            ));
        }
        let mat = to_faer(self);
        Ok(mat.as_ref().determinant())
    }
}
