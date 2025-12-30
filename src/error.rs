use thiserror::Error;

/// Custom error types for the Greeners library.
#[derive(Error, Debug)]
pub enum GreenersError {
    /// Error thrown when input dimensions (shapes) do not match expectation.
    #[error("Dimension mismatch: {0}")]
    ShapeMismatch(String),

    /// Error thrown during matrix inversion if the matrix is singular (determinant is zero).
    #[error("Singular matrix encountered. The matrix cannot be inverted.")]
    SingularMatrix,

    /// Error thrown when statistical distributions cannot be initialized
    /// (e.g., degrees of freedom <= 0).
    #[error("Statistical distribution error (optimization/initialization failed)")]
    OptimizationFailed,

    /// Wrapper for errors coming from the ndarray-linalg backend.
    #[error("Linear Algebra backend error: {0}")]
    LinalgError(#[from] ndarray_linalg::error::LinalgError),

    /// Error thrown when parsing a formula string
    #[error("Formula parsing error: {0}")]
    FormulaError(String),

    /// Error thrown when a variable is not found in the data
    #[error("Variable not found in data: {0}")]
    VariableNotFound(String),

    /// Error thrown when an invalid operation is requested
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}
