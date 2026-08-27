//! greeners-core crate.

// Re-export public items that are unique within this crate at the crate root.
// Items with names duplicated across modules remain namespaced.

pub use biplot::{Biplot, BiplotResult, BiplotType};
pub use bootstrap::{Bootstrap, HypothesisTest};
pub use bspline::BSplineBasis;
pub use column::{CategoricalColumn, Column, DataType};
pub use copula::{Copula, CopulaResult, CopulaType};
pub use dataframe::{ColumnType, DataFrame, DataFrameBuilder, TypeInferenceConfig};
pub use datasets::Datasets;
pub use descrstatsw::DescrStatsW;
pub use distributions::{chi2_pvalue, f_pvalue, logistic, norm_pdf, t_pvalue_two, t_quantile};
pub use error::GreenersError;
pub use formula::Formula;
#[cfg(feature = "experimental")]
pub use functional_coef::{FunctionalCoef, FunctionalCoefResult, KernelType};
pub use gmm_clustering::{GmmClustering, GmmResult};
pub use isotonic::{IsotonicRegression, IsotonicResult};
pub use linalg::{
    drop_collinear, CollinearityResult, LinalgCholesky, LinalgDeterminant, LinalgEig, LinalgEigh,
    LinalgInverse, LinalgPinv, LinalgQR, LinalgSVD, UPLO,
};
pub use margins::{MarginalEffectsResult, Margins};
pub use moment_helpers::MomentHelpers;
pub use multipletests::{MultiTestMethod, MultipleTests};
pub use multivariate::{
    CanCorr, CanCorrResult, FactorAnalysis, FactorResult, ManovaResult, PCAResult, Rotation,
    MANOVA, PCA,
};
pub use nonparametric::{
    KDEMultivariate, KDEMultivariateResult, KDEResult, KDEUnivariate, Kernel, KernelReg,
    KernelRegResult, Lowess, LowessResult,
};
pub use predicate::{DsvRow, RowPredicate};
pub use proportion::ProportionTests;
pub use stats::{AnovaRegressionResult, AnovaResult, CompareMeansResult, Stats, TTestResult};
pub use summary_col::{ModelSummary, SummaryCol, SummaryColResult};
pub use transforms::Transforms;
pub use types::{array1_slice, CovarianceType, InferenceType};

pub mod biplot;
pub mod bootstrap;
pub mod bspline;
pub mod column;
pub mod copula;
pub mod dataframe;
pub mod datasets;
pub mod descrstatsw;
pub mod distributions;
pub mod error;
pub mod formula;
#[cfg(feature = "experimental")]
pub mod functional_coef;
pub mod gmm_clustering;
pub mod isotonic;
pub mod linalg;
pub mod margins;
pub mod moment_helpers;
pub mod multipletests;
pub mod multivariate;
pub mod nonparametric;
pub mod predicate;
pub mod proportion;
pub mod stats;
pub mod summary_col;
pub mod transforms;
pub mod types;
