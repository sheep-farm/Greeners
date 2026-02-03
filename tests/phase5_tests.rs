use greeners::*;
use ndarray::Array1;
use std::collections::HashMap;

// ─── Datasets ────────────────────────────────────────────────────────────────

#[test]
fn test_longley_dataset() {
    let df = Datasets::longley().unwrap();
    assert_eq!(df.n_rows(), 16);
    assert!(df.has_column("gnp"));
    assert!(df.has_column("employed"));
}

#[test]
fn test_stackloss_dataset() {
    let df = Datasets::stackloss().unwrap();
    assert_eq!(df.n_rows(), 21);
    assert!(df.has_column("air_flow"));
    assert!(df.has_column("stackloss"));
}

#[test]
fn test_simulated_linear_dataset() {
    let df = Datasets::simulated_linear(100, 42).unwrap();
    assert_eq!(df.n_rows(), 100);
    assert!(df.has_column("y"));
    assert!(df.has_column("x1"));
    assert!(df.has_column("x2"));
}

#[test]
fn test_simulated_panel_dataset() {
    let df = Datasets::simulated_panel(5, 10, 42).unwrap();
    assert_eq!(df.n_rows(), 50);
    assert!(df.has_column("entity"));
    assert!(df.has_column("time"));
}

// ─── DescrStatsW ─────────────────────────────────────────────────────────────

#[test]
fn test_descrstatsw_basic() {
    let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let stats = DescrStatsW::new(&data, None).unwrap();
    assert!((stats.mean - 3.0).abs() < 1e-10);
    assert!((stats.median - 3.0).abs() < 1e-10);
    assert_eq!(stats.min, 1.0);
    assert_eq!(stats.max, 5.0);
    assert_eq!(stats.nobs, 5.0);
    // Display
    let s = format!("{}", stats);
    assert!(s.contains("Mean:"));
}

#[test]
fn test_descrstatsw_weighted() {
    let data = Array1::from(vec![1.0, 2.0, 3.0]);
    let weights = Array1::from(vec![1.0, 2.0, 1.0]);
    let stats = DescrStatsW::new(&data, Some(&weights)).unwrap();
    // Weighted mean: (1*1 + 2*2 + 3*1) / 4 = 8/4 = 2.0
    assert!((stats.mean - 2.0).abs() < 1e-10);
}

#[test]
fn test_descrstatsw_ttest() {
    let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let stats = DescrStatsW::new(&data, None).unwrap();
    let (t, p) = stats.ttest_mean(3.0).unwrap();
    assert!(t.abs() < 1e-10); // mean == mu0
    assert!(p > 0.9);
}

#[test]
fn test_descrstatsw_conf_int() {
    let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let stats = DescrStatsW::new(&data, None).unwrap();
    let (lo, hi) = stats.conf_int_mean(0.05).unwrap();
    assert!(lo < 3.0);
    assert!(hi > 3.0);
}

// ─── Stats ───────────────────────────────────────────────────────────────────

#[test]
fn test_anova_oneway() {
    let data = Array1::from(vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0]);
    let groups = Array1::from(vec![0, 0, 0, 1, 1, 1]);
    let result = Stats::anova_oneway(&data, &groups).unwrap();
    assert_eq!(result.n_groups, 2);
    assert_eq!(result.n_obs, 6);
    assert!(result.f_statistic > 10.0); // groups are clearly different
    assert!(result.p_value < 0.05);
    let s = format!("{}", result);
    assert!(s.contains("ANOVA"));
}

#[test]
fn test_ttest_1samp() {
    let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let (t, p) = Stats::ttest_1samp(&data, 3.0).unwrap();
    assert!(t.abs() < 1e-10);
    assert!(p > 0.9);
}

#[test]
fn test_ttest_ind() {
    let a = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = Array1::from(vec![10.0, 11.0, 12.0, 13.0, 14.0]);
    let (t, p) = Stats::ttest_ind(&a, &b).unwrap();
    assert!(t < -5.0); // a clearly < b
    assert!(p < 0.001);
}

#[test]
fn test_ttest_paired() {
    let a = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = Array1::from(vec![1.5, 2.3, 3.8, 4.2, 5.9]);
    let (t, p) = Stats::ttest_paired(&a, &b).unwrap();
    assert!(t < 0.0); // a < b on average
    assert!(p < 0.1);
}

#[test]
fn test_proportion_ztest() {
    let (z, p) = Stats::proportion_ztest(50, 100, 0.5).unwrap();
    assert!(z.abs() < 1e-10); // exactly 50%
    assert!(p > 0.9);
}

#[test]
fn test_proportion_ztest_2samp() {
    let (z, p) = Stats::proportion_ztest_2samp(80, 100, 20, 100).unwrap();
    assert!(z > 5.0); // very different proportions
    assert!(p < 0.001);
}

#[test]
fn test_bonferroni() {
    let pvals = vec![0.01, 0.04, 0.03];
    let adj = Stats::bonferroni(&pvals);
    assert!((adj[0] - 0.03).abs() < 1e-10);
    assert!((adj[1] - 0.12).abs() < 1e-10);
    assert!((adj[2] - 0.09).abs() < 1e-10);
}

#[test]
fn test_benjamini_hochberg() {
    let pvals = vec![0.01, 0.04, 0.03];
    let adj = Stats::benjamini_hochberg(&pvals);
    // All adjusted should be >= original
    for (i, &p) in pvals.iter().enumerate() {
        assert!(adj[i] >= p - 1e-10);
    }
}

#[test]
fn test_holm() {
    let pvals = vec![0.01, 0.04, 0.03];
    let adj = Stats::holm(&pvals);
    for (i, &p) in pvals.iter().enumerate() {
        assert!(adj[i] >= p - 1e-10);
    }
}

// ─── Influence ───────────────────────────────────────────────────────────────

#[test]
fn test_influence_diagnostics() {
    let df = Datasets::stackloss().unwrap();
    let formula = Formula::parse("stackloss ~ air_flow + water_temp + acid_conc").unwrap();
    let (y, x) = df.to_design_matrix(&formula).unwrap();
    let result = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();

    let mse = result.sigma * result.sigma;
    let infl = Influence::compute(&result.residuals(&y, &x), &x, mse).unwrap();
    assert_eq!(infl.n_obs, 21);
    assert_eq!(infl.n_params, 4); // intercept + 3 vars
    assert_eq!(infl.leverage.len(), 21);
    assert_eq!(infl.dffits.len(), 21);
    assert_eq!(infl.dfbetas.shape(), &[21, 4]);

    let s = format!("{}", infl);
    assert!(s.contains("Influence"));
}

#[test]
fn test_cusum() {
    let df = Datasets::stackloss().unwrap();
    let formula = Formula::parse("stackloss ~ air_flow + water_temp + acid_conc").unwrap();
    let (y, x) = df.to_design_matrix(&formula).unwrap();

    let cusum = CUSUMTest::test(&y, &x).unwrap();
    assert_eq!(cusum.n_obs, 21);
    assert!(cusum.cusum.len() > 0);
    let s = format!("{}", cusum);
    assert!(s.contains("CUSUM"));
}

// ─── SummaryCol ──────────────────────────────────────────────────────────────

#[test]
fn test_summary_col() {
    let m1 = ModelSummary::new("Model 1")
        .with_coefficients(
            &Array1::from(vec![1.0, 2.0]),
            &Array1::from(vec![0.1, 0.2]),
            &Array1::from(vec![0.001, 0.04]),
            &["const".to_string(), "x1".to_string()],
        )
        .with_fit_stats(100, Some(0.95), Some(0.94), None, None, None);

    let m2 = ModelSummary::new("Model 2")
        .with_coefficients(
            &Array1::from(vec![1.1, 2.1, 0.5]),
            &Array1::from(vec![0.1, 0.2, 0.3]),
            &Array1::from(vec![0.001, 0.04, 0.5]),
            &["const".to_string(), "x1".to_string(), "x2".to_string()],
        )
        .with_fit_stats(100, Some(0.96), Some(0.95), None, None, None);

    let result = SummaryCol::compare(&[m1, m2]);
    assert_eq!(result.all_vars.len(), 3); // const, x1, x2

    let display = format!("{}", result);
    assert!(display.contains("Model 1"));
    assert!(display.contains("Model 2"));

    let latex = result.to_latex();
    assert!(latex.contains("\\begin{table}"));

    let html = result.to_html();
    assert!(html.contains("<table"));

    let csv = result.to_csv();
    assert!(csv.contains("Variable"));
}

// ─── Advanced Formula (log, sqrt, poly, bs) ──────────────────────────────────

#[test]
fn test_formula_log_transform() {
    let f = Formula::parse("y ~ log(x1) + x2").unwrap();
    assert_eq!(f.independents, vec!["log(x1)", "x2"]);
}

#[test]
fn test_formula_poly() {
    let f = Formula::parse("y ~ poly(x1, 3) + x2").unwrap();
    assert_eq!(f.independents, vec!["poly(x1, 3)", "x2"]);
}

#[test]
fn test_design_matrix_log() {
    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
    data.insert(
        "x".to_string(),
        Array1::from(vec![
            1.0,
            std::f64::consts::E,
            std::f64::consts::E * std::f64::consts::E,
        ]),
    );
    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ log(x)").unwrap();
    let (_, x_mat) = df.to_design_matrix(&formula).unwrap();
    // intercept + log(x)
    assert_eq!(x_mat.shape(), &[3, 2]);
    assert!((x_mat[[0, 1]] - 0.0).abs() < 1e-10); // log(1) = 0
    assert!((x_mat[[1, 1]] - 1.0).abs() < 1e-10); // log(e) = 1
    assert!((x_mat[[2, 1]] - 2.0).abs() < 1e-10); // log(e^2) = 2
}

#[test]
fn test_design_matrix_sqrt() {
    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
    data.insert("x".to_string(), Array1::from(vec![1.0, 4.0, 9.0]));
    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ sqrt(x)").unwrap();
    let (_, x_mat) = df.to_design_matrix(&formula).unwrap();
    assert!((x_mat[[0, 1]] - 1.0).abs() < 1e-10);
    assert!((x_mat[[1, 1]] - 2.0).abs() < 1e-10);
    assert!((x_mat[[2, 1]] - 3.0).abs() < 1e-10);
}

#[test]
fn test_design_matrix_poly() {
    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
    data.insert("x".to_string(), Array1::from(vec![2.0, 3.0, 4.0]));
    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ poly(x, 2)").unwrap();
    let (_, x_mat) = df.to_design_matrix(&formula).unwrap();
    // intercept + x + x^2 = 3 columns
    assert_eq!(x_mat.shape(), &[3, 3]);
    assert!((x_mat[[0, 1]] - 2.0).abs() < 1e-10); // x^1
    assert!((x_mat[[0, 2]] - 4.0).abs() < 1e-10); // x^2
    assert!((x_mat[[1, 1]] - 3.0).abs() < 1e-10);
    assert!((x_mat[[1, 2]] - 9.0).abs() < 1e-10);
}

#[test]
fn test_design_matrix_bs() {
    let mut data = HashMap::new();
    data.insert("y".to_string(), Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    data.insert("x".to_string(), Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    let df = DataFrame::new(data).unwrap();
    let formula = Formula::parse("y ~ bs(x, 4)").unwrap();
    let (_, x_mat) = df.to_design_matrix(&formula).unwrap();
    // intercept + 4 basis functions = 5 columns
    assert_eq!(x_mat.shape(), &[5, 5]);
}

// ─── ANOVA Regression ────────────────────────────────────────────────────────

#[test]
fn test_anova_regression() {
    let df = Datasets::stackloss().unwrap();
    let formula = Formula::parse("stackloss ~ air_flow + water_temp + acid_conc").unwrap();
    let (y, x) = df.to_design_matrix(&formula).unwrap();
    let result = OLS::fit(&y, &x, CovarianceType::NonRobust).unwrap();
    let anova = Stats::anova_regression(&y, &result.residuals(&y, &x), 3).unwrap();
    assert!(anova.f_statistic > 1.0);
    assert!(anova.ss_model > 0.0);
    let s = format!("{}", anova);
    assert!(s.contains("ANOVA"));
}
