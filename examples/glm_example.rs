use greeners::{CovarianceType, Family, Link, GLM};
use ndarray::{Array1, Array2};

fn main() {
    println!("=== GLM Examples ===\n");

    // --- 1. Gaussian GLM (equivalent to OLS) ---
    println!("--- Gaussian GLM ---");
    let y = Array1::from(vec![1.0, 2.1, 3.0, 3.9, 5.1, 6.0, 7.2, 7.9, 9.1, 10.0]);
    let x = Array2::from_shape_vec(
        (10, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0,
            9.0, 1.0, 10.0,
        ],
    )
    .unwrap();

    let res = GLM::fit(&y, &x, Family::Gaussian, CovarianceType::NonRobust).unwrap();
    println!("{}", res);

    // --- 2. Poisson GLM for count data ---
    println!("--- Poisson GLM ---");
    let counts = Array1::from(vec![2.0, 3.0, 5.0, 7.0, 11.0, 14.0, 18.0, 22.0, 28.0, 35.0]);
    let res = GLM::fit(&counts, &x, Family::Poisson, CovarianceType::NonRobust).unwrap();
    println!("{}", res);

    // --- 3. Binomial GLM ---
    println!("--- Binomial GLM ---");
    let binary = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    let res = GLM::fit(&binary, &x, Family::Binomial, CovarianceType::NonRobust).unwrap();
    println!("{}", res);

    // --- 4. Gamma GLM ---
    println!("--- Gamma GLM ---");
    let positive = Array1::from(vec![0.5, 1.2, 2.3, 3.1, 4.5, 5.0, 6.8, 7.2, 8.9, 10.1]);
    let res = GLM::fit(&positive, &x, Family::Gamma, CovarianceType::NonRobust).unwrap();
    println!("{}", res);

    // --- 5. Non-canonical link: Poisson + Identity ---
    println!("--- Poisson + Identity Link ---");
    let linear_counts = Array1::from(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
    let res = GLM::fit_with_link(
        &linear_counts,
        &x,
        Family::Poisson,
        Link::Identity,
        CovarianceType::NonRobust,
    )
    .unwrap();
    println!("{}", res);

    // --- 6. Robust standard errors ---
    println!("--- Gaussian GLM with HC1 ---");
    let res = GLM::fit(&y, &x, Family::Gaussian, CovarianceType::HC1).unwrap();
    println!("{}", res);
}
