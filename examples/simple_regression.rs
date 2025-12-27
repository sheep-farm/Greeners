use greeners::{CovarianceType, OLS};
use ndarray::{Array1, Array2};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Simple example data
    let y = Array1::from(vec![1.0, 3.0, 5.0, 7.0, 9.0]);

    // Matrix X with Intercept (column of 1s) and one explanatory variable
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
    )?;

    println!("--- Simple Regression (OLS) ---");

    let result = OLS::fit(&y, &x, CovarianceType::NonRobust)?;

    println!("{}", result);

    Ok(())
}
