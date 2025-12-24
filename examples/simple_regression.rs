use greeners::{OLS, CovarianceType};
use ndarray::{Array1, Array2};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Dados simples de exemplo
    let y = Array1::from(vec![1.0, 3.0, 5.0, 7.0, 9.0]);
    
    // Matriz X com Intercepto (coluna de 1s) e uma variável explicativa
    let x = Array2::from_shape_vec((5, 2), vec![
        1.0, 1.0,
        1.0, 2.0,
        1.0, 3.0,
        1.0, 4.0,
        1.0, 5.0,
    ])?;

    println!("--- Simple Regression (OLS) ---");

    let result = OLS::fit(&y, &x, CovarianceType::NonRobust)?;

    println!("{}", result);

    Ok(())
}