use greeners::{CovarianceType, FixedEffects, OLS};
use ndarray::{Array1, Array2, Axis};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Simulating 2 Firms (Entities), 5 Years each.
    // Model: y = alpha_i + 2.0 * x + error

    // Firm 1: Alpha = 10 (High fixed performance)
    // Firm 2: Alpha = -10 (Low fixed performance)

    // X aumenta com o tempo
    let x_vals = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, // Emp 1
        1.0, 2.0, 3.0, 4.0, 5.0, // Emp 2
    ];

    // Y = Alpha + 2*X
    let y_vals = vec![
        12.0, 14.0, 16.0, 18.0, 20.0, // Emp 1 (10 + 2*x)
        -8.0, -6.0, -4.0, -2.0, 0.0, // Emp 2 (-10 + 2*x)
    ];

    let ids = vec![1, 1, 1, 1, 1, 2, 2, 2, 2, 2];

    let y = Array1::from(y_vals);
    let x_col = Array1::from(x_vals);
    // Transform X into Nx1 Matrix
    let x = x_col.view().insert_axis(Axis(1)).to_owned();

    println!("True Beta (Slope): 2.0");
    println!("Level Difference (Fixed Effect): Firm 1 is 20 units higher than Firm 2");

    // 1. Run Pooled OLS (Ignores the firm)
    // Pooled OLS will try to draw a line that passes through the middle of the two point clouds.
    // It will probably find that X doesn't explain much, or find a strange Beta if X were correlated with Alpha.

    // We need to add constant manually for Pooled
    let ones = Array2::ones((10, 1));
    let x_pooled = ndarray::concatenate(Axis(1), &[ones.view(), x.view()])?;

    let pooled_res = OLS::fit(&y, &x_pooled, CovarianceType::NonRobust)?;
    println!("\n--- Pooled OLS (Naive) ---");
    println!("R2: {:.4}", pooled_res.r_squared);
    // R2 will be low because the "Between Firms" variance is huge and not explained by X.

    // 2. Run Fixed Effects
    // We don't pass constant! FE removes the global constant and the alphas.
    let fe_res = FixedEffects::fit(&y, &x, &ids)?;

    println!("\n--- Fixed Effects (Within) ---");
    println!("Estimated Beta: {:.4}", fe_res.params[0]);
    println!("{}", fe_res);
    println!(
        "Note how the R2 ('Within') should be 1.0 or very close, since we cleaned the fixed effect."
    );

    Ok(())
}
