use greeners::OLS;
use ndarray::{Array1, Array2, Axis};
use std::error::Error;
use std::fs::File;

fn main() -> Result<(), Box<dyn Error>> {
    // 1. Open the CSV file
    let file = File::open("dataset.csv").expect("File dataset.csv not found in root folder.");
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    // 2. Vectors to hold data temporarily
    // We will read row by row and push to these vectors
    let mut y_vec: Vec<f64> = Vec::new();
    let mut x_flat_vec: Vec<f64> = Vec::new();

    let mut n_rows = 0;

    // 3. Iterate over the CSV records
    for result in rdr.records() {
        let record = result?;

        // Parse columns. Assume struct:
        // 0: income (y), 1: education (x1), 2: age (x2), 3: experience (x3)

        let income: f64 = record[0].parse()?; // y
        let education: f64 = record[1].parse()?; // x1
        let age: f64 = record[2].parse()?; // x2
        let experience: f64 = record[3].parse()?; // x3

        y_vec.push(income);

        // We push X variables into a flat vector.
        // We will reshape it into a Matrix later.
        x_flat_vec.push(education);
        x_flat_vec.push(age);
        x_flat_vec.push(experience);

        n_rows += 1;
    }

    let n_cols_x = 3; // education, age, experience

    // 4. Convert Vectors to Ndarray
    let y = Array1::from(y_vec);

    // Create Array2 from flat vector (Row-major order is default in C/Rust)
    let x_raw = Array2::from_shape_vec((n_rows, n_cols_x), x_flat_vec)?;

    // 5. Add Constant (Intercept)
    let ones = Array2::ones((n_rows, 1));
    let x_with_intercept = ndarray::concatenate(Axis(1), &[ones.view(), x_raw.view()])?;

    // ... (CSV reading code same as before) ...

    // 6. Run Robust OLS (HC1)
    use greeners::CovarianceType; // Import the Enum

    println!("Running OLS with White's Robust Errors (HC1)...");

    // Now we pass the third argument
    let result = OLS::fit(&y, &x_with_intercept, CovarianceType::HC1)?;

    println!("{}", result);

    Ok(())
}
