use crate::{GreenersError, formula::Formula};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::path::Path;

/// A simple DataFrame-like structure for storing column-oriented data.
///
/// This provides a minimal interface similar to pandas/polars for working with
/// tabular data and formulas.
#[derive(Debug, Clone)]
pub struct DataFrame {
    /// Column data stored as f64 arrays
    columns: HashMap<String, Array1<f64>>,
    /// Number of rows (all columns must have the same length)
    n_rows: usize,
}

impl DataFrame {
    /// Create a new DataFrame from a HashMap of column names to data arrays.
    ///
    /// # Examples
    /// ```
    /// use greeners::dataframe::DataFrame;
    /// use ndarray::Array1;
    /// use std::collections::HashMap;
    ///
    /// let mut data = HashMap::new();
    /// data.insert("x".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
    /// data.insert("y".to_string(), Array1::from(vec![2.0, 4.0, 6.0]));
    ///
    /// let df = DataFrame::new(data).unwrap();
    /// assert_eq!(df.n_rows(), 3);
    /// ```
    pub fn new(columns: HashMap<String, Array1<f64>>) -> Result<Self, GreenersError> {
        if columns.is_empty() {
            return Ok(DataFrame {
                columns,
                n_rows: 0,
            });
        }

        // Check that all columns have the same length
        let first_len = columns.values().next().unwrap().len();
        for (name, col) in &columns {
            if col.len() != first_len {
                return Err(GreenersError::ShapeMismatch(
                    format!("Column '{}' has length {}, expected {}", name, col.len(), first_len)
                ));
            }
        }

        Ok(DataFrame {
            columns,
            n_rows: first_len,
        })
    }

    /// Get the number of rows in the DataFrame
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Get the number of columns in the DataFrame
    pub fn n_cols(&self) -> usize {
        self.columns.len()
    }

    /// Get a column by name, returning a reference to the Array1
    pub fn get(&self, name: &str) -> Result<&Array1<f64>, GreenersError> {
        self.columns.get(name).ok_or_else(|| {
            GreenersError::VariableNotFound(format!("Column '{}' not found", name))
        })
    }

    /// Get a mutable column by name
    pub fn get_mut(&mut self, name: &str) -> Result<&mut Array1<f64>, GreenersError> {
        self.columns.get_mut(name).ok_or_else(|| {
            GreenersError::VariableNotFound(format!("Column '{}' not found", name))
        })
    }

    /// Check if a column exists
    pub fn has_column(&self, name: &str) -> bool {
        self.columns.contains_key(name)
    }

    /// Get all column names
    pub fn column_names(&self) -> Vec<String> {
        self.columns.keys().cloned().collect()
    }

    /// Count the actual number of columns in the design matrix
    /// (accounting for categorical variable expansion)
    fn count_design_matrix_cols(&self, formula: &Formula) -> Result<usize, GreenersError> {
        let mut count = 0;

        // Intercept
        if formula.intercept {
            count += 1;
        }

        // Independent variables
        for var_name in &formula.independents {
            if var_name.starts_with("C(") && var_name.ends_with(')') {
                // Categorical: count unique values - 1 (drop first)
                let var = var_name[2..var_name.len()-1].trim();
                let col_data = self.get(var)?;

                use std::collections::BTreeSet;
                let unique_vals: BTreeSet<i32> = col_data.iter()
                    .map(|&v| v.round() as i32)
                    .collect();

                let n_categories = unique_vals.len();
                if n_categories < 2 {
                    return Err(GreenersError::FormulaError(
                        format!("Categorical variable '{}' must have at least 2 categories", var)
                    ));
                }

                count += n_categories - 1; // Drop first category
            } else {
                // Regular, interaction, or polynomial: 1 column each
                count += 1;
            }
        }

        Ok(count)
    }

    /// Build design matrix (X) and response vector (y) from a Formula.
    ///
    /// # Examples
    /// ```
    /// use greeners::dataframe::DataFrame;
    /// use greeners::formula::Formula;
    /// use ndarray::Array1;
    /// use std::collections::HashMap;
    ///
    /// let mut data = HashMap::new();
    /// data.insert("y".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
    /// data.insert("x1".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
    /// data.insert("x2".to_string(), Array1::from(vec![2.0, 3.0, 4.0]));
    ///
    /// let df = DataFrame::new(data).unwrap();
    /// let formula = Formula::parse("y ~ x1 + x2").unwrap();
    ///
    /// let (y, x) = df.to_design_matrix(&formula).unwrap();
    /// assert_eq!(y.len(), 3);
    /// assert_eq!(x.shape(), &[3, 3]); // 3 rows, 3 cols (intercept + x1 + x2)
    /// ```
    pub fn to_design_matrix(&self, formula: &Formula) -> Result<(Array1<f64>, Array2<f64>), GreenersError> {
        // Extract y (dependent variable)
        let y = self.get(&formula.dependent)?.clone();

        // Build X matrix
        let n_rows = self.n_rows;

        // Count actual number of columns (accounting for categorical expansion)
        let actual_n_cols = self.count_design_matrix_cols(formula)?;
        let mut x_mat = Array2::<f64>::zeros((n_rows, actual_n_cols));

        let mut col_idx = 0;

        // Add intercept if requested
        if formula.intercept {
            for i in 0..n_rows {
                x_mat[[i, col_idx]] = 1.0;
            }
            col_idx += 1;
        }

        // Add independent variables (including interactions, categorical, polynomial)
        for var_name in &formula.independents {
            // Check for categorical variable: C(var)
            if var_name.starts_with("C(") && var_name.ends_with(')') {
                // Extract variable name from C(var)
                let var = var_name[2..var_name.len()-1].trim();
                let col_data = self.get(var)?;

                // Get unique values (categories)
                use std::collections::BTreeSet;
                let unique_vals: BTreeSet<i32> = col_data.iter()
                    .map(|&v| v.round() as i32)
                    .collect();

                let mut categories: Vec<i32> = unique_vals.into_iter().collect();
                categories.sort();

                // Create dummies (drop first category for identification)
                // e.g., if categories are [0, 1, 2], create dummies for 1 and 2
                if categories.len() < 2 {
                    return Err(GreenersError::FormulaError(
                        format!("Categorical variable '{}' must have at least 2 categories", var)
                    ));
                }

                // Skip first category (reference level)
                for &cat_val in categories.iter().skip(1) {
                    for i in 0..n_rows {
                        x_mat[[i, col_idx]] = if (col_data[i].round() as i32) == cat_val { 1.0 } else { 0.0 };
                    }
                    col_idx += 1;
                }

                continue; // Skip col_idx increment at end since we handled it
            }

            // Check for polynomial term: I(expr)
            if var_name.starts_with("I(") && var_name.ends_with(')') {
                // Extract expression from I(...)
                let expr = var_name[2..var_name.len()-1].trim();

                // Parse simple expressions: var^power or var**power
                if expr.find('^').or_else(|| expr.find("**")).is_some() {
                    let var_part;
                    let power_part;

                    if expr.contains("**") {
                        let parts: Vec<&str> = expr.split("**").collect();
                        if parts.len() != 2 {
                            return Err(GreenersError::FormulaError(
                                format!("Invalid polynomial expression '{}'", expr)
                            ));
                        }
                        var_part = parts[0].trim();
                        power_part = parts[1].trim();
                    } else {
                        let parts: Vec<&str> = expr.split('^').collect();
                        if parts.len() != 2 {
                            return Err(GreenersError::FormulaError(
                                format!("Invalid polynomial expression '{}'", expr)
                            ));
                        }
                        var_part = parts[0].trim();
                        power_part = parts[1].trim();
                    }

                    let col_data = self.get(var_part)?;
                    let power: i32 = power_part.parse().map_err(|_| {
                        GreenersError::FormulaError(
                            format!("Invalid power in expression '{}'", expr)
                        )
                    })?;

                    // Compute x^power
                    for i in 0..n_rows {
                        x_mat[[i, col_idx]] = col_data[i].powi(power);
                    }
                } else {
                    return Err(GreenersError::FormulaError(
                        format!("I() expression must contain ^ or **: '{}'", expr)
                    ));
                }
            }
            // Check if this is an interaction term (contains ':')
            else if var_name.contains(':') {
                // Parse interaction: "x1:x2"
                let parts: Vec<&str> = var_name.split(':').collect();
                if parts.len() != 2 {
                    return Err(GreenersError::FormulaError(
                        format!("Invalid interaction term '{}'", var_name)
                    ));
                }

                let var1 = self.get(parts[0].trim())?;
                let var2 = self.get(parts[1].trim())?;

                // Compute interaction: elementwise multiplication
                for i in 0..n_rows {
                    x_mat[[i, col_idx]] = var1[i] * var2[i];
                }
            } else {
                // Regular variable
                let col_data = self.get(var_name)?;
                for i in 0..n_rows {
                    x_mat[[i, col_idx]] = col_data[i];
                }
            }
            col_idx += 1;
        }

        Ok((y, x_mat))
    }

    /// Insert or update a column
    pub fn insert(&mut self, name: String, data: Array1<f64>) -> Result<(), GreenersError> {
        if !self.columns.is_empty() && data.len() != self.n_rows {
            return Err(GreenersError::ShapeMismatch(
                format!("New column '{}' has length {}, expected {}", name, data.len(), self.n_rows)
            ));
        }

        if self.columns.is_empty() {
            self.n_rows = data.len();
        }

        self.columns.insert(name, data);
        Ok(())
    }

    /// Read a DataFrame from a CSV file with headers.
    ///
    /// # Arguments
    /// * `path` - Path to the CSV file
    ///
    /// # Examples
    /// ```no_run
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::from_csv("data.csv").unwrap();
    /// println!("Loaded {} rows and {} columns", df.n_rows(), df.n_cols());
    /// println!("Columns: {:?}", df.column_names());
    /// ```
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self, GreenersError> {
        use csv::ReaderBuilder;

        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)
            .map_err(|e| GreenersError::FormulaError(format!("Failed to read CSV: {}", e)))?;

        // Get headers
        let headers = reader.headers()
            .map_err(|e| GreenersError::FormulaError(format!("Failed to read headers: {}", e)))?
            .clone();

        // Initialize column vectors
        let mut columns: HashMap<String, Vec<f64>> = HashMap::new();
        for header in headers.iter() {
            columns.insert(header.to_string(), Vec::new());
        }

        // Read all records
        for result in reader.records() {
            let record = result
                .map_err(|e| GreenersError::FormulaError(format!("Failed to read record: {}", e)))?;

            for (i, field) in record.iter().enumerate() {
                let header = &headers[i];
                let value: f64 = field.trim().parse()
                    .map_err(|_| GreenersError::FormulaError(
                        format!("Failed to parse '{}' as f64 in column '{}'", field, header)
                    ))?;

                columns.get_mut(header).unwrap().push(value);
            }
        }

        // Convert Vec<f64> to Array1<f64>
        let mut data = HashMap::new();
        for (name, values) in columns {
            data.insert(name, Array1::from(values));
        }

        DataFrame::new(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataframe_creation() {
        let mut data = HashMap::new();
        data.insert("x".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
        data.insert("y".to_string(), Array1::from(vec![2.0, 4.0, 6.0]));

        let df = DataFrame::new(data).unwrap();
        assert_eq!(df.n_rows(), 3);
        assert_eq!(df.n_cols(), 2);
    }

    #[test]
    fn test_dataframe_get() {
        let mut data = HashMap::new();
        data.insert("x".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));

        let df = DataFrame::new(data).unwrap();
        let col = df.get("x").unwrap();
        assert_eq!(col[0], 1.0);
        assert_eq!(col[1], 2.0);
        assert_eq!(col[2], 3.0);
    }

    #[test]
    fn test_to_design_matrix() {
        let mut data = HashMap::new();
        data.insert("y".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
        data.insert("x1".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
        data.insert("x2".to_string(), Array1::from(vec![2.0, 3.0, 4.0]));

        let df = DataFrame::new(data).unwrap();
        let formula = Formula::parse("y ~ x1 + x2").unwrap();

        let (y, x) = df.to_design_matrix(&formula).unwrap();

        assert_eq!(y.len(), 3);
        assert_eq!(x.shape(), &[3, 3]); // intercept + x1 + x2

        // Check intercept column
        assert_eq!(x[[0, 0]], 1.0);
        assert_eq!(x[[1, 0]], 1.0);
        assert_eq!(x[[2, 0]], 1.0);

        // Check x1 column
        assert_eq!(x[[0, 1]], 1.0);
        assert_eq!(x[[1, 1]], 2.0);
        assert_eq!(x[[2, 1]], 3.0);
    }

    #[test]
    fn test_design_matrix_no_intercept() {
        let mut data = HashMap::new();
        data.insert("y".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
        data.insert("x1".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));

        let df = DataFrame::new(data).unwrap();
        let formula = Formula::parse("y ~ x1 - 1").unwrap();

        let (y, x) = df.to_design_matrix(&formula).unwrap();

        assert_eq!(y.len(), 3);
        assert_eq!(x.shape(), &[3, 1]); // only x1, no intercept
    }
}
