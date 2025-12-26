use crate::{GreenersError, formula::Formula};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

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
        let n_cols = formula.n_cols();
        let mut x_mat = Array2::<f64>::zeros((n_rows, n_cols));

        let mut col_idx = 0;

        // Add intercept if requested
        if formula.intercept {
            for i in 0..n_rows {
                x_mat[[i, col_idx]] = 1.0;
            }
            col_idx += 1;
        }

        // Add independent variables
        for var_name in &formula.independents {
            let col_data = self.get(var_name)?;
            for i in 0..n_rows {
                x_mat[[i, col_idx]] = col_data[i];
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
