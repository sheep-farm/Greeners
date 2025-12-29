use crate::{formula::Formula, GreenersError};
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
            return Ok(DataFrame { columns, n_rows: 0 });
        }

        // Check that all columns have the same length
        let first_len = columns.values().next().unwrap().len();
        for (name, col) in &columns {
            if col.len() != first_len {
                return Err(GreenersError::ShapeMismatch(format!(
                    "Column '{}' has length {}, expected {}",
                    name,
                    col.len(),
                    first_len
                )));
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
        self.columns
            .get(name)
            .ok_or_else(|| GreenersError::VariableNotFound(format!("Column '{}' not found", name)))
    }

    /// Get a mutable column by name
    pub fn get_mut(&mut self, name: &str) -> Result<&mut Array1<f64>, GreenersError> {
        self.columns
            .get_mut(name)
            .ok_or_else(|| GreenersError::VariableNotFound(format!("Column '{}' not found", name)))
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
                let var = var_name[2..var_name.len() - 1].trim();
                let col_data = self.get(var)?;

                use std::collections::BTreeSet;
                let unique_vals: BTreeSet<i32> =
                    col_data.iter().map(|&v| v.round() as i32).collect();

                let n_categories = unique_vals.len();
                if n_categories < 2 {
                    return Err(GreenersError::FormulaError(format!(
                        "Categorical variable '{}' must have at least 2 categories",
                        var
                    )));
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
    pub fn to_design_matrix(
        &self,
        formula: &Formula,
    ) -> Result<(Array1<f64>, Array2<f64>), GreenersError> {
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
                let var = var_name[2..var_name.len() - 1].trim();
                let col_data = self.get(var)?;

                // Get unique values (categories)
                use std::collections::BTreeSet;
                let unique_vals: BTreeSet<i32> =
                    col_data.iter().map(|&v| v.round() as i32).collect();

                let mut categories: Vec<i32> = unique_vals.into_iter().collect();
                categories.sort();

                // Create dummies (drop first category for identification)
                // e.g., if categories are [0, 1, 2], create dummies for 1 and 2
                if categories.len() < 2 {
                    return Err(GreenersError::FormulaError(format!(
                        "Categorical variable '{}' must have at least 2 categories",
                        var
                    )));
                }

                // Skip first category (reference level)
                for &cat_val in categories.iter().skip(1) {
                    for i in 0..n_rows {
                        x_mat[[i, col_idx]] = if (col_data[i].round() as i32) == cat_val {
                            1.0
                        } else {
                            0.0
                        };
                    }
                    col_idx += 1;
                }

                continue; // Skip col_idx increment at end since we handled it
            }

            // Check for polynomial term: I(expr)
            if var_name.starts_with("I(") && var_name.ends_with(')') {
                // Extract expression from I(...)
                let expr = var_name[2..var_name.len() - 1].trim();

                // Parse simple expressions: var^power or var**power
                if expr.find('^').or_else(|| expr.find("**")).is_some() {
                    let var_part;
                    let power_part;

                    if expr.contains("**") {
                        let parts: Vec<&str> = expr.split("**").collect();
                        if parts.len() != 2 {
                            return Err(GreenersError::FormulaError(format!(
                                "Invalid polynomial expression '{}'",
                                expr
                            )));
                        }
                        var_part = parts[0].trim();
                        power_part = parts[1].trim();
                    } else {
                        let parts: Vec<&str> = expr.split('^').collect();
                        if parts.len() != 2 {
                            return Err(GreenersError::FormulaError(format!(
                                "Invalid polynomial expression '{}'",
                                expr
                            )));
                        }
                        var_part = parts[0].trim();
                        power_part = parts[1].trim();
                    }

                    let col_data = self.get(var_part)?;
                    let power: i32 = power_part.parse().map_err(|_| {
                        GreenersError::FormulaError(format!(
                            "Invalid power in expression '{}'",
                            expr
                        ))
                    })?;

                    // Compute x^power
                    for i in 0..n_rows {
                        x_mat[[i, col_idx]] = col_data[i].powi(power);
                    }
                } else {
                    return Err(GreenersError::FormulaError(format!(
                        "I() expression must contain ^ or **: '{}'",
                        expr
                    )));
                }
            }
            // Check if this is an interaction term (contains ':')
            else if var_name.contains(':') {
                // Parse interaction: "x1:x2"
                let parts: Vec<&str> = var_name.split(':').collect();
                if parts.len() != 2 {
                    return Err(GreenersError::FormulaError(format!(
                        "Invalid interaction term '{}'",
                        var_name
                    )));
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
            return Err(GreenersError::ShapeMismatch(format!(
                "New column '{}' has length {}, expected {}",
                name,
                data.len(),
                self.n_rows
            )));
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
        let headers = reader
            .headers()
            .map_err(|e| GreenersError::FormulaError(format!("Failed to read headers: {}", e)))?
            .clone();

        // Initialize column vectors
        let mut columns: HashMap<String, Vec<f64>> = HashMap::new();
        for header in headers.iter() {
            columns.insert(header.to_string(), Vec::new());
        }

        // Read all records
        for result in reader.records() {
            let record = result.map_err(|e| {
                GreenersError::FormulaError(format!("Failed to read record: {}", e))
            })?;

            for (i, field) in record.iter().enumerate() {
                let header = &headers[i];
                let value: f64 = field.trim().parse().map_err(|_| {
                    GreenersError::FormulaError(format!(
                        "Failed to parse '{}' as f64 in column '{}'",
                        field, header
                    ))
                })?;

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

    /// Read a DataFrame from a CSV file via URL.
    ///
    /// # Arguments
    /// * `url` - URL to the CSV file
    ///
    /// # Examples
    /// ```no_run
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::from_csv_url("https://example.com/data.csv").unwrap();
    /// println!("Loaded {} rows and {} columns", df.n_rows(), df.n_cols());
    /// ```
    pub fn from_csv_url(url: &str) -> Result<Self, GreenersError> {
        use csv::ReaderBuilder;

        // Fetch CSV content from URL
        let response = reqwest::blocking::get(url)
            .map_err(|e| GreenersError::FormulaError(format!("Failed to fetch URL: {}", e)))?;

        if !response.status().is_success() {
            return Err(GreenersError::FormulaError(format!(
                "HTTP error {}: failed to fetch CSV from URL",
                response.status()
            )));
        }

        let csv_content = response
            .text()
            .map_err(|e| GreenersError::FormulaError(format!("Failed to read response: {}", e)))?;

        // Parse CSV from string
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(csv_content.as_bytes());

        // Get headers
        let headers = reader
            .headers()
            .map_err(|e| GreenersError::FormulaError(format!("Failed to read headers: {}", e)))?
            .clone();

        // Initialize column vectors
        let mut columns: HashMap<String, Vec<f64>> = HashMap::new();
        for header in headers.iter() {
            columns.insert(header.to_string(), Vec::new());
        }

        // Read all records
        for result in reader.records() {
            let record = result.map_err(|e| {
                GreenersError::FormulaError(format!("Failed to read record: {}", e))
            })?;

            for (i, field) in record.iter().enumerate() {
                let header = &headers[i];
                let value: f64 = field.trim().parse().map_err(|_| {
                    GreenersError::FormulaError(format!(
                        "Failed to parse '{}' as f64 in column '{}'",
                        field, header
                    ))
                })?;

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

    /// Read a DataFrame from a JSON file.
    ///
    /// Expected JSON format (records orientation):
    /// ```json
    /// [
    ///   {"x": 1.0, "y": 2.0},
    ///   {"x": 2.0, "y": 4.0},
    ///   {"x": 3.0, "y": 6.0}
    /// ]
    /// ```
    ///
    /// Or column orientation:
    /// ```json
    /// {
    ///   "x": [1.0, 2.0, 3.0],
    ///   "y": [2.0, 4.0, 6.0]
    /// }
    /// ```
    ///
    /// # Examples
    /// ```no_run
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::from_json("data.json").unwrap();
    /// println!("Loaded {} rows and {} columns", df.n_rows(), df.n_cols());
    /// ```
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self, GreenersError> {
        use std::fs::File;
        use std::io::BufReader;

        let path_ref = path.as_ref();

        let file = File::open(path_ref)
            .map_err(|e| GreenersError::FormulaError(format!("Failed to open JSON file: {}", e)))?;

        let reader = BufReader::new(file);

        // Try to parse as column-oriented first
        if let Ok(columns) = serde_json::from_reader::<_, HashMap<String, Vec<f64>>>(reader) {
            let mut data = HashMap::new();
            for (name, values) in columns {
                data.insert(name, Array1::from(values));
            }
            return DataFrame::new(data);
        }

        let file = File::open(path_ref).map_err(|e| {
            GreenersError::FormulaError(format!("Failed to reopen JSON file: {}", e))
        })?;

        let reader = BufReader::new(file);
        let records: Vec<HashMap<String, f64>> = serde_json::from_reader(reader)
            .map_err(|e| GreenersError::FormulaError(format!("Failed to parse JSON: {}", e)))?;

        if records.is_empty() {
            return DataFrame::new(HashMap::new());
        }

        // Get column names from first record
        let first_record = &records[0];
        let mut columns: HashMap<String, Vec<f64>> = HashMap::new();
        for key in first_record.keys() {
            columns.insert(key.clone(), Vec::new());
        }

        // Fill columns
        for record in records {
            for (key, value) in record {
                columns.get_mut(&key).unwrap().push(value);
            }
        }

        // Convert to Array1
        let mut data = HashMap::new();
        for (name, values) in columns {
            data.insert(name, Array1::from(values));
        }

        DataFrame::new(data)
    }

    /// Read a DataFrame from a JSON file via URL.
    ///
    /// Expected JSON format (records orientation):
    /// ```json
    /// [
    ///   {"x": 1.0, "y": 2.0},
    ///   {"x": 2.0, "y": 4.0}
    /// ]
    /// ```
    ///
    /// Or column orientation:
    /// ```json
    /// {
    ///   "x": [1.0, 2.0],
    ///   "y": [2.0, 4.0]
    /// }
    /// ```
    ///
    /// # Examples
    /// ```no_run
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::from_json_url("https://example.com/data.json").unwrap();
    /// println!("Loaded {} rows and {} columns", df.n_rows(), df.n_cols());
    /// ```
    pub fn from_json_url(url: &str) -> Result<Self, GreenersError> {
        // Fetch JSON content from URL
        let response = reqwest::blocking::get(url)
            .map_err(|e| GreenersError::FormulaError(format!("Failed to fetch URL: {}", e)))?;

        if !response.status().is_success() {
            return Err(GreenersError::FormulaError(format!(
                "HTTP error {}: failed to fetch JSON from URL",
                response.status()
            )));
        }

        let json_text = response
            .text()
            .map_err(|e| GreenersError::FormulaError(format!("Failed to read response: {}", e)))?;

        // Try to parse as column-oriented first
        if let Ok(columns) = serde_json::from_str::<HashMap<String, Vec<f64>>>(&json_text) {
            let mut data = HashMap::new();
            for (name, values) in columns {
                data.insert(name, Array1::from(values));
            }
            return DataFrame::new(data);
        }

        // If that fails, try record-oriented
        let records: Vec<HashMap<String, f64>> = serde_json::from_str(&json_text)
            .map_err(|e| GreenersError::FormulaError(format!("Failed to parse JSON: {}", e)))?;

        if records.is_empty() {
            return DataFrame::new(HashMap::new());
        }

        // Get column names from first record
        let first_record = &records[0];
        let mut columns: HashMap<String, Vec<f64>> = HashMap::new();
        for key in first_record.keys() {
            columns.insert(key.clone(), Vec::new());
        }

        // Fill columns
        for record in records {
            for (key, value) in record {
                columns.get_mut(&key).unwrap().push(value);
            }
        }

        // Convert to Array1
        let mut data = HashMap::new();
        for (name, values) in columns {
            data.insert(name, Array1::from(values));
        }

        DataFrame::new(data)
    }

    /// Create a DataFrame from a builder pattern for convenient construction.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0])
    ///     .add_column("y", vec![2.0, 4.0, 6.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(df.n_rows(), 3);
    /// assert_eq!(df.n_cols(), 2);
    /// ```
    pub fn builder() -> DataFrameBuilder {
        DataFrameBuilder::new()
    }

    /// Return the first n rows of the DataFrame.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0, 4.0, 5.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let head = df.head(3).unwrap();
    /// assert_eq!(head.n_rows(), 3);
    /// ```
    pub fn head(&self, n: usize) -> Result<Self, GreenersError> {
        let n = n.min(self.n_rows);
        let mut new_columns = HashMap::new();

        for (name, col) in &self.columns {
            new_columns.insert(name.clone(), col.slice(ndarray::s![..n]).to_owned());
        }

        DataFrame::new(new_columns)
    }

    /// Return the last n rows of the DataFrame.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0, 4.0, 5.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let tail = df.tail(3).unwrap();
    /// assert_eq!(tail.n_rows(), 3);
    /// ```
    pub fn tail(&self, n: usize) -> Result<Self, GreenersError> {
        let n = n.min(self.n_rows);
        let start = self.n_rows.saturating_sub(n);
        let mut new_columns = HashMap::new();

        for (name, col) in &self.columns {
            new_columns.insert(name.clone(), col.slice(ndarray::s![start..]).to_owned());
        }

        DataFrame::new(new_columns)
    }

    /// Calculate the mean of each column.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0])
    ///     .add_column("y", vec![2.0, 4.0, 6.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let means = df.mean();
    /// assert_eq!(means.get("x"), Some(&2.0));
    /// assert_eq!(means.get("y"), Some(&4.0));
    /// ```
    pub fn mean(&self) -> HashMap<String, f64> {
        self.columns
            .iter()
            .map(|(name, col)| {
                let mean = col.sum() / col.len() as f64;
                (name.clone(), mean)
            })
            .collect()
    }

    /// Calculate the sum of each column.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let sums = df.sum();
    /// assert_eq!(sums.get("x"), Some(&6.0));
    /// ```
    pub fn sum(&self) -> HashMap<String, f64> {
        self.columns
            .iter()
            .map(|(name, col)| (name.clone(), col.sum()))
            .collect()
    }

    /// Calculate the minimum value of each column.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let mins = df.min();
    /// assert_eq!(mins.get("x"), Some(&1.0));
    /// ```
    pub fn min(&self) -> HashMap<String, f64> {
        self.columns
            .iter()
            .map(|(name, col)| {
                let min = col
                    .iter()
                    .fold(f64::INFINITY, |a, &b| if b < a { b } else { a });
                (name.clone(), min)
            })
            .collect()
    }

    /// Calculate the maximum value of each column.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let maxs = df.max();
    /// assert_eq!(maxs.get("x"), Some(&3.0));
    /// ```
    pub fn max(&self) -> HashMap<String, f64> {
        self.columns
            .iter()
            .map(|(name, col)| {
                let max = col
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| if b > a { b } else { a });
                (name.clone(), max)
            })
            .collect()
    }

    /// Calculate the standard deviation of each column.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let stds = df.std();
    /// assert!(stds.get("x").unwrap() > 0.0);
    /// ```
    pub fn std(&self) -> HashMap<String, f64> {
        self.columns
            .iter()
            .map(|(name, col)| {
                let mean = col.sum() / col.len() as f64;
                let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / col.len() as f64;
                (name.clone(), variance.sqrt())
            })
            .collect()
    }

    /// Calculate the variance of each column.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let vars = df.var();
    /// assert!(vars.get("x").unwrap() > 0.0);
    /// ```
    pub fn var(&self) -> HashMap<String, f64> {
        self.columns
            .iter()
            .map(|(name, col)| {
                let mean = col.sum() / col.len() as f64;
                let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / col.len() as f64;
                (name.clone(), variance)
            })
            .collect()
    }

    /// Calculate the median of each column.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let medians = df.median();
    /// assert_eq!(medians.get("x"), Some(&2.0));
    /// ```
    pub fn median(&self) -> HashMap<String, f64> {
        self.columns
            .iter()
            .map(|(name, col)| {
                let mut sorted = col.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mid = sorted.len() / 2;
                let median = if sorted.len() % 2 == 0 {
                    (sorted[mid - 1] + sorted[mid]) / 2.0
                } else {
                    sorted[mid]
                };
                (name.clone(), median)
            })
            .collect()
    }

    /// Generate descriptive statistics for each column.
    /// Returns a HashMap where each column name maps to a HashMap of statistics.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0, 4.0, 5.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let stats = df.describe();
    /// let x_stats = stats.get("x").unwrap();
    /// assert_eq!(x_stats.get("count"), Some(&5.0));
    /// assert_eq!(x_stats.get("mean"), Some(&3.0));
    /// ```
    pub fn describe(&self) -> HashMap<String, HashMap<String, f64>> {
        let means = self.mean();
        let stds = self.std();
        let mins = self.min();
        let maxs = self.max();
        let medians = self.median();

        self.columns
            .iter()
            .map(|(name, col)| {
                let mut stats = HashMap::new();
                stats.insert("count".to_string(), col.len() as f64);
                stats.insert("mean".to_string(), means[name]);
                stats.insert("std".to_string(), stds[name]);
                stats.insert("min".to_string(), mins[name]);
                stats.insert("median".to_string(), medians[name]);
                stats.insert("max".to_string(), maxs[name]);
                (name.clone(), stats)
            })
            .collect()
    }

    /// Drop (remove) columns from the DataFrame.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0])
    ///     .add_column("y", vec![3.0, 4.0])
    ///     .add_column("z", vec![5.0, 6.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let df2 = df.drop(&["z"]).unwrap();
    /// assert_eq!(df2.n_cols(), 2);
    /// assert!(!df2.has_column("z"));
    /// ```
    pub fn drop(&self, columns: &[&str]) -> Result<Self, GreenersError> {
        let mut new_columns = self.columns.clone();

        for col_name in columns {
            if !new_columns.contains_key(*col_name) {
                return Err(GreenersError::VariableNotFound(format!(
                    "Column '{}' not found",
                    col_name
                )));
            }
            new_columns.remove(*col_name);
        }

        DataFrame::new(new_columns)
    }

    /// Drop (remove) rows by indices.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0, 4.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let df2 = df.drop_rows(&[1, 3]).unwrap();
    /// assert_eq!(df2.n_rows(), 2);
    /// ```
    pub fn drop_rows(&self, indices: &[usize]) -> Result<Self, GreenersError> {
        // Check bounds
        for &idx in indices {
            if idx >= self.n_rows {
                return Err(GreenersError::ShapeMismatch(format!(
                    "Index {} out of bounds for DataFrame with {} rows",
                    idx, self.n_rows
                )));
            }
        }

        // Create a set of indices to drop for O(1) lookup
        use std::collections::HashSet;
        let drop_set: HashSet<usize> = indices.iter().copied().collect();

        let mut new_columns = HashMap::new();
        for (name, col) in &self.columns {
            let filtered: Vec<f64> = col
                .iter()
                .enumerate()
                .filter(|(i, _)| !drop_set.contains(i))
                .map(|(_, &v)| v)
                .collect();
            new_columns.insert(name.clone(), Array1::from(filtered));
        }

        DataFrame::new(new_columns)
    }

    /// Rename columns in the DataFrame.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    /// use std::collections::HashMap;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0])
    ///     .add_column("y", vec![3.0, 4.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let mut rename_map = HashMap::new();
    /// rename_map.insert("x".to_string(), "new_x".to_string());
    ///
    /// let df2 = df.rename(&rename_map).unwrap();
    /// assert!(df2.has_column("new_x"));
    /// assert!(!df2.has_column("x"));
    /// ```
    pub fn rename(&self, rename_map: &HashMap<String, String>) -> Result<Self, GreenersError> {
        let mut new_columns = HashMap::new();

        for (old_name, col) in &self.columns {
            let new_name = rename_map.get(old_name).unwrap_or(old_name);
            new_columns.insert(new_name.clone(), col.clone());
        }

        DataFrame::new(new_columns)
    }

    /// Sort the DataFrame by a column.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![3.0, 1.0, 2.0])
    ///     .add_column("y", vec![6.0, 4.0, 5.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let sorted = df.sort_by("x", true).unwrap();
    /// let x_col = sorted.get("x").unwrap();
    /// assert_eq!(x_col[0], 1.0);
    /// assert_eq!(x_col[1], 2.0);
    /// assert_eq!(x_col[2], 3.0);
    /// ```
    pub fn sort_by(&self, column: &str, ascending: bool) -> Result<Self, GreenersError> {
        let sort_col = self.get(column)?;

        // Create index vector and sort it based on the column values
        let mut indices: Vec<usize> = (0..self.n_rows).collect();
        indices.sort_by(|&a, &b| {
            let cmp = sort_col[a].partial_cmp(&sort_col[b]).unwrap();
            if ascending {
                cmp
            } else {
                cmp.reverse()
            }
        });

        // Reorder all columns based on sorted indices
        let mut new_columns = HashMap::new();
        for (name, col) in &self.columns {
            let sorted_values: Vec<f64> = indices.iter().map(|&i| col[i]).collect();
            new_columns.insert(name.clone(), Array1::from(sorted_values));
        }

        DataFrame::new(new_columns)
    }

    /// Filter rows based on a predicate function.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0, 4.0, 5.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let filtered = df.filter(|row| {
    ///     row.get("x").map(|&v| v > 2.0).unwrap_or(false)
    /// }).unwrap();
    ///
    /// assert_eq!(filtered.n_rows(), 3);
    /// ```
    pub fn filter<F>(&self, predicate: F) -> Result<Self, GreenersError>
    where
        F: Fn(&HashMap<String, f64>) -> bool,
    {
        let mut keep_indices = Vec::new();

        for i in 0..self.n_rows {
            let mut row = HashMap::new();
            for (name, col) in &self.columns {
                row.insert(name.clone(), col[i]);
            }

            if predicate(&row) {
                keep_indices.push(i);
            }
        }

        let mut new_columns = HashMap::new();
        for (name, col) in &self.columns {
            let filtered: Vec<f64> = keep_indices.iter().map(|&i| col[i]).collect();
            new_columns.insert(name.clone(), Array1::from(filtered));
        }

        DataFrame::new(new_columns)
    }

    /// Write DataFrame to a CSV file.
    ///
    /// # Examples
    /// ```no_run
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0])
    ///     .add_column("y", vec![4.0, 5.0, 6.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// df.to_csv("output.csv").unwrap();
    /// ```
    pub fn to_csv<P: AsRef<Path>>(&self, path: P) -> Result<(), GreenersError> {
        use csv::Writer;
        use std::fs::File;

        let file = File::create(path)
            .map_err(|e| GreenersError::FormulaError(format!("Failed to create CSV file: {}", e)))?;

        let mut writer = Writer::from_writer(file);

        // Write headers
        let mut column_names: Vec<String> = self.columns.keys().cloned().collect();
        column_names.sort(); // Sort for consistent ordering
        writer
            .write_record(&column_names)
            .map_err(|e| GreenersError::FormulaError(format!("Failed to write headers: {}", e)))?;

        // Write data rows
        for i in 0..self.n_rows {
            let row: Vec<String> = column_names
                .iter()
                .map(|name| self.columns[name][i].to_string())
                .collect();
            writer
                .write_record(&row)
                .map_err(|e| GreenersError::FormulaError(format!("Failed to write row: {}", e)))?;
        }

        writer
            .flush()
            .map_err(|e| GreenersError::FormulaError(format!("Failed to flush CSV: {}", e)))?;

        Ok(())
    }

    /// Write DataFrame to a JSON file in column-oriented format.
    ///
    /// # Examples
    /// ```no_run
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0])
    ///     .add_column("y", vec![4.0, 5.0, 6.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// df.to_json("output.json").unwrap();
    /// ```
    pub fn to_json<P: AsRef<Path>>(&self, path: P) -> Result<(), GreenersError> {
        use std::fs::File;
        use std::io::BufWriter;

        let file = File::create(path).map_err(|e| {
            GreenersError::FormulaError(format!("Failed to create JSON file: {}", e))
        })?;

        let writer = BufWriter::new(file);

        // Convert to column-oriented format
        let mut data: HashMap<String, Vec<f64>> = HashMap::new();
        for (name, col) in &self.columns {
            data.insert(name.clone(), col.to_vec());
        }

        serde_json::to_writer_pretty(writer, &data)
            .map_err(|e| GreenersError::FormulaError(format!("Failed to write JSON: {}", e)))?;

        Ok(())
    }

    /// Display information about the DataFrame (similar to pandas.info()).
    ///
    /// Returns a string with:
    /// - Number of rows and columns
    /// - Column names and data types
    /// - Memory usage estimate
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0])
    ///     .add_column("y", vec![4.0, 5.0, 6.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let info = df.info();
    /// assert!(info.contains("2 columns"));
    /// assert!(info.contains("3 rows"));
    /// ```
    pub fn info(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("DataFrame: {} rows, {} columns\n", self.n_rows, self.n_cols()));
        output.push_str("\nColumns:\n");

        let mut column_names: Vec<String> = self.columns.keys().cloned().collect();
        column_names.sort();

        for name in &column_names {
            output.push_str(&format!("  {} (f64)\n", name));
        }

        // Estimate memory usage (rough approximation)
        let memory_bytes = self.n_rows * self.n_cols() * std::mem::size_of::<f64>();
        let memory_kb = memory_bytes as f64 / 1024.0;
        output.push_str(&format!("\nMemory usage: {:.2} KB\n", memory_kb));

        output
    }

    /// Select multiple columns by name, returning a new DataFrame.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0])
    ///     .add_column("y", vec![4.0, 5.0, 6.0])
    ///     .add_column("z", vec![7.0, 8.0, 9.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let selected = df.select(&["x", "z"]).unwrap();
    /// assert_eq!(selected.n_cols(), 2);
    /// assert!(selected.has_column("x"));
    /// assert!(selected.has_column("z"));
    /// assert!(!selected.has_column("y"));
    /// ```
    pub fn select(&self, columns: &[&str]) -> Result<Self, GreenersError> {
        let mut new_columns = HashMap::new();

        for &col_name in columns {
            if !self.columns.contains_key(col_name) {
                return Err(GreenersError::VariableNotFound(format!(
                    "Column '{}' not found",
                    col_name
                )));
            }
            new_columns.insert(col_name.to_string(), self.columns[col_name].clone());
        }

        DataFrame::new(new_columns)
    }

    /// Select rows and columns by index (similar to pandas.iloc).
    ///
    /// # Arguments
    /// * `rows` - Row indices to select (None for all rows)
    /// * `cols` - Column indices to select (None for all columns)
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0, 4.0])
    ///     .add_column("y", vec![5.0, 6.0, 7.0, 8.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// // Select rows 0 and 2, all columns
    /// let subset = df.iloc(Some(&[0, 2]), None).unwrap();
    /// assert_eq!(subset.n_rows(), 2);
    ///
    /// // Select all rows, first column only
    /// let subset2 = df.iloc(None, Some(&[0])).unwrap();
    /// assert_eq!(subset2.n_cols(), 1);
    /// ```
    pub fn iloc(
        &self,
        rows: Option<&[usize]>,
        cols: Option<&[usize]>,
    ) -> Result<Self, GreenersError> {
        // Get sorted column names for consistent indexing
        let mut column_names: Vec<String> = self.columns.keys().cloned().collect();
        column_names.sort();

        // Determine which columns to select
        let selected_col_names: Vec<String> = if let Some(col_indices) = cols {
            col_indices
                .iter()
                .map(|&i| {
                    if i >= column_names.len() {
                        Err(GreenersError::ShapeMismatch(format!(
                            "Column index {} out of bounds (max {})",
                            i,
                            column_names.len() - 1
                        )))
                    } else {
                        Ok(column_names[i].clone())
                    }
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            column_names.clone()
        };

        // Determine which rows to select
        let selected_rows: Vec<usize> = if let Some(row_indices) = rows {
            // Validate row indices
            for &i in row_indices {
                if i >= self.n_rows {
                    return Err(GreenersError::ShapeMismatch(format!(
                        "Row index {} out of bounds (max {})",
                        i,
                        self.n_rows - 1
                    )));
                }
            }
            row_indices.to_vec()
        } else {
            (0..self.n_rows).collect()
        };

        // Build new DataFrame with selected rows and columns
        let mut new_columns = HashMap::new();
        for col_name in selected_col_names {
            let col_data = &self.columns[&col_name];
            let filtered: Vec<f64> = selected_rows.iter().map(|&i| col_data[i]).collect();
            new_columns.insert(col_name, Array1::from(filtered));
        }

        DataFrame::new(new_columns)
    }

    /// Concatenate another DataFrame vertically (row-wise).
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df1 = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0])
    ///     .add_column("y", vec![3.0, 4.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let df2 = DataFrame::builder()
    ///     .add_column("x", vec![5.0, 6.0])
    ///     .add_column("y", vec![7.0, 8.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let combined = df1.concat(&df2).unwrap();
    /// assert_eq!(combined.n_rows(), 4);
    /// ```
    pub fn concat(&self, other: &DataFrame) -> Result<Self, GreenersError> {
        // Check that both DataFrames have the same columns
        if self.n_cols() != other.n_cols() {
            return Err(GreenersError::ShapeMismatch(format!(
                "Cannot concatenate DataFrames with different number of columns ({} vs {})",
                self.n_cols(),
                other.n_cols()
            )));
        }

        for col_name in self.columns.keys() {
            if !other.columns.contains_key(col_name) {
                return Err(GreenersError::VariableNotFound(format!(
                    "Column '{}' not found in second DataFrame",
                    col_name
                )));
            }
        }

        // Concatenate columns
        let mut new_columns = HashMap::new();
        for (col_name, col_data) in &self.columns {
            let other_col_data = &other.columns[col_name];
            let mut combined = col_data.to_vec();
            combined.extend_from_slice(other_col_data.as_slice().unwrap());
            new_columns.insert(col_name.clone(), Array1::from(combined));
        }

        DataFrame::new(new_columns)
    }

    /// Apply a function to each column, returning a new DataFrame.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    /// use ndarray::Array1;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0])
    ///     .add_column("y", vec![4.0, 5.0, 6.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// // Square all values
    /// let squared = df.apply(|col| {
    ///     col.mapv(|v| v * v)
    /// }).unwrap();
    ///
    /// assert_eq!(squared.get("x").unwrap()[0], 1.0);
    /// assert_eq!(squared.get("x").unwrap()[1], 4.0);
    /// ```
    pub fn apply<F>(&self, func: F) -> Result<Self, GreenersError>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let mut new_columns = HashMap::new();

        for (col_name, col_data) in &self.columns {
            let transformed = func(col_data);
            if transformed.len() != self.n_rows {
                return Err(GreenersError::ShapeMismatch(format!(
                    "Applied function changed column length from {} to {}",
                    self.n_rows,
                    transformed.len()
                )));
            }
            new_columns.insert(col_name.clone(), transformed);
        }

        DataFrame::new(new_columns)
    }

    /// Apply a function to transform values in a specific column.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0])
    ///     .add_column("y", vec![4.0, 5.0, 6.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// // Double all values in column 'x'
    /// let modified = df.map_column("x", |v| v * 2.0).unwrap();
    ///
    /// assert_eq!(modified.get("x").unwrap()[0], 2.0);
    /// assert_eq!(modified.get("x").unwrap()[1], 4.0);
    /// assert_eq!(modified.get("y").unwrap()[0], 4.0); // y unchanged
    /// ```
    pub fn map_column<F>(&self, column: &str, func: F) -> Result<Self, GreenersError>
    where
        F: Fn(f64) -> f64,
    {
        if !self.columns.contains_key(column) {
            return Err(GreenersError::VariableNotFound(format!(
                "Column '{}' not found",
                column
            )));
        }

        let mut new_columns = self.columns.clone();
        let col_data = &self.columns[column];
        let transformed = col_data.mapv(&func);
        new_columns.insert(column.to_string(), transformed);

        DataFrame::new(new_columns)
    }

    /// Calculate the correlation matrix between all columns.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0, 4.0, 5.0])
    ///     .add_column("y", vec![2.0, 4.0, 6.0, 8.0, 10.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let corr = df.corr().unwrap();
    /// // x and y are perfectly correlated
    /// assert!((corr[[0, 1]] - 1.0).abs() < 1e-10);
    /// ```
    pub fn corr(&self) -> Result<Array2<f64>, GreenersError> {
        if self.n_cols() == 0 {
            return Err(GreenersError::ShapeMismatch(
                "Cannot compute correlation for empty DataFrame".to_string(),
            ));
        }

        let mut column_names: Vec<String> = self.columns.keys().cloned().collect();
        column_names.sort();

        let n_cols = column_names.len();
        let mut corr_matrix = Array2::<f64>::zeros((n_cols, n_cols));

        // Calculate correlation for each pair
        for i in 0..n_cols {
            for j in 0..n_cols {
                let col_i = &self.columns[&column_names[i]];
                let col_j = &self.columns[&column_names[j]];

                if i == j {
                    corr_matrix[[i, j]] = 1.0;
                } else {
                    // Calculate Pearson correlation
                    let mean_i = col_i.sum() / col_i.len() as f64;
                    let mean_j = col_j.sum() / col_j.len() as f64;

                    let mut num = 0.0;
                    let mut denom_i = 0.0;
                    let mut denom_j = 0.0;

                    for k in 0..col_i.len() {
                        let diff_i = col_i[k] - mean_i;
                        let diff_j = col_j[k] - mean_j;
                        num += diff_i * diff_j;
                        denom_i += diff_i * diff_i;
                        denom_j += diff_j * diff_j;
                    }

                    let corr = if denom_i > 0.0 && denom_j > 0.0 {
                        num / (denom_i.sqrt() * denom_j.sqrt())
                    } else {
                        0.0
                    };

                    corr_matrix[[i, j]] = corr;
                }
            }
        }

        Ok(corr_matrix)
    }

    /// Calculate the covariance matrix between all columns.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0, 4.0, 5.0])
    ///     .add_column("y", vec![2.0, 4.0, 6.0, 8.0, 10.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let cov = df.cov().unwrap();
    /// assert!(cov[[0, 0]] > 0.0); // Variance of x
    /// ```
    pub fn cov(&self) -> Result<Array2<f64>, GreenersError> {
        if self.n_cols() == 0 {
            return Err(GreenersError::ShapeMismatch(
                "Cannot compute covariance for empty DataFrame".to_string(),
            ));
        }

        let mut column_names: Vec<String> = self.columns.keys().cloned().collect();
        column_names.sort();

        let n_cols = column_names.len();
        let mut cov_matrix = Array2::<f64>::zeros((n_cols, n_cols));

        // Calculate covariance for each pair
        for i in 0..n_cols {
            for j in 0..n_cols {
                let col_i = &self.columns[&column_names[i]];
                let col_j = &self.columns[&column_names[j]];

                let mean_i = col_i.sum() / col_i.len() as f64;
                let mean_j = col_j.sum() / col_j.len() as f64;

                let mut covariance = 0.0;
                for k in 0..col_i.len() {
                    covariance += (col_i[k] - mean_i) * (col_j[k] - mean_j);
                }
                covariance /= col_i.len() as f64;

                cov_matrix[[i, j]] = covariance;
            }
        }

        Ok(cov_matrix)
    }

    /// Sample n random rows from the DataFrame.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0, 4.0, 5.0])
    ///     .add_column("y", vec![6.0, 7.0, 8.0, 9.0, 10.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let sample = df.sample(3).unwrap();
    /// assert_eq!(sample.n_rows(), 3);
    /// ```
    pub fn sample(&self, n: usize) -> Result<Self, GreenersError> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        if n > self.n_rows {
            return Err(GreenersError::ShapeMismatch(format!(
                "Cannot sample {} rows from DataFrame with {} rows",
                n, self.n_rows
            )));
        }

        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..self.n_rows).collect();
        indices.shuffle(&mut rng);
        let sample_indices = &indices[..n];

        // Use iloc to select the sampled rows
        self.iloc(Some(sample_indices), None)
    }

    /// Drop rows that contain any NaN (Not a Number) values.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, f64::NAN, 4.0])
    ///     .add_column("y", vec![5.0, f64::NAN, 7.0, 8.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let cleaned = df.dropna().unwrap();
    /// assert_eq!(cleaned.n_rows(), 2); // Only rows 0 and 3 remain
    /// ```
    pub fn dropna(&self) -> Result<Self, GreenersError> {
        let mut keep_indices = Vec::new();

        for i in 0..self.n_rows {
            let mut has_nan = false;
            for col in self.columns.values() {
                if col[i].is_nan() {
                    has_nan = true;
                    break;
                }
            }
            if !has_nan {
                keep_indices.push(i);
            }
        }

        if keep_indices.is_empty() {
            return Ok(DataFrame {
                columns: HashMap::new(),
                n_rows: 0,
            });
        }

        self.iloc(Some(&keep_indices), None)
    }

    /// Fill all NaN values with a specified value.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, f64::NAN, 3.0])
    ///     .add_column("y", vec![4.0, 5.0, f64::NAN])
    ///     .build()
    ///     .unwrap();
    ///
    /// let filled = df.fillna(0.0).unwrap();
    /// assert_eq!(filled.get("x").unwrap()[1], 0.0);
    /// assert_eq!(filled.get("y").unwrap()[2], 0.0);
    /// ```
    pub fn fillna(&self, value: f64) -> Result<Self, GreenersError> {
        let mut new_columns = HashMap::new();

        for (col_name, col_data) in &self.columns {
            let filled = col_data.mapv(|v| if v.is_nan() { value } else { v });
            new_columns.insert(col_name.clone(), filled);
        }

        DataFrame::new(new_columns)
    }

    /// Fill NaN values in a specific column with a specified value.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, f64::NAN, 3.0])
    ///     .add_column("y", vec![4.0, 5.0, f64::NAN])
    ///     .build()
    ///     .unwrap();
    ///
    /// let filled = df.fillna_column("x", 999.0).unwrap();
    /// assert_eq!(filled.get("x").unwrap()[1], 999.0);
    /// assert!(filled.get("y").unwrap()[2].is_nan()); // y unchanged
    /// ```
    pub fn fillna_column(&self, column: &str, value: f64) -> Result<Self, GreenersError> {
        if !self.columns.contains_key(column) {
            return Err(GreenersError::VariableNotFound(format!(
                "Column '{}' not found",
                column
            )));
        }

        let mut new_columns = self.columns.clone();
        let col_data = &self.columns[column];
        let filled = col_data.mapv(|v| if v.is_nan() { value } else { v });
        new_columns.insert(column.to_string(), filled);

        DataFrame::new(new_columns)
    }

    /// Fill NaN values in each column with the mean of that column.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, f64::NAN, 3.0, 4.0])
    ///     .add_column("y", vec![10.0, 20.0, f64::NAN, 40.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let filled = df.fillna_mean().unwrap();
    /// // x: mean of [1, 3, 4] = 2.67
    /// assert!((filled.get("x").unwrap()[1] - 2.666666).abs() < 0.001);
    /// // y: mean of [10, 20, 40] = 23.33
    /// assert!((filled.get("y").unwrap()[2] - 23.333333).abs() < 0.001);
    /// ```
    pub fn fillna_mean(&self) -> Result<Self, GreenersError> {
        let mut new_columns = HashMap::new();

        for (col_name, col_data) in &self.columns {
            // Calculate mean excluding NaN values
            let valid_values: Vec<f64> = col_data.iter().filter(|v| !v.is_nan()).copied().collect();

            if valid_values.is_empty() {
                // If all values are NaN, keep them as NaN
                new_columns.insert(col_name.clone(), col_data.clone());
                continue;
            }

            let mean: f64 = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
            let filled = col_data.mapv(|v| if v.is_nan() { mean } else { v });
            new_columns.insert(col_name.clone(), filled);
        }

        DataFrame::new(new_columns)
    }

    /// Fill NaN values in each column with the median of that column.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, f64::NAN, 3.0, 4.0, 5.0])
    ///     .add_column("y", vec![10.0, 20.0, f64::NAN, 40.0, 50.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let filled = df.fillna_median().unwrap();
    /// // x: median of [1, 3, 4, 5] = 3.5
    /// assert_eq!(filled.get("x").unwrap()[1], 3.5);
    /// // y: median of [10, 20, 40, 50] = 30.0
    /// assert_eq!(filled.get("y").unwrap()[2], 30.0);
    /// ```
    pub fn fillna_median(&self) -> Result<Self, GreenersError> {
        let mut new_columns = HashMap::new();

        for (col_name, col_data) in &self.columns {
            // Calculate median excluding NaN values
            let mut valid_values: Vec<f64> = col_data.iter().filter(|v| !v.is_nan()).copied().collect();

            if valid_values.is_empty() {
                // If all values are NaN, keep them as NaN
                new_columns.insert(col_name.clone(), col_data.clone());
                continue;
            }

            valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = valid_values.len() / 2;
            let median = if valid_values.len() % 2 == 0 {
                (valid_values[mid - 1] + valid_values[mid]) / 2.0
            } else {
                valid_values[mid]
            };

            let filled = col_data.mapv(|v| if v.is_nan() { median } else { v });
            new_columns.insert(col_name.clone(), filled);
        }

        DataFrame::new(new_columns)
    }

    /// Count the number of NaN values in each column.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1.0, f64::NAN, 3.0, f64::NAN])
    ///     .add_column("y", vec![4.0, 5.0, f64::NAN, 7.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let na_counts = df.count_na();
    /// assert_eq!(na_counts.get("x"), Some(&2));
    /// assert_eq!(na_counts.get("y"), Some(&1));
    /// ```
    pub fn count_na(&self) -> HashMap<String, usize> {
        self.columns
            .iter()
            .map(|(name, col)| {
                let count = col.iter().filter(|v| v.is_nan()).count();
                (name.clone(), count)
            })
            .collect()
    }

    /// Check if any value in the DataFrame is NaN.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df1 = DataFrame::builder()
    ///     .add_column("x", vec![1.0, 2.0, 3.0])
    ///     .build()
    ///     .unwrap();
    /// assert!(!df1.has_na());
    ///
    /// let df2 = DataFrame::builder()
    ///     .add_column("x", vec![1.0, f64::NAN, 3.0])
    ///     .build()
    ///     .unwrap();
    /// assert!(df2.has_na());
    /// ```
    pub fn has_na(&self) -> bool {
        self.columns.values().any(|col| col.iter().any(|v| v.is_nan()))
    }
}

impl std::fmt::Display for DataFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.n_rows == 0 {
            return write!(f, "Empty DataFrame");
        }

        // Get sorted column names for consistent display
        let mut column_names: Vec<String> = self.columns.keys().cloned().collect();
        column_names.sort();

        // Calculate column widths
        let mut widths: HashMap<String, usize> = HashMap::new();
        for name in &column_names {
            let col = &self.columns[name];
            let max_value_width = col
                .iter()
                .map(|v| format!("{:.2}", v).len())
                .max()
                .unwrap_or(0);
            widths.insert(name.clone(), name.len().max(max_value_width));
        }

        // Write header
        write!(f, "│ ")?;
        for (i, name) in column_names.iter().enumerate() {
            if i > 0 {
                write!(f, " │ ")?;
            }
            write!(f, "{:>width$}", name, width = widths[name])?;
        }
        writeln!(f, " │")?;

        // Write separator
        write!(f, "├")?;
        for (i, name) in column_names.iter().enumerate() {
            if i > 0 {
                write!(f, "─┼")?;
            }
            write!(f, "─{:─<width$}─", "", width = widths[name])?;
        }
        writeln!(f, "┤")?;

        // Write rows (limit to 10 for display)
        let display_rows = self.n_rows.min(10);
        for row_idx in 0..display_rows {
            write!(f, "│ ")?;
            for (i, name) in column_names.iter().enumerate() {
                if i > 0 {
                    write!(f, " │ ")?;
                }
                let value = self.columns[name][row_idx];
                write!(f, "{:>width$.2}", value, width = widths[name])?;
            }
            writeln!(f, " │")?;
        }

        // Show ellipsis if there are more rows
        if self.n_rows > 10 {
            writeln!(f, "... ({} more rows)", self.n_rows - 10)?;
        }

        Ok(())
    }
}

/// Builder for creating DataFrames conveniently
pub struct DataFrameBuilder {
    columns: HashMap<String, Array1<f64>>,
}

impl DataFrameBuilder {
    /// Create a new DataFrameBuilder
    pub fn new() -> Self {
        DataFrameBuilder {
            columns: HashMap::new(),
        }
    }

    /// Add a column from a Vec<f64>
    pub fn add_column(mut self, name: &str, data: Vec<f64>) -> Self {
        self.columns.insert(name.to_string(), Array1::from(data));
        self
    }

    /// Add a column from an Array1<f64>
    pub fn add_column_array(mut self, name: &str, data: Array1<f64>) -> Self {
        self.columns.insert(name.to_string(), data);
        self
    }

    /// Build the DataFrame
    pub fn build(self) -> Result<DataFrame, GreenersError> {
        DataFrame::new(self.columns)
    }
}

impl Default for DataFrameBuilder {
    fn default() -> Self {
        Self::new()
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
