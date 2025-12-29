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
    /// assert!(*stds.get("x").unwrap() > 0.0);
    /// ```
    pub fn std(&self) -> HashMap<String, f64> {
        self.columns
            .iter()
            .map(|(name, col)| {
                let mean = col.sum() / col.len() as f64;
                let variance =
                    col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / col.len() as f64;
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
    /// assert!(*vars.get("x").unwrap() > 0.0);
    /// ```
    pub fn var(&self) -> HashMap<String, f64> {
        self.columns
            .iter()
            .map(|(name, col)| {
                let mean = col.sum() / col.len() as f64;
                let variance =
                    col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / col.len() as f64;
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

        let file = File::create(path).map_err(|e| {
            GreenersError::FormulaError(format!("Failed to create CSV file: {}", e))
        })?;

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
        output.push_str(&format!(
            "DataFrame: {} rows, {} columns\n",
            self.n_rows,
            self.n_cols()
        ));
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
            let mut valid_values: Vec<f64> =
                col_data.iter().filter(|v| !v.is_nan()).copied().collect();

            if valid_values.is_empty() {
                // If all values are NaN, keep them as NaN
                new_columns.insert(col_name.clone(), col_data.clone());
                continue;
            }

            valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = valid_values.len() / 2;
            let median = if valid_values.len().is_multiple_of(2) {
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
        self.columns
            .values()
            .any(|col| col.iter().any(|v| v.is_nan()))
    }

    /// Append a single row to the DataFrame.
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
    /// let mut row = HashMap::new();
    /// row.insert("x".to_string(), 5.0);
    /// row.insert("y".to_string(), 6.0);
    ///
    /// let df2 = df.append_row(&row).unwrap();
    /// assert_eq!(df2.n_rows(), 3);
    /// ```
    pub fn append_row(&self, row: &HashMap<String, f64>) -> Result<Self, GreenersError> {
        // Check that row has all required columns
        for col_name in self.columns.keys() {
            if !row.contains_key(col_name) {
                return Err(GreenersError::VariableNotFound(format!(
                    "Row is missing column '{}'",
                    col_name
                )));
            }
        }

        // Check for extra columns in row
        for col_name in row.keys() {
            if !self.columns.contains_key(col_name) {
                return Err(GreenersError::VariableNotFound(format!(
                    "Unknown column '{}' in row",
                    col_name
                )));
            }
        }

        let mut new_columns = HashMap::new();
        for (col_name, col_data) in &self.columns {
            let mut new_col = col_data.to_vec();
            new_col.push(row[col_name]);
            new_columns.insert(col_name.clone(), Array1::from(new_col));
        }

        DataFrame::new(new_columns)
    }

    /// Merge (join) two DataFrames on a common column.
    ///
    /// # Arguments
    /// * `other` - The DataFrame to merge with
    /// * `on` - The column name to join on
    /// * `how` - Join type: "inner", "left", "right", or "outer"
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df1 = DataFrame::builder()
    ///     .add_column("id", vec![1.0, 2.0, 3.0])
    ///     .add_column("value_a", vec![10.0, 20.0, 30.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let df2 = DataFrame::builder()
    ///     .add_column("id", vec![2.0, 3.0, 4.0])
    ///     .add_column("value_b", vec![200.0, 300.0, 400.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// // Inner join
    /// let merged = df1.merge(&df2, "id", "inner").unwrap();
    /// assert_eq!(merged.n_rows(), 2); // Only rows with id 2 and 3
    /// ```
    pub fn merge(&self, other: &DataFrame, on: &str, how: &str) -> Result<Self, GreenersError> {
        // Validate join column exists in both DataFrames
        if !self.has_column(on) {
            return Err(GreenersError::VariableNotFound(format!(
                "Join column '{}' not found in left DataFrame",
                on
            )));
        }
        if !other.has_column(on) {
            return Err(GreenersError::VariableNotFound(format!(
                "Join column '{}' not found in right DataFrame",
                on
            )));
        }

        let left_key = self.get(on)?;
        let right_key = other.get(on)?;

        // Build result columns
        let mut result_data: HashMap<String, Vec<f64>> = HashMap::new();

        // Initialize all columns
        for col_name in self.columns.keys() {
            result_data.insert(col_name.clone(), Vec::new());
        }
        for col_name in other.columns.keys() {
            if col_name != on {
                result_data.insert(col_name.clone(), Vec::new());
            }
        }

        match how {
            "inner" => {
                // Inner join: only rows with matching keys
                for i in 0..self.n_rows {
                    let left_val = left_key[i];
                    for j in 0..other.n_rows {
                        if (left_val - right_key[j]).abs() < 1e-10 {
                            // Match found
                            for (col_name, col_data) in &self.columns {
                                result_data.get_mut(col_name).unwrap().push(col_data[i]);
                            }
                            for (col_name, col_data) in &other.columns {
                                if col_name != on {
                                    result_data.get_mut(col_name).unwrap().push(col_data[j]);
                                }
                            }
                        }
                    }
                }
            }
            "left" => {
                // Left join: all rows from left, matched from right
                for i in 0..self.n_rows {
                    let left_val = left_key[i];
                    let mut found = false;

                    for j in 0..other.n_rows {
                        if (left_val - right_key[j]).abs() < 1e-10 {
                            found = true;
                            for (col_name, col_data) in &self.columns {
                                result_data.get_mut(col_name).unwrap().push(col_data[i]);
                            }
                            for (col_name, col_data) in &other.columns {
                                if col_name != on {
                                    result_data.get_mut(col_name).unwrap().push(col_data[j]);
                                }
                            }
                        }
                    }

                    if !found {
                        // No match: include left row with NaN for right columns
                        for (col_name, col_data) in &self.columns {
                            result_data.get_mut(col_name).unwrap().push(col_data[i]);
                        }
                        for col_name in other.columns.keys() {
                            if col_name != on {
                                result_data.get_mut(col_name).unwrap().push(f64::NAN);
                            }
                        }
                    }
                }
            }
            "right" => {
                // Right join: all rows from right, matched from left
                for j in 0..other.n_rows {
                    let right_val = right_key[j];
                    let mut found = false;

                    for i in 0..self.n_rows {
                        if (left_key[i] - right_val).abs() < 1e-10 {
                            found = true;
                            for (col_name, col_data) in &self.columns {
                                result_data.get_mut(col_name).unwrap().push(col_data[i]);
                            }
                            for (col_name, col_data) in &other.columns {
                                if col_name != on {
                                    result_data.get_mut(col_name).unwrap().push(col_data[j]);
                                }
                            }
                        }
                    }

                    if !found {
                        // No match: include right row with NaN for left columns
                        for col_name in self.columns.keys() {
                            if col_name != on {
                                result_data.get_mut(col_name).unwrap().push(f64::NAN);
                            } else {
                                result_data.get_mut(col_name).unwrap().push(right_val);
                            }
                        }
                        for (col_name, col_data) in &other.columns {
                            if col_name != on {
                                result_data.get_mut(col_name).unwrap().push(col_data[j]);
                            }
                        }
                    }
                }
            }
            "outer" => {
                // Outer join: all rows from both, with NaN where no match
                use std::collections::HashSet;
                let mut processed_right: HashSet<usize> = HashSet::new();

                // Process left rows
                for i in 0..self.n_rows {
                    let left_val = left_key[i];
                    let mut found = false;

                    for j in 0..other.n_rows {
                        if (left_val - right_key[j]).abs() < 1e-10 {
                            found = true;
                            processed_right.insert(j);

                            for (col_name, col_data) in &self.columns {
                                result_data.get_mut(col_name).unwrap().push(col_data[i]);
                            }
                            for (col_name, col_data) in &other.columns {
                                if col_name != on {
                                    result_data.get_mut(col_name).unwrap().push(col_data[j]);
                                }
                            }
                        }
                    }

                    if !found {
                        for (col_name, col_data) in &self.columns {
                            result_data.get_mut(col_name).unwrap().push(col_data[i]);
                        }
                        for col_name in other.columns.keys() {
                            if col_name != on {
                                result_data.get_mut(col_name).unwrap().push(f64::NAN);
                            }
                        }
                    }
                }

                // Add unmatched right rows
                for j in 0..other.n_rows {
                    if !processed_right.contains(&j) {
                        for col_name in self.columns.keys() {
                            if col_name != on {
                                result_data.get_mut(col_name).unwrap().push(f64::NAN);
                            } else {
                                result_data.get_mut(col_name).unwrap().push(right_key[j]);
                            }
                        }
                        for (col_name, col_data) in &other.columns {
                            if col_name != on {
                                result_data.get_mut(col_name).unwrap().push(col_data[j]);
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(GreenersError::FormulaError(format!(
                    "Unknown join type '{}'. Use 'inner', 'left', 'right', or 'outer'",
                    how
                )));
            }
        }

        // Convert to Array1
        let mut final_columns = HashMap::new();
        for (col_name, values) in result_data {
            final_columns.insert(col_name, Array1::from(values));
        }

        DataFrame::new(final_columns)
    }

    /// Group by one or more columns and apply aggregation functions.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("category", vec![1.0, 1.0, 2.0, 2.0, 3.0])
    ///     .add_column("value", vec![10.0, 20.0, 30.0, 40.0, 50.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// // Group by category and sum values
    /// let grouped = df.groupby(&["category"], "value", "sum").unwrap();
    /// assert_eq!(grouped.n_rows(), 3); // 3 unique categories
    /// ```
    pub fn groupby(&self, by: &[&str], value_col: &str, agg: &str) -> Result<Self, GreenersError> {
        // Validate columns exist
        for col in by {
            if !self.has_column(col) {
                return Err(GreenersError::VariableNotFound(format!(
                    "Grouping column '{}' not found",
                    col
                )));
            }
        }
        if !self.has_column(value_col) {
            return Err(GreenersError::VariableNotFound(format!(
                "Value column '{}' not found",
                value_col
            )));
        }

        // Build groups
        use std::collections::BTreeMap;
        let mut groups: BTreeMap<Vec<i64>, Vec<usize>> = BTreeMap::new();

        for i in 0..self.n_rows {
            let mut key = Vec::new();
            for col_name in by {
                let val = self.get(col_name)?[i];
                key.push(val.round() as i64);
            }
            groups.entry(key).or_default().push(i);
        }

        // Apply aggregation
        let value_data = self.get(value_col)?;
        let mut result_keys: Vec<Vec<f64>> = vec![Vec::new(); by.len()];
        let mut result_values: Vec<f64> = Vec::new();

        for (key, indices) in groups {
            // Add group keys
            for (i, &k) in key.iter().enumerate() {
                result_keys[i].push(k as f64);
            }

            // Calculate aggregation
            let group_values: Vec<f64> = indices.iter().map(|&i| value_data[i]).collect();
            let agg_value = match agg {
                "sum" => group_values.iter().sum(),
                "mean" => group_values.iter().sum::<f64>() / group_values.len() as f64,
                "count" => group_values.len() as f64,
                "min" => group_values
                    .iter()
                    .fold(f64::INFINITY, |a, &b| if b < a { b } else { a }),
                "max" => group_values
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| if b > a { b } else { a }),
                "median" => {
                    let mut sorted = group_values.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mid = sorted.len() / 2;
                    if sorted.len().is_multiple_of(2) {
                        (sorted[mid - 1] + sorted[mid]) / 2.0
                    } else {
                        sorted[mid]
                    }
                }
                _ => {
                    return Err(GreenersError::FormulaError(format!(
                        "Unknown aggregation '{}'. Use 'sum', 'mean', 'count', 'min', 'max', or 'median'",
                        agg
                    )));
                }
            };
            result_values.push(agg_value);
        }

        // Build result DataFrame
        let mut result_columns = HashMap::new();
        for (i, col_name) in by.iter().enumerate() {
            result_columns.insert(col_name.to_string(), Array1::from(result_keys[i].clone()));
        }
        result_columns.insert(
            format!("{}_{}", value_col, agg),
            Array1::from(result_values),
        );

        DataFrame::new(result_columns)
    }

    /// Create a pivot table - reshape data from long to wide format.
    ///
    /// # Arguments
    /// * `index` - Column to use as row index
    /// * `columns` - Column to use as column headers
    /// * `values` - Column to aggregate
    /// * `aggfunc` - Aggregation function: "sum", "mean", "count", "min", "max", "median"
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// // Sales data in long format
    /// let df = DataFrame::builder()
    ///     .add_column("product", vec![1.0, 1.0, 2.0, 2.0])
    ///     .add_column("region", vec![1.0, 2.0, 1.0, 2.0])
    ///     .add_column("sales", vec![100.0, 150.0, 200.0, 250.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// // Pivot: products as rows, regions as columns
    /// let pivoted = df.pivot_table("product", "region", "sales", "sum").unwrap();
    /// // Result: 2 rows (products), 2 columns (regions) + index column
    /// ```
    pub fn pivot_table(
        &self,
        index: &str,
        columns: &str,
        values: &str,
        aggfunc: &str,
    ) -> Result<Self, GreenersError> {
        // Validate columns exist
        if !self.has_column(index) {
            return Err(GreenersError::VariableNotFound(format!(
                "Index column '{}' not found",
                index
            )));
        }
        if !self.has_column(columns) {
            return Err(GreenersError::VariableNotFound(format!(
                "Columns column '{}' not found",
                columns
            )));
        }
        if !self.has_column(values) {
            return Err(GreenersError::VariableNotFound(format!(
                "Values column '{}' not found",
                values
            )));
        }

        // Get unique values for index and columns
        use std::collections::BTreeSet;
        let index_data = self.get(index)?;
        let columns_data = self.get(columns)?;
        let values_data = self.get(values)?;

        let unique_indices: BTreeSet<i64> = index_data.iter().map(|&v| v.round() as i64).collect();
        let unique_columns: BTreeSet<i64> =
            columns_data.iter().map(|&v| v.round() as i64).collect();

        // Build pivot table structure
        use std::collections::HashMap as StdHashMap;

        // Map (index_val, column_val) -> list of values
        let mut pivot_data: StdHashMap<(i64, i64), Vec<f64>> = StdHashMap::new();

        for i in 0..self.n_rows {
            let idx = index_data[i].round() as i64;
            let col = columns_data[i].round() as i64;
            let val = values_data[i];

            pivot_data.entry((idx, col)).or_default().push(val);
        }

        // Apply aggregation function
        let agg_fn = |vals: &[f64]| -> f64 {
            match aggfunc {
                "sum" => vals.iter().sum(),
                "mean" => vals.iter().sum::<f64>() / vals.len() as f64,
                "count" => vals.len() as f64,
                "min" => vals.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                "max" => vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                "median" => {
                    let mut sorted = vals.to_vec();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mid = sorted.len() / 2;
                    if sorted.len().is_multiple_of(2) {
                        (sorted[mid - 1] + sorted[mid]) / 2.0
                    } else {
                        sorted[mid]
                    }
                }
                _ => f64::NAN,
            }
        };

        // Build result DataFrame
        let mut result_columns = HashMap::new();

        // Add index column
        let index_vec: Vec<f64> = unique_indices.iter().map(|&v| v as f64).collect();
        result_columns.insert(index.to_string(), Array1::from(index_vec.clone()));

        // Add a column for each unique column value
        for &col_val in &unique_columns {
            let mut col_data = Vec::new();

            for &idx_val in &unique_indices {
                let key = (idx_val, col_val);
                let agg_value = if let Some(vals) = pivot_data.get(&key) {
                    agg_fn(vals)
                } else {
                    f64::NAN // No data for this combination
                };
                col_data.push(agg_value);
            }

            // Column name: original_column_name + "_" + column_value
            let col_name = format!("{}_{}", columns, col_val);
            result_columns.insert(col_name, Array1::from(col_data));
        }

        DataFrame::new(result_columns)
    }

    /// Apply a rolling window function to a column.
    ///
    /// # Arguments
    /// * `column` - Column to apply rolling window to
    /// * `window` - Window size
    /// * `func` - Function: "mean", "sum", "min", "max", "std"
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("value", vec![1.0, 2.0, 3.0, 4.0, 5.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// // 3-period rolling mean
    /// let rolled = df.rolling("value", 3, "mean").unwrap();
    /// // First 2 values will be NaN (not enough data)
    /// ```
    pub fn rolling(&self, column: &str, window: usize, func: &str) -> Result<Self, GreenersError> {
        if !self.has_column(column) {
            return Err(GreenersError::VariableNotFound(format!(
                "Column '{}' not found",
                column
            )));
        }

        if window == 0 {
            return Err(GreenersError::FormulaError(
                "Window size must be > 0".to_string(),
            ));
        }

        let col_data = self.get(column)?;
        let mut result = vec![f64::NAN; col_data.len()];

        #[allow(clippy::needless_range_loop)]
        for i in 0..col_data.len() {
            if i + 1 < window {
                // Not enough data yet
                result[i] = f64::NAN;
            } else {
                let window_data: Vec<f64> =
                    col_data.slice(ndarray::s![i + 1 - window..=i]).to_vec();

                result[i] = match func {
                    "mean" => window_data.iter().sum::<f64>() / window_data.len() as f64,
                    "sum" => window_data.iter().sum(),
                    "min" => window_data.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                    "max" => window_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                    "std" => {
                        let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
                        let variance = window_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                            / window_data.len() as f64;
                        variance.sqrt()
                    }
                    _ => {
                        return Err(GreenersError::FormulaError(format!(
                            "Unknown function '{}'. Use 'mean', 'sum', 'min', 'max', or 'std'",
                            func
                        )));
                    }
                };
            }
        }

        let mut new_df = self.clone();
        new_df.insert(format!("{}_rolling_{}", column, func), Array1::from(result))?;
        Ok(new_df)
    }

    /// Calculate cumulative sum for a column.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("value", vec![1.0, 2.0, 3.0, 4.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let cumsum = df.cumsum("value").unwrap();
    /// // Result: [1.0, 3.0, 6.0, 10.0]
    /// ```
    pub fn cumsum(&self, column: &str) -> Result<Self, GreenersError> {
        if !self.has_column(column) {
            return Err(GreenersError::VariableNotFound(format!(
                "Column '{}' not found",
                column
            )));
        }

        let col_data = self.get(column)?;
        let mut cumsum = Vec::with_capacity(col_data.len());
        let mut sum = 0.0;

        for &val in col_data.iter() {
            sum += val;
            cumsum.push(sum);
        }

        let mut new_df = self.clone();
        new_df.insert(format!("{}_cumsum", column), Array1::from(cumsum))?;
        Ok(new_df)
    }

    /// Calculate cumulative product for a column.
    pub fn cumprod(&self, column: &str) -> Result<Self, GreenersError> {
        if !self.has_column(column) {
            return Err(GreenersError::VariableNotFound(format!(
                "Column '{}' not found",
                column
            )));
        }

        let col_data = self.get(column)?;
        let mut cumprod = Vec::with_capacity(col_data.len());
        let mut prod = 1.0;

        for &val in col_data.iter() {
            prod *= val;
            cumprod.push(prod);
        }

        let mut new_df = self.clone();
        new_df.insert(format!("{}_cumprod", column), Array1::from(cumprod))?;
        Ok(new_df)
    }

    /// Calculate cumulative maximum for a column.
    pub fn cummax(&self, column: &str) -> Result<Self, GreenersError> {
        if !self.has_column(column) {
            return Err(GreenersError::VariableNotFound(format!(
                "Column '{}' not found",
                column
            )));
        }

        let col_data = self.get(column)?;
        let mut cummax = Vec::with_capacity(col_data.len());
        let mut max_val = f64::NEG_INFINITY;

        for &val in col_data.iter() {
            max_val = max_val.max(val);
            cummax.push(max_val);
        }

        let mut new_df = self.clone();
        new_df.insert(format!("{}_cummax", column), Array1::from(cummax))?;
        Ok(new_df)
    }

    /// Calculate cumulative minimum for a column.
    pub fn cummin(&self, column: &str) -> Result<Self, GreenersError> {
        if !self.has_column(column) {
            return Err(GreenersError::VariableNotFound(format!(
                "Column '{}' not found",
                column
            )));
        }

        let col_data = self.get(column)?;
        let mut cummin = Vec::with_capacity(col_data.len());
        let mut min_val = f64::INFINITY;

        for &val in col_data.iter() {
            min_val = min_val.min(val);
            cummin.push(min_val);
        }

        let mut new_df = self.clone();
        new_df.insert(format!("{}_cummin", column), Array1::from(cummin))?;
        Ok(new_df)
    }

    /// Shift column values by a number of periods.
    ///
    /// # Arguments
    /// * `column` - Column to shift
    /// * `periods` - Number of periods to shift (positive = down, negative = up)
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("value", vec![1.0, 2.0, 3.0, 4.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let shifted = df.shift("value", 1).unwrap();
    /// // Result: [NaN, 1.0, 2.0, 3.0]
    /// ```
    pub fn shift(&self, column: &str, periods: i32) -> Result<Self, GreenersError> {
        if !self.has_column(column) {
            return Err(GreenersError::VariableNotFound(format!(
                "Column '{}' not found",
                column
            )));
        }

        let col_data = self.get(column)?;
        let n = col_data.len();
        let mut shifted = vec![f64::NAN; n];

        if periods > 0 {
            // Shift down
            let p = periods as usize;
            for i in p..n {
                shifted[i] = col_data[i - p];
            }
        } else if periods < 0 {
            // Shift up
            let p = (-periods) as usize;
            for i in 0..(n.saturating_sub(p)) {
                shifted[i] = col_data[i + p];
            }
        } else {
            // No shift
            shifted = col_data.to_vec();
        }

        let mut new_df = self.clone();
        new_df.insert(
            format!("{}_shift_{}", column, periods),
            Array1::from(shifted),
        )?;
        Ok(new_df)
    }

    /// Calculate quantile (percentile) for a column.
    ///
    /// # Arguments
    /// * `column` - Column to calculate quantile for
    /// * `q` - Quantile to compute (0.0 to 1.0, e.g., 0.5 = median)
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("value", vec![1.0, 2.0, 3.0, 4.0, 5.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let q75 = df.quantile("value", 0.75).unwrap();
    /// // 75th percentile
    /// ```
    pub fn quantile(&self, column: &str, q: f64) -> Result<f64, GreenersError> {
        if !self.has_column(column) {
            return Err(GreenersError::VariableNotFound(format!(
                "Column '{}' not found",
                column
            )));
        }

        if !(0.0..=1.0).contains(&q) {
            return Err(GreenersError::FormulaError(
                "Quantile must be between 0 and 1".to_string(),
            ));
        }

        let col_data = self.get(column)?;
        let mut sorted: Vec<f64> = col_data.iter().filter(|v| !v.is_nan()).copied().collect();

        if sorted.is_empty() {
            return Ok(f64::NAN);
        }

        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = q * (sorted.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            Ok(sorted[lower])
        } else {
            let weight = index - lower as f64;
            Ok(sorted[lower] * (1.0 - weight) + sorted[upper] * weight)
        }
    }

    /// Rank values in a column.
    ///
    /// # Arguments
    /// * `column` - Column to rank
    /// * `ascending` - If true, smallest value gets rank 1
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("value", vec![30.0, 10.0, 20.0, 40.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let ranked = df.rank("value", true).unwrap();
    /// // Result: [3.0, 1.0, 2.0, 4.0]
    /// ```
    pub fn rank(&self, column: &str, ascending: bool) -> Result<Self, GreenersError> {
        if !self.has_column(column) {
            return Err(GreenersError::VariableNotFound(format!(
                "Column '{}' not found",
                column
            )));
        }

        let col_data = self.get(column)?;
        let mut indexed: Vec<(usize, f64)> = col_data.iter().copied().enumerate().collect();

        if ascending {
            indexed.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        } else {
            indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        }

        let mut ranks = vec![0.0; col_data.len()];
        for (rank, (original_idx, _)) in indexed.iter().enumerate() {
            ranks[*original_idx] = (rank + 1) as f64;
        }

        let mut new_df = self.clone();
        new_df.insert(format!("{}_rank", column), Array1::from(ranks))?;
        Ok(new_df)
    }

    /// Drop duplicate rows based on a column.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("id", vec![1.0, 2.0, 1.0, 3.0])
    ///     .add_column("value", vec![10.0, 20.0, 30.0, 40.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let unique = df.drop_duplicates("id").unwrap();
    /// assert_eq!(unique.n_rows(), 3); // Only first occurrence kept
    /// ```
    pub fn drop_duplicates(&self, column: &str) -> Result<Self, GreenersError> {
        if !self.has_column(column) {
            return Err(GreenersError::VariableNotFound(format!(
                "Column '{}' not found",
                column
            )));
        }

        let col_data = self.get(column)?;
        let mut seen = std::collections::HashSet::new();
        let mut keep_indices = Vec::new();

        for (i, &val) in col_data.iter().enumerate() {
            let key = val.to_bits(); // Use bit pattern as key (handles NaN)
            if seen.insert(key) {
                keep_indices.push(i);
            }
        }

        self.iloc(Some(&keep_indices), None)
    }

    /// Interpolate NaN values linearly in a column.
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// let df = DataFrame::builder()
    ///     .add_column("value", vec![1.0, f64::NAN, f64::NAN, 4.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let interpolated = df.interpolate("value").unwrap();
    /// // Result: [1.0, 2.0, 3.0, 4.0]
    /// ```
    pub fn interpolate(&self, column: &str) -> Result<Self, GreenersError> {
        if !self.has_column(column) {
            return Err(GreenersError::VariableNotFound(format!(
                "Column '{}' not found",
                column
            )));
        }

        let col_data = self.get(column)?;
        let mut result = col_data.to_vec();

        // Find first non-NaN
        let mut last_valid_idx = None;
        for (i, &val) in result.iter().enumerate() {
            if !val.is_nan() {
                last_valid_idx = Some(i);
                break;
            }
        }

        if last_valid_idx.is_none() {
            // All NaN
            return Ok(self.clone());
        }

        let mut last_valid_idx = last_valid_idx.unwrap();
        let mut last_valid_val = result[last_valid_idx];

        for i in (last_valid_idx + 1)..result.len() {
            if result[i].is_nan() {
                // Find next valid value
                let mut next_valid_idx = None;
                let mut next_valid_val = f64::NAN;

                #[allow(clippy::needless_range_loop)]
                for j in (i + 1)..result.len() {
                    if !result[j].is_nan() {
                        next_valid_idx = Some(j);
                        next_valid_val = result[j];
                        break;
                    }
                }

                if let Some(next_idx) = next_valid_idx {
                    // Linear interpolation
                    let gap = (next_idx - last_valid_idx) as f64;
                    let step = (next_valid_val - last_valid_val) / gap;
                    let offset = (i - last_valid_idx) as f64;
                    result[i] = last_valid_val + step * offset;
                }
                // else: leave as NaN (no next valid value)
            } else {
                last_valid_idx = i;
                last_valid_val = result[i];
            }
        }

        let mut new_df = self.clone();
        new_df
            .columns
            .insert(column.to_string(), Array1::from(result));
        Ok(new_df)
    }

    /// Melt a DataFrame from wide to long format (opposite of pivot).
    ///
    /// # Arguments
    /// * `id_vars` - Columns to keep as identifiers
    /// * `value_vars` - Columns to unpivot (if None, use all except id_vars)
    /// * `var_name` - Name for the variable column (default: "variable")
    /// * `value_name` - Name for the value column (default: "value")
    ///
    /// # Examples
    /// ```
    /// use greeners::DataFrame;
    ///
    /// // Wide format
    /// let df = DataFrame::builder()
    ///     .add_column("id", vec![1.0, 2.0])
    ///     .add_column("Jan", vec![100.0, 200.0])
    ///     .add_column("Feb", vec![150.0, 250.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// // Melt to long format
    /// let melted = df.melt(&["id"], None, "month", "sales").unwrap();
    /// // Result has columns: id, month, sales
    /// assert_eq!(melted.n_rows(), 4); // 2 ids * 2 months
    /// ```
    pub fn melt(
        &self,
        id_vars: &[&str],
        value_vars: Option<&[&str]>,
        var_name: &str,
        value_name: &str,
    ) -> Result<Self, GreenersError> {
        // Validate id_vars exist
        for &var in id_vars {
            if !self.has_column(var) {
                return Err(GreenersError::VariableNotFound(format!(
                    "ID variable '{}' not found",
                    var
                )));
            }
        }

        // Determine which columns to melt
        let cols_to_melt: Vec<String> = if let Some(vars) = value_vars {
            // Use specified columns
            for &var in vars {
                if !self.has_column(var) {
                    return Err(GreenersError::VariableNotFound(format!(
                        "Value variable '{}' not found",
                        var
                    )));
                }
            }
            vars.iter().map(|&s| s.to_string()).collect()
        } else {
            // Use all columns except id_vars
            let id_set: std::collections::HashSet<&str> = id_vars.iter().copied().collect();
            self.column_names()
                .into_iter()
                .filter(|name| !id_set.contains(name.as_str()))
                .collect()
        };

        if cols_to_melt.is_empty() {
            return Err(GreenersError::FormulaError(
                "No columns to melt".to_string(),
            ));
        }

        // Build melted data
        let mut result_data: HashMap<String, Vec<f64>> = HashMap::new();

        // Initialize columns
        for &id_var in id_vars {
            result_data.insert(id_var.to_string(), Vec::new());
        }
        result_data.insert(var_name.to_string(), Vec::new());
        result_data.insert(value_name.to_string(), Vec::new());

        // For each row in original DataFrame
        for i in 0..self.n_rows {
            // For each column to melt
            for col_name in &cols_to_melt {
                // Add id variable values
                for &id_var in id_vars {
                    let val = self.get(id_var)?[i];
                    result_data.get_mut(id_var).unwrap().push(val);
                }

                // Add variable name (encoded as number for simplicity)
                // In a real implementation, you might want to support string columns
                let var_idx = cols_to_melt.iter().position(|n| n == col_name).unwrap() as f64;
                result_data.get_mut(var_name).unwrap().push(var_idx);

                // Add value
                let val = self.get(col_name)?[i];
                result_data.get_mut(value_name).unwrap().push(val);
            }
        }

        // Convert to Array1
        let mut final_columns = HashMap::new();
        for (name, values) in result_data {
            final_columns.insert(name, Array1::from(values));
        }

        DataFrame::new(final_columns)
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

    // ========== TESTS FOR NEW TIME SERIES FEATURES ==========

    #[test]
    fn test_rolling_mean() {
        let df = DataFrame::builder()
            .add_column("x", vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .build()
            .unwrap();

        let result = df.rolling("x", 3, "mean").unwrap();
        let col = result.get("x_rolling_mean").unwrap();

        assert!(col[0].is_nan());
        assert!(col[1].is_nan());
        assert_eq!(col[2], 2.0); // (1+2+3)/3
        assert_eq!(col[3], 3.0); // (2+3+4)/3
        assert_eq!(col[4], 4.0); // (3+4+5)/3
    }

    #[test]
    fn test_rolling_sum() {
        let df = DataFrame::builder()
            .add_column("x", vec![1.0, 2.0, 3.0, 4.0])
            .build()
            .unwrap();

        let result = df.rolling("x", 2, "sum").unwrap();
        let col = result.get("x_rolling_sum").unwrap();

        assert!(col[0].is_nan());
        assert_eq!(col[1], 3.0); // 1+2
        assert_eq!(col[2], 5.0); // 2+3
        assert_eq!(col[3], 7.0); // 3+4
    }

    #[test]
    fn test_rolling_min_max() {
        let df = DataFrame::builder()
            .add_column("x", vec![5.0, 2.0, 8.0, 1.0, 9.0])
            .build()
            .unwrap();

        let result_min = df.rolling("x", 3, "min").unwrap();
        let col_min = result_min.get("x_rolling_min").unwrap();
        assert_eq!(col_min[2], 2.0);
        assert_eq!(col_min[3], 1.0);

        let result_max = df.rolling("x", 3, "max").unwrap();
        let col_max = result_max.get("x_rolling_max").unwrap();
        assert_eq!(col_max[2], 8.0);
        assert_eq!(col_max[3], 8.0);
        assert_eq!(col_max[4], 9.0);
    }

    #[test]
    fn test_rolling_std() {
        let df = DataFrame::builder()
            .add_column("x", vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
            .build()
            .unwrap();

        let result = df.rolling("x", 3, "std").unwrap();
        let col = result.get("x_rolling_std").unwrap();

        assert!(col[0].is_nan());
        assert!(col[1].is_nan());
        assert!(col[2] > 0.0); // Has some variation
    }

    #[test]
    fn test_cumsum() {
        let df = DataFrame::builder()
            .add_column("x", vec![1.0, 2.0, 3.0, 4.0])
            .build()
            .unwrap();

        let result = df.cumsum("x").unwrap();
        let col = result.get("x_cumsum").unwrap();

        assert_eq!(col[0], 1.0);
        assert_eq!(col[1], 3.0); // 1+2
        assert_eq!(col[2], 6.0); // 1+2+3
        assert_eq!(col[3], 10.0); // 1+2+3+4
    }

    #[test]
    fn test_cumprod() {
        let df = DataFrame::builder()
            .add_column("x", vec![2.0, 3.0, 4.0])
            .build()
            .unwrap();

        let result = df.cumprod("x").unwrap();
        let col = result.get("x_cumprod").unwrap();

        assert_eq!(col[0], 2.0);
        assert_eq!(col[1], 6.0); // 2*3
        assert_eq!(col[2], 24.0); // 2*3*4
    }

    #[test]
    fn test_cummax() {
        let df = DataFrame::builder()
            .add_column("x", vec![1.0, 5.0, 3.0, 7.0, 2.0])
            .build()
            .unwrap();

        let result = df.cummax("x").unwrap();
        let col = result.get("x_cummax").unwrap();

        assert_eq!(col[0], 1.0);
        assert_eq!(col[1], 5.0);
        assert_eq!(col[2], 5.0); // Still 5
        assert_eq!(col[3], 7.0); // New max
        assert_eq!(col[4], 7.0); // Still 7
    }

    #[test]
    fn test_cummin() {
        let df = DataFrame::builder()
            .add_column("x", vec![5.0, 2.0, 8.0, 1.0, 9.0])
            .build()
            .unwrap();

        let result = df.cummin("x").unwrap();
        let col = result.get("x_cummin").unwrap();

        assert_eq!(col[0], 5.0);
        assert_eq!(col[1], 2.0);
        assert_eq!(col[2], 2.0); // Still 2
        assert_eq!(col[3], 1.0); // New min
        assert_eq!(col[4], 1.0); // Still 1
    }

    #[test]
    fn test_shift_positive() {
        let df = DataFrame::builder()
            .add_column("x", vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .build()
            .unwrap();

        let result = df.shift("x", 1).unwrap();
        let col = result.get("x_shift_1").unwrap();

        assert!(col[0].is_nan()); // First value is NaN
        assert_eq!(col[1], 1.0);
        assert_eq!(col[2], 2.0);
        assert_eq!(col[3], 3.0);
        assert_eq!(col[4], 4.0);
    }

    #[test]
    fn test_shift_negative() {
        let df = DataFrame::builder()
            .add_column("x", vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .build()
            .unwrap();

        let result = df.shift("x", -1).unwrap();
        let col = result.get("x_shift_-1").unwrap();

        assert_eq!(col[0], 2.0);
        assert_eq!(col[1], 3.0);
        assert_eq!(col[2], 4.0);
        assert_eq!(col[3], 5.0);
        assert!(col[4].is_nan()); // Last value is NaN
    }

    #[test]
    fn test_shift_multiple() {
        let df = DataFrame::builder()
            .add_column("x", vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .build()
            .unwrap();

        let result = df.shift("x", 2).unwrap();
        let col = result.get("x_shift_2").unwrap();

        assert!(col[0].is_nan());
        assert!(col[1].is_nan());
        assert_eq!(col[2], 1.0);
        assert_eq!(col[3], 2.0);
        assert_eq!(col[4], 3.0);
    }

    #[test]
    fn test_quantile() {
        let df = DataFrame::builder()
            .add_column("x", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
            .build()
            .unwrap();

        let q25 = df.quantile("x", 0.25).unwrap();
        let q50 = df.quantile("x", 0.50).unwrap();
        let q75 = df.quantile("x", 0.75).unwrap();

        assert_eq!(q25, 3.25);
        assert_eq!(q50, 5.5);
        assert_eq!(q75, 7.75);
    }

    #[test]
    fn test_quantile_median() {
        let df = DataFrame::builder()
            .add_column("x", vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .build()
            .unwrap();

        let median = df.quantile("x", 0.5).unwrap();
        assert_eq!(median, 3.0);
    }

    #[test]
    fn test_rank_descending() {
        let df = DataFrame::builder()
            .add_column("x", vec![5.0, 2.0, 8.0, 2.0, 1.0])
            .build()
            .unwrap();

        let result = df.rank("x", false).unwrap();
        let col = result.get("x_rank").unwrap();

        assert_eq!(col[0], 2.0); // 5 is 2nd highest
        assert_eq!(col[1], 3.0); // 2 (first occurrence)
        assert_eq!(col[2], 1.0); // 8 is highest
        assert_eq!(col[3], 4.0); // 2 (second occurrence)
        assert_eq!(col[4], 5.0); // 1 is lowest
    }

    #[test]
    fn test_rank_ascending() {
        let df = DataFrame::builder()
            .add_column("x", vec![5.0, 2.0, 8.0, 1.0])
            .build()
            .unwrap();

        let result = df.rank("x", true).unwrap();
        let col = result.get("x_rank").unwrap();

        assert_eq!(col[0], 3.0); // 5 is 3rd lowest
        assert_eq!(col[1], 2.0); // 2 is 2nd lowest
        assert_eq!(col[2], 4.0); // 8 is highest
        assert_eq!(col[3], 1.0); // 1 is lowest
    }

    #[test]
    fn test_drop_duplicates() {
        let df = DataFrame::builder()
            .add_column("id", vec![1.0, 2.0, 3.0, 4.0])
            .add_column("category", vec![1.0, 2.0, 1.0, 3.0])
            .add_column("value", vec![10.0, 20.0, 30.0, 40.0])
            .build()
            .unwrap();

        let result = df.drop_duplicates("category").unwrap();

        assert_eq!(result.n_rows(), 3); // 4 rows -> 3 rows (one duplicate removed)

        let cat = result.get("category").unwrap();
        let val = result.get("value").unwrap();

        assert_eq!(cat[0], 1.0);
        assert_eq!(cat[1], 2.0);
        assert_eq!(cat[2], 3.0);

        // Should keep first occurrence (value 10.0, not 30.0)
        assert_eq!(val[0], 10.0);
        assert_eq!(val[1], 20.0);
        assert_eq!(val[2], 40.0);
    }

    #[test]
    fn test_drop_duplicates_all_unique() {
        let df = DataFrame::builder()
            .add_column("x", vec![1.0, 2.0, 3.0])
            .build()
            .unwrap();

        let result = df.drop_duplicates("x").unwrap();
        assert_eq!(result.n_rows(), 3); // No change
    }

    #[test]
    fn test_drop_duplicates_all_same() {
        let df = DataFrame::builder()
            .add_column("x", vec![5.0, 5.0, 5.0, 5.0])
            .add_column("y", vec![1.0, 2.0, 3.0, 4.0])
            .build()
            .unwrap();

        let result = df.drop_duplicates("x").unwrap();
        assert_eq!(result.n_rows(), 1); // Only first occurrence kept
    }

    #[test]
    fn test_interpolate_simple() {
        let df = DataFrame::builder()
            .add_column("x", vec![1.0, f64::NAN, 3.0])
            .build()
            .unwrap();

        let result = df.interpolate("x").unwrap();
        let col = result.get("x").unwrap();

        assert_eq!(col[0], 1.0);
        assert_eq!(col[1], 2.0); // Interpolated
        assert_eq!(col[2], 3.0);
    }

    #[test]
    fn test_interpolate_multiple_gaps() {
        let df = DataFrame::builder()
            .add_column("x", vec![10.0, f64::NAN, f64::NAN, 40.0])
            .build()
            .unwrap();

        let result = df.interpolate("x").unwrap();
        let col = result.get("x").unwrap();

        assert_eq!(col[0], 10.0);
        assert_eq!(col[1], 20.0); // (10+40)/2 - first third
        assert_eq!(col[2], 30.0); // (10+40)/2 - second third
        assert_eq!(col[3], 40.0);
    }

    #[test]
    fn test_interpolate_no_gaps() {
        let df = DataFrame::builder()
            .add_column("x", vec![1.0, 2.0, 3.0, 4.0])
            .build()
            .unwrap();

        let result = df.interpolate("x").unwrap();
        let col = result.get("x").unwrap();

        // Should remain unchanged
        assert_eq!(col[0], 1.0);
        assert_eq!(col[1], 2.0);
        assert_eq!(col[2], 3.0);
        assert_eq!(col[3], 4.0);
    }

    #[test]
    fn test_interpolate_leading_trailing_nan() {
        let df = DataFrame::builder()
            .add_column("x", vec![f64::NAN, 2.0, f64::NAN, 4.0, f64::NAN])
            .build()
            .unwrap();

        let result = df.interpolate("x").unwrap();
        let col = result.get("x").unwrap();

        assert!(col[0].is_nan()); // Leading NaN unchanged
        assert_eq!(col[1], 2.0);
        assert_eq!(col[2], 3.0); // Interpolated between 2 and 4
        assert_eq!(col[3], 4.0);
        assert!(col[4].is_nan()); // Trailing NaN unchanged
    }

    #[test]
    fn test_rolling_invalid_column() {
        let df = DataFrame::builder()
            .add_column("x", vec![1.0, 2.0, 3.0])
            .build()
            .unwrap();

        let result = df.rolling("nonexistent", 3, "mean");
        assert!(result.is_err());
    }

    #[test]
    fn test_shift_invalid_column() {
        let df = DataFrame::builder()
            .add_column("x", vec![1.0, 2.0, 3.0])
            .build()
            .unwrap();

        let result = df.shift("nonexistent", 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantile_bounds() {
        let df = DataFrame::builder()
            .add_column("x", vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .build()
            .unwrap();

        let q0 = df.quantile("x", 0.0).unwrap();
        let q100 = df.quantile("x", 1.0).unwrap();

        assert_eq!(q0, 1.0); // Minimum
        assert_eq!(q100, 5.0); // Maximum
    }
}
