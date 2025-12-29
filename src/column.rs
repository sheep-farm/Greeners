use ndarray::Array1;
use std::collections::HashMap;

/// Data type of a column
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Float,
    Categorical,
    Bool,
    Int,
}

/// Categorical column with string levels and integer codes
#[derive(Debug, Clone)]
pub struct CategoricalColumn {
    /// Unique category levels (e.g., ["SP", "RJ", "MG"])
    pub levels: Vec<String>,
    /// Integer codes mapping to levels (e.g., [0, 1, 0, 2])
    pub codes: Vec<u32>,
    /// Reverse mapping: level name -> code
    level_to_code: HashMap<String, u32>,
}

impl CategoricalColumn {
    /// Create a new categorical column from string values
    pub fn from_strings(values: Vec<String>) -> Self {
        let mut levels = Vec::new();
        let mut level_to_code = HashMap::new();
        let mut codes = Vec::new();

        for value in values {
            let code = if let Some(&existing_code) = level_to_code.get(&value) {
                existing_code
            } else {
                let new_code = levels.len() as u32;
                levels.push(value.clone());
                level_to_code.insert(value.clone(), new_code);
                new_code
            };
            codes.push(code);
        }

        CategoricalColumn {
            levels,
            codes,
            level_to_code,
        }
    }

    /// Create from existing levels and codes (for internal use)
    pub fn from_codes(levels: Vec<String>, codes: Vec<u32>) -> Self {
        let level_to_code = levels
            .iter()
            .enumerate()
            .map(|(i, level)| (level.clone(), i as u32))
            .collect();

        CategoricalColumn {
            levels,
            codes,
            level_to_code,
        }
    }

    /// Get the number of rows
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Get level name by code
    pub fn get_level(&self, code: u32) -> Option<&str> {
        self.levels.get(code as usize).map(|s| s.as_str())
    }

    /// Get code by level name
    pub fn get_code(&self, level: &str) -> Option<u32> {
        self.level_to_code.get(level).copied()
    }

    /// Get string value at index
    pub fn get_string(&self, index: usize) -> Option<&str> {
        self.codes.get(index).and_then(|&code| self.get_level(code))
    }

    /// Convert to string vector
    pub fn to_strings(&self) -> Vec<String> {
        self.codes
            .iter()
            .map(|&code| {
                self.get_level(code)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "NA".to_string())
            })
            .collect()
    }

    /// Convert to float codes (for numeric operations)
    pub fn to_float_codes(&self) -> Array1<f64> {
        Array1::from(self.codes.iter().map(|&c| c as f64).collect::<Vec<_>>())
    }

    /// Get number of unique levels
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Get value counts
    pub fn value_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for &code in &self.codes {
            if let Some(level) = self.get_level(code) {
                *counts.entry(level.to_string()).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Filter by indices
    pub fn filter_indices(&self, indices: &[usize]) -> Self {
        let codes = indices.iter().map(|&i| self.codes[i]).collect();
        CategoricalColumn::from_codes(self.levels.clone(), codes)
    }

    /// Create dummy variables (one-hot encoding)
    /// Returns HashMap of column_name -> Array1<f64>
    pub fn get_dummies(&self, prefix: &str, drop_first: bool) -> HashMap<String, Array1<f64>> {
        let mut dummies = HashMap::new();
        let start_idx = if drop_first { 1 } else { 0 };

        for (i, level) in self.levels.iter().enumerate().skip(start_idx) {
            let col_name = format!("{}_{}", prefix, level);
            let values: Vec<f64> = self
                .codes
                .iter()
                .map(|&code| if code == i as u32 { 1.0 } else { 0.0 })
                .collect();
            dummies.insert(col_name, Array1::from(values));
        }

        dummies
    }
}

/// Column enum supporting multiple data types
#[derive(Debug, Clone)]
pub enum Column {
    /// Numeric floating-point column
    Float(Array1<f64>),
    /// Categorical column with string levels
    Categorical(CategoricalColumn),
    /// Boolean column
    Bool(Array1<bool>),
    /// Integer column (signed 64-bit)
    Int(Array1<i64>),
}

impl Column {
    /// Get the data type of this column
    pub fn dtype(&self) -> DataType {
        match self {
            Column::Float(_) => DataType::Float,
            Column::Categorical(_) => DataType::Categorical,
            Column::Bool(_) => DataType::Bool,
            Column::Int(_) => DataType::Int,
        }
    }

    /// Get the number of elements in this column
    pub fn len(&self) -> usize {
        match self {
            Column::Float(arr) => arr.len(),
            Column::Categorical(cat) => cat.len(),
            Column::Bool(arr) => arr.len(),
            Column::Int(arr) => arr.len(),
        }
    }

    /// Check if the column is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Try to get as float array
    pub fn as_float(&self) -> Option<&Array1<f64>> {
        match self {
            Column::Float(arr) => Some(arr),
            Column::Categorical(_) => None,
            Column::Bool(_) => None,
            Column::Int(_) => None,
        }
    }

    /// Try to get as categorical
    pub fn as_categorical(&self) -> Option<&CategoricalColumn> {
        match self {
            Column::Float(_) => None,
            Column::Categorical(cat) => Some(cat),
            Column::Bool(_) => None,
            Column::Int(_) => None,
        }
    }

    /// Try to get as bool array
    pub fn as_bool(&self) -> Option<&Array1<bool>> {
        match self {
            Column::Float(_) => None,
            Column::Categorical(_) => None,
            Column::Bool(arr) => Some(arr),
            Column::Int(_) => None,
        }
    }

    /// Try to get as int array
    pub fn as_int(&self) -> Option<&Array1<i64>> {
        match self {
            Column::Float(_) => None,
            Column::Categorical(_) => None,
            Column::Bool(_) => None,
            Column::Int(arr) => Some(arr),
        }
    }

    /// Convert to float array (categorical -> codes as f64, bool -> 1.0/0.0, int -> f64)
    pub fn to_float(&self) -> Array1<f64> {
        match self {
            Column::Float(arr) => arr.clone(),
            Column::Categorical(cat) => cat.to_float_codes(),
            Column::Bool(arr) => Array1::from(
                arr.iter()
                    .map(|&b| if b { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>(),
            ),
            Column::Int(arr) => Array1::from(arr.iter().map(|&i| i as f64).collect::<Vec<_>>()),
        }
    }

    /// Filter by indices (for DataFrame operations)
    pub fn filter_indices(&self, indices: &[usize]) -> Self {
        match self {
            Column::Float(arr) => {
                let filtered: Vec<f64> = indices.iter().map(|&i| arr[i]).collect();
                Column::Float(Array1::from(filtered))
            }
            Column::Categorical(cat) => Column::Categorical(cat.filter_indices(indices)),
            Column::Bool(arr) => {
                let filtered: Vec<bool> = indices.iter().map(|&i| arr[i]).collect();
                Column::Bool(Array1::from(filtered))
            }
            Column::Int(arr) => {
                let filtered: Vec<i64> = indices.iter().map(|&i| arr[i]).collect();
                Column::Int(Array1::from(filtered))
            }
        }
    }

    /// Create from float array
    pub fn from_float(arr: Array1<f64>) -> Self {
        Column::Float(arr)
    }

    /// Create from string vector
    pub fn from_strings(values: Vec<String>) -> Self {
        Column::Categorical(CategoricalColumn::from_strings(values))
    }

    /// Create from bool array
    pub fn from_bool(arr: Array1<bool>) -> Self {
        Column::Bool(arr)
    }

    /// Create from int array
    pub fn from_int(arr: Array1<i64>) -> Self {
        Column::Int(arr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_categorical_from_strings() {
        let values = vec![
            "SP".to_string(),
            "RJ".to_string(),
            "SP".to_string(),
            "MG".to_string(),
        ];
        let cat = CategoricalColumn::from_strings(values);

        assert_eq!(cat.len(), 4);
        assert_eq!(cat.n_levels(), 3);
        assert_eq!(cat.levels, vec!["SP", "RJ", "MG"]);
        assert_eq!(cat.codes, vec![0, 1, 0, 2]);
    }

    #[test]
    fn test_categorical_get_string() {
        let values = vec!["A".to_string(), "B".to_string(), "A".to_string()];
        let cat = CategoricalColumn::from_strings(values);

        assert_eq!(cat.get_string(0), Some("A"));
        assert_eq!(cat.get_string(1), Some("B"));
        assert_eq!(cat.get_string(2), Some("A"));
    }

    #[test]
    fn test_categorical_to_strings() {
        let values = vec!["X".to_string(), "Y".to_string(), "X".to_string()];
        let cat = CategoricalColumn::from_strings(values.clone());

        assert_eq!(cat.to_strings(), values);
    }

    #[test]
    fn test_categorical_to_float_codes() {
        let values = vec!["A".to_string(), "B".to_string(), "A".to_string()];
        let cat = CategoricalColumn::from_strings(values);
        let codes = cat.to_float_codes();

        assert_eq!(codes[0], 0.0);
        assert_eq!(codes[1], 1.0);
        assert_eq!(codes[2], 0.0);
    }

    #[test]
    fn test_categorical_value_counts() {
        let values = vec![
            "SP".to_string(),
            "RJ".to_string(),
            "SP".to_string(),
            "SP".to_string(),
            "MG".to_string(),
        ];
        let cat = CategoricalColumn::from_strings(values);
        let counts = cat.value_counts();

        assert_eq!(counts.get("SP"), Some(&3));
        assert_eq!(counts.get("RJ"), Some(&1));
        assert_eq!(counts.get("MG"), Some(&1));
    }

    #[test]
    fn test_categorical_get_dummies() {
        let values = vec![
            "A".to_string(),
            "B".to_string(),
            "A".to_string(),
            "C".to_string(),
        ];
        let cat = CategoricalColumn::from_strings(values);

        // Without dropping first
        let dummies = cat.get_dummies("cat", false);
        assert_eq!(dummies.len(), 3); // A, B, C

        // With dropping first
        let dummies_drop = cat.get_dummies("cat", true);
        assert_eq!(dummies_drop.len(), 2); // B, C (A dropped)
    }

    #[test]
    fn test_column_dtype() {
        let float_col = Column::Float(Array1::from(vec![1.0, 2.0, 3.0]));
        let cat_col = Column::from_strings(vec!["A".to_string(), "B".to_string()]);

        assert_eq!(float_col.dtype(), DataType::Float);
        assert_eq!(cat_col.dtype(), DataType::Categorical);
    }

    #[test]
    fn test_column_len() {
        let float_col = Column::Float(Array1::from(vec![1.0, 2.0, 3.0]));
        let cat_col = Column::from_strings(vec!["A".to_string(), "B".to_string()]);

        assert_eq!(float_col.len(), 3);
        assert_eq!(cat_col.len(), 2);
    }

    #[test]
    fn test_column_to_float() {
        let float_col = Column::Float(Array1::from(vec![1.0, 2.0, 3.0]));
        let cat_col = Column::from_strings(vec!["A".to_string(), "B".to_string(), "A".to_string()]);

        let float_arr = float_col.to_float();
        assert_eq!(float_arr[0], 1.0);

        let cat_arr = cat_col.to_float();
        assert_eq!(cat_arr[0], 0.0);
        assert_eq!(cat_arr[1], 1.0);
        assert_eq!(cat_arr[2], 0.0);
    }

    #[test]
    fn test_column_filter_indices() {
        let float_col = Column::Float(Array1::from(vec![1.0, 2.0, 3.0, 4.0]));
        let filtered = float_col.filter_indices(&[0, 2, 3]);

        if let Column::Float(arr) = filtered {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], 1.0);
            assert_eq!(arr[1], 3.0);
            assert_eq!(arr[2], 4.0);
        } else {
            panic!("Expected Float column");
        }
    }

    #[test]
    fn test_bool_column_creation() {
        let bool_col = Column::from_bool(Array1::from(vec![true, false, true, false]));

        assert_eq!(bool_col.dtype(), DataType::Bool);
        assert_eq!(bool_col.len(), 4);
        assert!(!bool_col.is_empty());
    }

    #[test]
    fn test_bool_column_as_bool() {
        let bool_col = Column::from_bool(Array1::from(vec![true, false, true]));

        let arr = bool_col.as_bool().unwrap();
        assert_eq!(arr[0], true);
        assert_eq!(arr[1], false);
        assert_eq!(arr[2], true);
    }

    #[test]
    fn test_bool_column_to_float() {
        let bool_col = Column::from_bool(Array1::from(vec![true, false, true, false]));

        let float_arr = bool_col.to_float();
        assert_eq!(float_arr[0], 1.0);
        assert_eq!(float_arr[1], 0.0);
        assert_eq!(float_arr[2], 1.0);
        assert_eq!(float_arr[3], 0.0);
    }

    #[test]
    fn test_bool_column_filter_indices() {
        let bool_col = Column::from_bool(Array1::from(vec![true, false, true, false, true]));
        let filtered = bool_col.filter_indices(&[0, 2, 4]);

        if let Column::Bool(arr) = filtered {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], true);
            assert_eq!(arr[1], true);
            assert_eq!(arr[2], true);
        } else {
            panic!("Expected Bool column");
        }
    }

    #[test]
    fn test_column_type_accessors() {
        let float_col = Column::Float(Array1::from(vec![1.0, 2.0]));
        let cat_col = Column::from_strings(vec!["A".to_string()]);
        let bool_col = Column::from_bool(Array1::from(vec![true, false]));
        let int_col = Column::from_int(Array1::from(vec![1, 2, 3]));

        // Float column
        assert!(float_col.as_float().is_some());
        assert!(float_col.as_categorical().is_none());
        assert!(float_col.as_bool().is_none());
        assert!(float_col.as_int().is_none());

        // Categorical column
        assert!(cat_col.as_float().is_none());
        assert!(cat_col.as_categorical().is_some());
        assert!(cat_col.as_bool().is_none());
        assert!(cat_col.as_int().is_none());

        // Bool column
        assert!(bool_col.as_float().is_none());
        assert!(bool_col.as_categorical().is_none());
        assert!(bool_col.as_bool().is_some());
        assert!(bool_col.as_int().is_none());

        // Int column
        assert!(int_col.as_float().is_none());
        assert!(int_col.as_categorical().is_none());
        assert!(int_col.as_bool().is_none());
        assert!(int_col.as_int().is_some());
    }

    #[test]
    fn test_int_column_creation() {
        let int_col = Column::from_int(Array1::from(vec![1, 2, 3, 4, 5]));

        assert_eq!(int_col.dtype(), DataType::Int);
        assert_eq!(int_col.len(), 5);
        assert!(!int_col.is_empty());
    }

    #[test]
    fn test_int_column_as_int() {
        let int_col = Column::from_int(Array1::from(vec![10, 20, 30]));

        let arr = int_col.as_int().unwrap();
        assert_eq!(arr[0], 10);
        assert_eq!(arr[1], 20);
        assert_eq!(arr[2], 30);
    }

    #[test]
    fn test_int_column_to_float() {
        let int_col = Column::from_int(Array1::from(vec![100, 200, 300, 400]));

        let float_arr = int_col.to_float();
        assert_eq!(float_arr[0], 100.0);
        assert_eq!(float_arr[1], 200.0);
        assert_eq!(float_arr[2], 300.0);
        assert_eq!(float_arr[3], 400.0);
    }

    #[test]
    fn test_int_column_filter_indices() {
        let int_col = Column::from_int(Array1::from(vec![1, 2, 3, 4, 5]));
        let filtered = int_col.filter_indices(&[0, 2, 4]);

        if let Column::Int(arr) = filtered {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], 1);
            assert_eq!(arr[1], 3);
            assert_eq!(arr[2], 5);
        } else {
            panic!("Expected Int column");
        }
    }

    #[test]
    fn test_int_column_negative_values() {
        let int_col = Column::from_int(Array1::from(vec![-10, -5, 0, 5, 10]));

        let arr = int_col.as_int().unwrap();
        assert_eq!(arr[0], -10);
        assert_eq!(arr[2], 0);
        assert_eq!(arr[4], 10);

        let float_arr = int_col.to_float();
        assert_eq!(float_arr[0], -10.0);
        assert_eq!(float_arr[4], 10.0);
    }
}
