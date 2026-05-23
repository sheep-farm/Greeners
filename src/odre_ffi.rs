//! FFI wrapper for Odre plugin system
//!
//! This module provides C ABI exports for loading Greeners as a dynamic plugin
//! in the Odre node editor.
//!
//! Build with: `cargo build --release --features odre-ffi`

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::{CovarianceType, OlsResult, OLS};

/// Data types that flow between Odre nodes (must match Odre's PluginData)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum PluginData {
    Numbers(Vec<f64>),
    Single(f64),
    Matrix {
        rows: usize,
        cols: usize,
        data: Vec<f64>,
    },
    Text(String),
    Boolean(bool),
    Table {
        columns: Vec<String>,
        data: Vec<Vec<f64>>,
    },
    None,
}

/// OLS result formatted for Odre
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OlsOutputs {
    coefficients: Vec<f64>,
    std_errors: Vec<f64>,
    t_values: Vec<f64>,
    p_values: Vec<f64>,
    r_squared: f64,
    adj_r_squared: f64,
    f_statistic: f64,
    n_obs: usize,
}

impl From<&OlsResult> for OlsOutputs {
    fn from(r: &OlsResult) -> Self {
        OlsOutputs {
            coefficients: r.params.to_vec(),
            std_errors: r.std_errors.to_vec(),
            t_values: r.t_values.to_vec(),
            p_values: r.p_values.to_vec(),
            r_squared: r.r_squared,
            adj_r_squared: r.adj_r_squared,
            f_statistic: r.f_statistic,
            n_obs: r.n_obs,
        }
    }
}

/// Returns plugin metadata as JSON.
#[no_mangle]
pub extern "C" fn odre_get_metadata() -> *const c_char {
    let json = r#"{
        "id": "greeners.ols",
        "name": "OLS Regression (Greeners)",
        "version": "1.4.0",
        "category": "Econometrics",
        "description": "Ordinary Least Squares regression with robust standard errors (HC1-HC4, Newey-West, Clustered)",
        "color": [0.2, 0.6, 0.3, 1.0],
        "inputs": [
            {"name": "y", "type": "Numbers", "description": "Dependent variable (n observations)"},
            {"name": "X", "type": "Matrix", "description": "Independent variables (n x k matrix, include intercept column if desired)"}
        ],
        "outputs": [
            {"name": "coefficients", "type": "Numbers", "description": "Estimated coefficients"},
            {"name": "std_errors", "type": "Numbers", "description": "Standard errors"},
            {"name": "t_values", "type": "Numbers", "description": "T-statistics"},
            {"name": "p_values", "type": "Numbers", "description": "P-values"},
            {"name": "r_squared", "type": "Single", "description": "R-squared"},
            {"name": "summary", "type": "Text", "description": "Full regression summary"}
        ]
    }"#;

    // Note: This leaks memory, but it's only called once at load time
    CString::new(json).unwrap().into_raw()
}

/// Executes OLS regression.
///
/// # Inputs
/// - inputs[0]: y (Numbers) - dependent variable
/// - inputs[1]: X (Matrix) - independent variables
///
/// # Outputs
/// - coefficients, std_errors, t_values, p_values, r_squared, summary
#[no_mangle]
pub extern "C" fn odre_execute(inputs_json: *const c_char) -> *mut c_char {
    let result = execute_internal(inputs_json);
    CString::new(result).unwrap().into_raw()
}

fn execute_internal(inputs_json: *const c_char) -> String {
    // Parse input
    let inputs_str = unsafe {
        if inputs_json.is_null() {
            return error_response("Null input pointer");
        }
        match CStr::from_ptr(inputs_json).to_str() {
            Ok(s) => s,
            Err(_) => return error_response("Invalid UTF-8 in input"),
        }
    };

    let inputs: Vec<PluginData> = match serde_json::from_str(inputs_str) {
        Ok(v) => v,
        Err(e) => return error_response(&format!("JSON parse error: {}", e)),
    };

    // Extract y (dependent variable)
    let y_vec = match inputs.get(0) {
        Some(PluginData::Numbers(nums)) => nums.clone(),
        _ => return error_response("Input 0 (y) must be Numbers"),
    };

    // Extract X (independent variables)
    let (x_data, rows, cols) = match inputs.get(1) {
        Some(PluginData::Matrix { rows, cols, data }) => (data.clone(), *rows, *cols),
        Some(PluginData::Numbers(nums)) => {
            // If X is just Numbers, treat as single column with intercept
            let n = nums.len();
            let mut data = Vec::with_capacity(n * 2);
            for &x in nums {
                data.push(1.0); // intercept
                data.push(x);
            }
            (data, n, 2)
        }
        _ => return error_response("Input 1 (X) must be Matrix or Numbers"),
    };

    if y_vec.len() != rows {
        return error_response(&format!(
            "Dimension mismatch: y has {} obs, X has {} rows",
            y_vec.len(),
            rows
        ));
    }

    // Convert to ndarray
    let y = Array1::from(y_vec);
    let x = match Array2::from_shape_vec((rows, cols), x_data) {
        Ok(arr) => arr,
        Err(e) => return error_response(&format!("Matrix shape error: {}", e)),
    };

    // Run OLS with HC1 robust standard errors
    let result = match OLS::fit(&y, &x, CovarianceType::HC1) {
        Ok(r) => r,
        Err(e) => return error_response(&format!("OLS error: {}", e)),
    };

    // Format summary
    let summary = format!("{}", result);

    // Build outputs
    let outputs = vec![
        PluginData::Numbers(result.params.to_vec()),
        PluginData::Numbers(result.std_errors.to_vec()),
        PluginData::Numbers(result.t_values.to_vec()),
        PluginData::Numbers(result.p_values.to_vec()),
        PluginData::Single(result.r_squared),
        PluginData::Text(summary),
    ];

    serde_json::to_string(&outputs).unwrap_or_else(|_| "[]".to_string())
}

fn error_response(msg: &str) -> String {
    let outputs = vec![
        PluginData::None,
        PluginData::None,
        PluginData::None,
        PluginData::None,
        PluginData::None,
        PluginData::Text(format!("Error: {}", msg)),
    ];
    serde_json::to_string(&outputs).unwrap_or_else(|_| "[]".to_string())
}

/// Frees memory allocated by odre_execute.
#[no_mangle]
pub extern "C" fn odre_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            drop(CString::from_raw(ptr));
        }
    }
}
