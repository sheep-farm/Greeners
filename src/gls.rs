use crate::{CovarianceType, DataFrame, Formula, GreenersError, OLS};
use ndarray::{s, Array1, Array2, Axis};
use std::fmt;

/// FGLS Result
#[derive(Debug)]
pub struct FglsResult {
    pub method: String, // "WLS" or "Cochrane-Orcutt"
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_values: Array1<f64>,
    pub p_values: Array1<f64>,
    pub r_squared: f64,
    pub rho: Option<f64>,    // Only for Cochrane-Orcutt
    pub iter: Option<usize>, // Iterations until convergence
    pub variable_names: Option<Vec<String>>,
}

impl fmt::Display for FglsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\n{:=^78}",
            format!(" FGLS Estimation ({}) ", self.method)
        )?;
        if let Some(r) = self.rho {
            writeln!(f, "Autocorrelation (rho): {:>10.4}", r)?;
        }
        writeln!(f, "{:<20} {:>15.4}", "R-squared:", self.r_squared)?;

        writeln!(f, "\n{:-^78}", "")?;
        writeln!(
            f,
            "{:<10} | {:>10} | {:>10} | {:>8} | {:>8}",
            "Variable", "coef", "std err", "t", "P>|t|"
        )?;
        writeln!(f, "{:-^78}", "")?;

        for i in 0..self.params.len() {
            let var_name = if let Some(ref names) = self.variable_names {
                if i < names.len() {
                    names[i].clone()
                } else {
                    format!("x{}", i)
                }
            } else {
                format!("x{}", i)
            };

            writeln!(
                f,
                "{:<10} | {:>10.4} | {:>10.4} | {:>8.3} | {:>8.3}",
                var_name, self.params[i], self.std_errors[i], self.t_values[i], self.p_values[i]
            )?;
        }
        writeln!(f, "{:=^78}", "")
    }
}

pub struct FGLS;

impl FGLS {
    /// Weighted Least Squares (WLS) from a formula and DataFrame.
    ///
    /// # Examples
    /// ```
    /// use greeners::{FGLS, DataFrame, Formula};
    /// use ndarray::Array1;
    /// use std::collections::HashMap;
    ///
    /// let mut data = HashMap::new();
    /// data.insert("y".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
    /// data.insert("x1".to_string(), Array1::from(vec![1.0, 2.0, 3.0]));
    /// data.insert("weight".to_string(), Array1::from(vec![1.0, 1.0, 1.0]));
    ///
    /// let df = DataFrame::new(data).unwrap();
    /// let formula = Formula::parse("y ~ x1").unwrap();
    /// let weights = df.get("weight").unwrap();
    ///
    /// let result = FGLS::wls_from_formula(&formula, &df, weights).unwrap();
    /// ```
    pub fn wls_from_formula(
        formula: &Formula,
        data: &DataFrame,
        weights: &Array1<f64>,
    ) -> Result<FglsResult, GreenersError> {
        let (y, x) = data.to_design_matrix(formula)?;

        // Build variable names from formula
        let mut var_names = Vec::new();
        if formula.intercept {
            var_names.push("const".to_string());
        }
        for var in &formula.independents {
            var_names.push(var.clone());
        }

        Self::wls_with_names(&y, &x, weights, Some(var_names))
    }

    /// Weighted Least Squares (WLS)
    /// Used when there is known or estimated heteroscedasticity.
    /// Weights (w) should be inversely proportional to variance: w_i = 1 / sigma_i^2
    pub fn wls(
        y: &Array1<f64>,
        x: &Array2<f64>,
        weights: &Array1<f64>,
    ) -> Result<FglsResult, GreenersError> {
        Self::wls_with_names(y, x, weights, None)
    }

    pub fn wls_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        weights: &Array1<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<FglsResult, GreenersError> {
        let n = y.len();
        if weights.len() != n {
            return Err(GreenersError::ShapeMismatch(
                "Weights length mismatch".into(),
            ));
        }

        // GLS Transformation: Multiply X and y by the square root of weights
        // y* = sqrt(w) * y
        // X* = sqrt(w) * X
        let sqrt_w = weights.mapv(f64::sqrt);

        let y_transformed = y * &sqrt_w;

        // Broadcast multiplication of weight column by X rows
        let mut x_transformed = x.clone();
        for (i, mut row) in x_transformed.axis_iter_mut(Axis(0)).enumerate() {
            row *= sqrt_w[i];
        }

        // Run OLS on transformed data
        let ols = OLS::fit(&y_transformed, &x_transformed, CovarianceType::NonRobust)?;

        Ok(FglsResult {
            method: "WLS".to_string(),
            params: ols.params,
            std_errors: ols.std_errors,
            t_values: ols.t_values,
            p_values: ols.p_values,
            r_squared: ols.r_squared,
            rho: None,
            iter: None,
            variable_names,
        })
    }

    /// Cochrane-Orcutt Iterative Procedure (AR(1)) from a formula and DataFrame.
    pub fn cochrane_orcutt_from_formula(
        formula: &Formula,
        data: &DataFrame,
    ) -> Result<FglsResult, GreenersError> {
        let (y, x) = data.to_design_matrix(formula)?;

        // Build variable names from formula
        let mut var_names = Vec::new();
        if formula.intercept {
            var_names.push("const".to_string());
        }
        for var in &formula.independents {
            var_names.push(var.clone());
        }

        Self::cochrane_orcutt_with_names(&y, &x, Some(var_names))
    }

    /// Cochrane-Orcutt Iterative Procedure (AR(1))
    /// Solves serial correlation: u_t = rho * u_{t-1} + e_t
    /// Recovers the efficiency (BLUE) that OLS loses.
    pub fn cochrane_orcutt(y: &Array1<f64>, x: &Array2<f64>) -> Result<FglsResult, GreenersError> {
        Self::cochrane_orcutt_with_names(y, x, None)
    }

    pub fn cochrane_orcutt_with_names(
        y: &Array1<f64>,
        x: &Array2<f64>,
        variable_names: Option<Vec<String>>,
    ) -> Result<FglsResult, GreenersError> {
        let n = y.len();
        // let k = x.ncols();
        let tol = 1e-6;
        let max_iter = 100;

        // 1. Initial OLS to get residuals
        let initial_ols = OLS::fit(y, x, CovarianceType::NonRobust)?;
        let mut residuals = y - &x.dot(&initial_ols.params);

        let mut rho = 0.0;
        let mut iter = 0;
        let mut diff = 1.0;
        let mut final_ols = initial_ols; // Placeholder

        while diff > tol && iter < max_iter {
            let old_rho = rho;

            // 2. Estimar Rho regressando resid[t] em resid[t-1]
            let u_t = residuals.slice(s![1..]).to_owned();
            let u_tm1 = residuals.slice(s![..n - 1]).to_owned();

            // Regressão simples sem intercepto: rho = (u_{t-1}' u_{t-1})^-1 u_{t-1}' u_t
            let num = u_tm1.dot(&u_t);
            let den = u_tm1.dot(&u_tm1);
            rho = num / den;

            // 3. Transformação Cochrane-Orcutt (Quase-Diferença)
            // y*_t = y_t - rho * y_{t-1}
            // x*_t = x_t - rho * x_{t-1}
            // Nota: Perdemos a primeira observação (n vira n-1)
            let y_t = y.slice(s![1..]);
            let y_tm1 = y.slice(s![..n - 1]);
            let y_star = &y_t - &(&y_tm1 * rho);

            let x_t = x.slice(s![1.., ..]);
            let x_tm1 = x.slice(s![..n - 1, ..]);
            let x_star = &x_t - &(&x_tm1 * rho);

            // 4. Re-estimar OLS nos dados transformados
            final_ols = OLS::fit(
                &y_star.to_owned(),
                &x_star.to_owned(),
                CovarianceType::NonRobust,
            )?;

            // Atualizar resíduos ORIGINAIS (não transformados) para próxima iteração
            residuals = y - &x.dot(&final_ols.params);

            diff = (rho - old_rho).abs();
            iter += 1;
        }

        Ok(FglsResult {
            method: "Cochrane-Orcutt AR(1)".to_string(),
            params: final_ols.params,
            std_errors: final_ols.std_errors,
            t_values: final_ols.t_values,
            p_values: final_ols.p_values,
            r_squared: final_ols.r_squared,
            rho: Some(rho),
            iter: Some(iter),
            variable_names,
        })
    }
}
