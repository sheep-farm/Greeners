//! Export functionality for model results as LaTeX, HTML, and CSV.

/// Holds all data needed to render an export table.
#[derive(Debug, Clone)]
pub struct ExportData {
    pub model_name: String,
    pub dep_var: String,
    pub n_obs: usize,
    pub param_names: Vec<String>,
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub t_values: Vec<f64>,
    pub p_values: Vec<f64>,
    pub conf_lower: Vec<f64>,
    pub conf_upper: Vec<f64>,
    pub extra_stats: Vec<(String, String)>,
}

/// Trait for exporting model results to various formats.
pub trait ExportableResult {
    /// Return the structured data for this model result.
    fn export_data(&self) -> ExportData;

    /// Render a booktabs-style LaTeX table (like stargazer in R).
    fn to_latex(&self) -> String {
        let d = self.export_data();
        let mut s = String::new();
        s.push_str("\\begin{table}[htbp]\n\\centering\n");
        s.push_str(&format!(
            "\\caption{{{}: Dependent Variable = {}}}\n",
            d.model_name, d.dep_var
        ));
        s.push_str("\\begin{tabular}{lcccccc}\n\\toprule\n");
        s.push_str(
            "Variable & Coef. & Std. Err. & t/z & P$>|$t$|$ & [0.025 & 0.975] \\\\\n\\midrule\n",
        );
        for i in 0..d.param_names.len() {
            s.push_str(&format!(
                "{} & {:.4} & {:.4} & {:.4} & {:.4} & {:.4} & {:.4} \\\\\n",
                d.param_names[i],
                d.coefficients[i],
                d.std_errors[i],
                d.t_values[i],
                d.p_values[i],
                d.conf_lower[i],
                d.conf_upper[i],
            ));
        }
        s.push_str("\\midrule\n");
        for (key, val) in &d.extra_stats {
            s.push_str(&format!(
                "{} & \\multicolumn{{6}}{{c}}{{{}}} \\\\\n",
                key, val
            ));
        }
        s.push_str(&format!(
            "N & \\multicolumn{{6}}{{c}}{{{}}} \\\\\n",
            d.n_obs
        ));
        s.push_str("\\bottomrule\n\\end{tabular}\n\\end{table}\n");
        s
    }

    /// Render a simple styled HTML table.
    fn to_html(&self) -> String {
        let d = self.export_data();
        let mut s = String::new();
        s.push_str("<table style=\"border-collapse:collapse;font-family:monospace;\">\n");
        s.push_str(&format!(
            "<caption><b>{}</b>: Dependent Variable = {}</caption>\n",
            d.model_name, d.dep_var
        ));
        s.push_str("<thead><tr style=\"border-bottom:2px solid black;\">\n");
        s.push_str("<th>Variable</th><th>Coef.</th><th>Std. Err.</th><th>t/z</th><th>P&gt;|t|</th><th>[0.025</th><th>0.975]</th>\n");
        s.push_str("</tr></thead>\n<tbody>\n");
        for i in 0..d.param_names.len() {
            s.push_str(&format!(
                "<tr><td>{}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td></tr>\n",
                d.param_names[i],
                d.coefficients[i],
                d.std_errors[i],
                d.t_values[i],
                d.p_values[i],
                d.conf_lower[i],
                d.conf_upper[i],
            ));
        }
        s.push_str("</tbody>\n<tfoot>\n");
        for (key, val) in &d.extra_stats {
            s.push_str(&format!(
                "<tr><td>{}</td><td colspan=\"6\">{}</td></tr>\n",
                key, val
            ));
        }
        s.push_str(&format!(
            "<tr style=\"border-top:2px solid black;\"><td>N</td><td colspan=\"6\">{}</td></tr>\n",
            d.n_obs
        ));
        s.push_str("</tfoot>\n</table>\n");
        s
    }

    /// Render a standard CSV with headers.
    fn to_csv(&self) -> String {
        let d = self.export_data();
        let mut s = String::new();
        s.push_str("Variable,Coef,Std_Err,t_z,P_value,CI_Lower,CI_Upper\n");
        for i in 0..d.param_names.len() {
            s.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                d.param_names[i],
                d.coefficients[i],
                d.std_errors[i],
                d.t_values[i],
                d.p_values[i],
                d.conf_lower[i],
                d.conf_upper[i],
            ));
        }
        s
    }
}

// --- Implementations ---

use crate::ols::OlsResult;

impl ExportableResult for OlsResult {
    fn export_data(&self) -> ExportData {
        let k = self.params.len();
        let names = self
            .variable_names
            .clone()
            .unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());
        ExportData {
            model_name: "OLS".to_string(),
            dep_var: names.first().cloned().unwrap_or_else(|| "y".to_string()),
            n_obs: self.n_obs,
            param_names: names,
            coefficients: self.params.to_vec(),
            std_errors: self.std_errors.to_vec(),
            t_values: self.t_values.to_vec(),
            p_values: self.p_values.to_vec(),
            conf_lower: self.conf_lower.to_vec(),
            conf_upper: self.conf_upper.to_vec(),
            extra_stats: vec![
                ("R-squared".to_string(), format!("{:.4}", self.r_squared)),
                (
                    "Adj. R-squared".to_string(),
                    format!("{:.4}", self.adj_r_squared),
                ),
                (
                    "F-statistic".to_string(),
                    format!("{:.4}", self.f_statistic),
                ),
                (
                    "Log-Likelihood".to_string(),
                    format!("{:.4}", self.log_likelihood),
                ),
                ("AIC".to_string(), format!("{:.4}", self.aic)),
                ("BIC".to_string(), format!("{:.4}", self.bic)),
            ],
        }
    }
}

use crate::glm::GlmResult;

impl ExportableResult for GlmResult {
    fn export_data(&self) -> ExportData {
        let k = self.params.len();
        let names = self
            .variable_names
            .clone()
            .unwrap_or_else(|| (0..k).map(|i| format!("x{}", i)).collect());
        ExportData {
            model_name: format!("GLM ({:?}/{:?})", self.family, self.link),
            dep_var: names.first().cloned().unwrap_or_else(|| "y".to_string()),
            n_obs: self.n_obs,
            param_names: names,
            coefficients: self.params.to_vec(),
            std_errors: self.std_errors.to_vec(),
            t_values: self.z_values.to_vec(),
            p_values: self.p_values.to_vec(),
            conf_lower: self.conf_lower.to_vec(),
            conf_upper: self.conf_upper.to_vec(),
            extra_stats: vec![
                ("Deviance".to_string(), format!("{:.4}", self.deviance)),
                (
                    "Null Deviance".to_string(),
                    format!("{:.4}", self.null_deviance),
                ),
                ("Pseudo R2".to_string(), format!("{:.4}", self.pseudo_r2)),
                (
                    "Log-Likelihood".to_string(),
                    format!("{:.4}", self.log_likelihood),
                ),
                ("AIC".to_string(), format!("{:.4}", self.aic)),
                ("BIC".to_string(), format!("{:.4}", self.bic)),
                ("Dispersion".to_string(), format!("{:.4}", self.dispersion)),
            ],
        }
    }
}

use crate::arima::ArimaResult;

impl ExportableResult for ArimaResult {
    fn export_data(&self) -> ExportData {
        ExportData {
            model_name: format!("ARIMA{:?}", self.order),
            dep_var: "y".to_string(),
            n_obs: self.n_obs,
            param_names: self.param_names.clone(),
            coefficients: {
                let mut v = vec![self.intercept];
                v.extend(self.ar_params.iter());
                v.extend(self.ma_params.iter());
                v.extend(self.seasonal_ar_params.iter());
                v.extend(self.seasonal_ma_params.iter());
                if let Some(ref ep) = self.exog_params {
                    v.extend(ep.iter());
                }
                v
            },
            std_errors: self.std_errors.to_vec(),
            t_values: self.t_values.to_vec(),
            p_values: self.p_values.to_vec(),
            conf_lower: self.conf_lower.to_vec(),
            conf_upper: self.conf_upper.to_vec(),
            extra_stats: vec![
                ("Sigma2".to_string(), format!("{:.4}", self.sigma2)),
                (
                    "Log-Likelihood".to_string(),
                    format!("{:.4}", self.log_likelihood),
                ),
                ("AIC".to_string(), format!("{:.4}", self.aic)),
                ("BIC".to_string(), format!("{:.4}", self.bic)),
            ],
        }
    }
}

use crate::garch::GarchResult;

impl ExportableResult for GarchResult {
    fn export_data(&self) -> ExportData {
        ExportData {
            model_name: format!("{:?}({},{})", self.model_type, self.p, self.q),
            dep_var: "volatility".to_string(),
            n_obs: self.n_obs,
            param_names: self.variable_names.clone(),
            coefficients: self.params.to_vec(),
            std_errors: self.std_errors.to_vec(),
            t_values: self.z_values.to_vec(),
            p_values: self.p_values.to_vec(),
            conf_lower: self.conf_lower.to_vec(),
            conf_upper: self.conf_upper.to_vec(),
            extra_stats: vec![
                (
                    "Log-Likelihood".to_string(),
                    format!("{:.4}", self.log_likelihood),
                ),
                ("AIC".to_string(), format!("{:.4}", self.aic)),
                ("BIC".to_string(), format!("{:.4}", self.bic)),
                ("Converged".to_string(), format!("{}", self.converged)),
                ("Iterations".to_string(), format!("{}", self.n_iter)),
            ],
        }
    }
}
