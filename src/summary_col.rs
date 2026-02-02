use ndarray::Array1;
use std::fmt;

/// A model summary for side-by-side comparison.
#[derive(Debug, Clone)]
pub struct ModelSummary {
    pub name: String,
    pub params: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub p_values: Vec<f64>,
    pub variable_names: Vec<String>,
    pub n_obs: usize,
    pub r_squared: Option<f64>,
    pub adj_r_squared: Option<f64>,
    pub aic: Option<f64>,
    pub bic: Option<f64>,
    pub log_likelihood: Option<f64>,
    pub f_statistic: Option<f64>,
}

impl ModelSummary {
    /// Create from OlsResult-like data.
    pub fn new(name: &str) -> Self {
        ModelSummary {
            name: name.to_string(),
            params: Vec::new(),
            std_errors: Vec::new(),
            p_values: Vec::new(),
            variable_names: Vec::new(),
            n_obs: 0,
            r_squared: None,
            adj_r_squared: None,
            aic: None,
            bic: None,
            log_likelihood: None,
            f_statistic: None,
        }
    }

    pub fn with_coefficients(
        mut self,
        params: &Array1<f64>,
        std_errors: &Array1<f64>,
        p_values: &Array1<f64>,
        variable_names: &[String],
    ) -> Self {
        self.params = params.to_vec();
        self.std_errors = std_errors.to_vec();
        self.p_values = p_values.to_vec();
        self.variable_names = variable_names.to_vec();
        self
    }

    pub fn with_fit_stats(
        mut self,
        n_obs: usize,
        r_squared: Option<f64>,
        adj_r_squared: Option<f64>,
        aic: Option<f64>,
        bic: Option<f64>,
        log_likelihood: Option<f64>,
    ) -> Self {
        self.n_obs = n_obs;
        self.r_squared = r_squared;
        self.adj_r_squared = adj_r_squared;
        self.aic = aic;
        self.bic = bic;
        self.log_likelihood = log_likelihood;
        self
    }

    fn significance_stars(p: f64) -> &'static str {
        if p < 0.001 {
            "***"
        } else if p < 0.01 {
            "**"
        } else if p < 0.05 {
            "*"
        } else if p < 0.1 {
            "."
        } else {
            ""
        }
    }
}

/// Side-by-side model comparison table (like Stata's esttab or R's stargazer).
pub struct SummaryCol;

impl SummaryCol {
    /// Generate a side-by-side comparison of multiple models.
    pub fn compare(models: &[ModelSummary]) -> SummaryColResult {
        // Collect all unique variable names in order of appearance
        let mut all_vars: Vec<String> = Vec::new();
        for m in models {
            for v in &m.variable_names {
                if !all_vars.contains(v) {
                    all_vars.push(v.clone());
                }
            }
        }

        SummaryColResult {
            models: models.to_vec(),
            all_vars,
        }
    }
}

/// Result of summary_col comparison, can be displayed or exported.
#[derive(Debug)]
pub struct SummaryColResult {
    pub models: Vec<ModelSummary>,
    pub all_vars: Vec<String>,
}

impl SummaryColResult {
    /// Export as LaTeX table.
    pub fn to_latex(&self) -> String {
        let n = self.models.len();
        let mut s = String::new();

        s.push_str("\\begin{table}[htbp]\n\\centering\n");
        s.push_str("\\begin{tabular}{l");
        for _ in 0..n {
            s.push('c');
        }
        s.push_str("}\n\\hline\\hline\n");

        // Header
        s.push_str("  ");
        for m in &self.models {
            s.push_str(&format!(" & {}", m.name));
        }
        s.push_str(" \\\\\n\\hline\n");

        // Coefficients
        for var in &self.all_vars {
            // Coefficient row
            s.push_str(var);
            for m in &self.models {
                if let Some(idx) = m.variable_names.iter().position(|v| v == var) {
                    let stars = ModelSummary::significance_stars(m.p_values[idx]);
                    s.push_str(&format!(" & {:.4}{}", m.params[idx], stars));
                } else {
                    s.push_str(" & ");
                }
            }
            s.push_str(" \\\\\n");

            // SE row
            s.push_str("  ");
            for m in &self.models {
                if let Some(idx) = m.variable_names.iter().position(|v| v == var) {
                    s.push_str(&format!(" & ({:.4})", m.std_errors[idx]));
                } else {
                    s.push_str(" & ");
                }
            }
            s.push_str(" \\\\\n");
        }

        s.push_str("\\hline\n");

        // Fit statistics
        s.push('N');
        for m in &self.models {
            s.push_str(&format!(" & {}", m.n_obs));
        }
        s.push_str(" \\\\\n");

        if self.models.iter().any(|m| m.r_squared.is_some()) {
            s.push_str("R$^2$");
            for m in &self.models {
                match m.r_squared {
                    Some(r) => s.push_str(&format!(" & {:.4}", r)),
                    None => s.push_str(" & "),
                }
            }
            s.push_str(" \\\\\n");
        }

        if self.models.iter().any(|m| m.adj_r_squared.is_some()) {
            s.push_str("Adj. R$^2$");
            for m in &self.models {
                match m.adj_r_squared {
                    Some(r) => s.push_str(&format!(" & {:.4}", r)),
                    None => s.push_str(" & "),
                }
            }
            s.push_str(" \\\\\n");
        }

        if self.models.iter().any(|m| m.aic.is_some()) {
            s.push_str("AIC");
            for m in &self.models {
                match m.aic {
                    Some(a) => s.push_str(&format!(" & {:.2}", a)),
                    None => s.push_str(" & "),
                }
            }
            s.push_str(" \\\\\n");
        }

        s.push_str("\\hline\\hline\n");
        s.push_str("\\end{tabular}\n");
        s.push_str("\\caption{Regression Results}\n");
        s.push_str("\\end{table}\n");
        s
    }

    /// Export as HTML table.
    pub fn to_html(&self) -> String {
        let mut s = String::new();
        s.push_str("<table class=\"regression-table\">\n<thead>\n<tr>\n<th></th>\n");
        for m in &self.models {
            s.push_str(&format!("<th>{}</th>\n", m.name));
        }
        s.push_str("</tr>\n</thead>\n<tbody>\n");

        for var in &self.all_vars {
            s.push_str(&format!("<tr>\n<td>{}</td>\n", var));
            for m in &self.models {
                if let Some(idx) = m.variable_names.iter().position(|v| v == var) {
                    let stars = ModelSummary::significance_stars(m.p_values[idx]);
                    s.push_str(&format!(
                        "<td>{:.4}{}<br><small>({:.4})</small></td>\n",
                        m.params[idx], stars, m.std_errors[idx]
                    ));
                } else {
                    s.push_str("<td></td>\n");
                }
            }
            s.push_str("</tr>\n");
        }

        // Fit stats
        s.push_str("<tr class=\"fit-stats\">\n<td>N</td>\n");
        for m in &self.models {
            s.push_str(&format!("<td>{}</td>\n", m.n_obs));
        }
        s.push_str("</tr>\n");

        if self.models.iter().any(|m| m.r_squared.is_some()) {
            s.push_str("<tr>\n<td>R²</td>\n");
            for m in &self.models {
                match m.r_squared {
                    Some(r) => s.push_str(&format!("<td>{:.4}</td>\n", r)),
                    None => s.push_str("<td></td>\n"),
                }
            }
            s.push_str("</tr>\n");
        }

        s.push_str("</tbody>\n</table>\n");
        s
    }

    /// Export as CSV.
    pub fn to_csv(&self) -> String {
        let mut s = String::new();

        // Header
        s.push_str("Variable");
        for m in &self.models {
            s.push_str(&format!(",{} (coef),{} (se)", m.name, m.name));
        }
        s.push('\n');

        // Coefficients
        for var in &self.all_vars {
            s.push_str(var);
            for m in &self.models {
                if let Some(idx) = m.variable_names.iter().position(|v| v == var) {
                    s.push_str(&format!(",{:.6},{:.6}", m.params[idx], m.std_errors[idx]));
                } else {
                    s.push_str(",,");
                }
            }
            s.push('\n');
        }

        // Fit stats
        s.push('N');
        for m in &self.models {
            s.push_str(&format!(",{},", m.n_obs));
        }
        s.push('\n');

        if self.models.iter().any(|m| m.r_squared.is_some()) {
            s.push_str("R-squared");
            for m in &self.models {
                match m.r_squared {
                    Some(r) => s.push_str(&format!(",{:.6},", r)),
                    None => s.push_str(",,"),
                }
            }
            s.push('\n');
        }

        s
    }
}

impl fmt::Display for SummaryColResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.models.len();
        let var_width = 14;
        let col_width = 16;
        let total_width = var_width + 3 + col_width * n;

        writeln!(f, "\n{:=^w$}", " Regression Comparison ", w = total_width)?;

        // Header
        write!(f, "{:<w$} |", "", w = var_width)?;
        for m in &self.models {
            write!(f, " {:^w$}", m.name, w = col_width - 1)?;
        }
        writeln!(f)?;
        writeln!(f, "{:-^w$}", "", w = total_width)?;

        // Coefficients
        for var in &self.all_vars {
            // Coefficient row
            write!(f, "{:<w$} |", var, w = var_width)?;
            for m in &self.models {
                if let Some(idx) = m.variable_names.iter().position(|v| v == var) {
                    let stars = ModelSummary::significance_stars(m.p_values[idx]);
                    write!(
                        f,
                        " {:>w$.4}{}",
                        m.params[idx],
                        stars,
                        w = col_width - 1 - stars.len()
                    )?;
                } else {
                    write!(f, " {:>w$}", "", w = col_width - 1)?;
                }
            }
            writeln!(f)?;

            // SE row
            write!(f, "{:<w$} |", "", w = var_width)?;
            for m in &self.models {
                if let Some(idx) = m.variable_names.iter().position(|v| v == var) {
                    let se_str = format!("({:.4})", m.std_errors[idx]);
                    write!(f, " {:>w$}", se_str, w = col_width - 1)?;
                } else {
                    write!(f, " {:>w$}", "", w = col_width - 1)?;
                }
            }
            writeln!(f)?;
        }

        writeln!(f, "{:-^w$}", "", w = total_width)?;

        // N
        write!(f, "{:<w$} |", "N", w = var_width)?;
        for m in &self.models {
            write!(f, " {:>w$}", m.n_obs, w = col_width - 1)?;
        }
        writeln!(f)?;

        // R²
        if self.models.iter().any(|m| m.r_squared.is_some()) {
            write!(f, "{:<w$} |", "R-squared", w = var_width)?;
            for m in &self.models {
                match m.r_squared {
                    Some(r) => write!(f, " {:>w$.4}", r, w = col_width - 1)?,
                    None => write!(f, " {:>w$}", "", w = col_width - 1)?,
                }
            }
            writeln!(f)?;
        }

        // Adj R²
        if self.models.iter().any(|m| m.adj_r_squared.is_some()) {
            write!(f, "{:<w$} |", "Adj. R-squared", w = var_width)?;
            for m in &self.models {
                match m.adj_r_squared {
                    Some(r) => write!(f, " {:>w$.4}", r, w = col_width - 1)?,
                    None => write!(f, " {:>w$}", "", w = col_width - 1)?,
                }
            }
            writeln!(f)?;
        }

        // AIC
        if self.models.iter().any(|m| m.aic.is_some()) {
            write!(f, "{:<w$} |", "AIC", w = var_width)?;
            for m in &self.models {
                match m.aic {
                    Some(a) => write!(f, " {:>w$.2}", a, w = col_width - 1)?,
                    None => write!(f, " {:>w$}", "", w = col_width - 1)?,
                }
            }
            writeln!(f)?;
        }

        // BIC
        if self.models.iter().any(|m| m.bic.is_some()) {
            write!(f, "{:<w$} |", "BIC", w = var_width)?;
            for m in &self.models {
                match m.bic {
                    Some(b) => write!(f, " {:>w$.2}", b, w = col_width - 1)?,
                    None => write!(f, " {:>w$}", "", w = col_width - 1)?,
                }
            }
            writeln!(f)?;
        }

        writeln!(f, "{:=^w$}", "", w = total_width)?;
        writeln!(f, "Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")
    }
}
