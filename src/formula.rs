use crate::GreenersError;

/// Represents a parsed formula in the form "y ~ x1 + x2 + ... + xn"
#[derive(Debug, Clone)]
pub struct Formula {
    /// Name of the dependent variable (left-hand side)
    pub dependent: String,
    /// Names of independent variables (right-hand side)
    pub independents: Vec<String>,
    /// Whether to include an intercept (default: true)
    pub intercept: bool,
}

impl Formula {
    /// Parse a formula string in the R/Python style: "y ~ x1 + x2 + x3"
    ///
    /// # Syntax
    /// - Basic: "y ~ x1 + x2 + x3" (with intercept)
    /// - No intercept: "y ~ x1 + x2 + x3 - 1" or "y ~ 0 + x1 + x2"
    /// - Intercept only: "y ~ 1"
    ///
    /// # Examples
    /// ```
    /// use greeners::formula::Formula;
    ///
    /// let f = Formula::parse("fte ~ tratado + t + effect").unwrap();
    /// assert_eq!(f.dependent, "fte");
    /// assert_eq!(f.independents, vec!["tratado", "t", "effect"]);
    /// assert_eq!(f.intercept, true);
    ///
    /// let f2 = Formula::parse("y ~ x1 + x2 - 1").unwrap();
    /// assert_eq!(f2.intercept, false);
    /// ```
    pub fn parse(formula: &str) -> Result<Self, GreenersError> {
        let formula = formula.trim();

        // Split by ~ to get LHS and RHS
        let parts: Vec<&str> = formula.split('~').collect();
        if parts.len() != 2 {
            return Err(GreenersError::FormulaError(
                format!("Invalid formula syntax. Expected 'y ~ x1 + x2', got: '{}'", formula)
            ));
        }

        let dependent = parts[0].trim().to_string();
        if dependent.is_empty() {
            return Err(GreenersError::FormulaError(
                "Dependent variable (LHS) cannot be empty".into()
            ));
        }

        let rhs = parts[1].trim();

        // First, handle "- 1" or "- intercept" by removing it from the string
        // Replace "- 1" or "-1" patterns before splitting
        let rhs_clean = rhs.replace("- 1", "").replace("-1", "");

        // Parse RHS: split by + and handle special cases
        let mut independents = Vec::new();
        let mut intercept = true;

        // Check if the original had "- 1" to disable intercept
        if rhs.contains("- 1") || rhs.contains("-1") {
            intercept = false;
        }

        // Split by + and process each term
        for term in rhs_clean.split('+') {
            let term = term.trim();

            if term.is_empty() {
                continue;
            }

            // Check for intercept control
            if term == "1" {
                // Explicit intercept, already default
                intercept = true;
                continue;
            } else if term == "0" {
                // Remove intercept
                intercept = false;
                continue;
            }

            independents.push(term.to_string());
        }

        let cleaned_independents = independents;

        Ok(Formula {
            dependent,
            independents: cleaned_independents,
            intercept,
        })
    }

    /// Get the total number of columns in the design matrix (including intercept if present)
    pub fn n_cols(&self) -> usize {
        let base = self.independents.len();
        if self.intercept {
            base + 1
        } else {
            base
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_formula() {
        let f = Formula::parse("y ~ x1 + x2 + x3").unwrap();
        assert_eq!(f.dependent, "y");
        assert_eq!(f.independents, vec!["x1", "x2", "x3"]);
        assert!(f.intercept);
        assert_eq!(f.n_cols(), 4); // intercept + 3 vars
    }

    #[test]
    fn test_formula_no_intercept() {
        let f = Formula::parse("y ~ x1 + x2 - 1").unwrap();
        assert_eq!(f.dependent, "y");
        assert_eq!(f.independents, vec!["x1", "x2"]);
        assert!(!f.intercept);
        assert_eq!(f.n_cols(), 2);
    }

    #[test]
    fn test_formula_zero_intercept() {
        let f = Formula::parse("y ~ 0 + x1 + x2").unwrap();
        assert_eq!(f.dependent, "y");
        assert_eq!(f.independents, vec!["x1", "x2"]);
        assert!(!f.intercept);
    }

    #[test]
    fn test_intercept_only() {
        let f = Formula::parse("y ~ 1").unwrap();
        assert_eq!(f.dependent, "y");
        assert_eq!(f.independents.len(), 0);
        assert!(f.intercept);
        assert_eq!(f.n_cols(), 1);
    }

    #[test]
    fn test_invalid_formula() {
        assert!(Formula::parse("invalid").is_err());
        assert!(Formula::parse("~ x1 + x2").is_err());
        assert!(Formula::parse("y ~").is_ok()); // empty RHS is technically ok
    }
}
