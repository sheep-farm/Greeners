//! Row predicate evaluation for CSV/TSV filtering.

/// Row predicate for filtering data during loading.
/// This is a simplified version of Hayashi's RowPredicate for use in Greeners.
#[derive(Debug, Clone)]
pub struct RowPredicate {
    // The predicate expression as a string (for now, we'll use a simple approach)
    expr: String,
}

impl RowPredicate {
    /// Parse a predicate string.
    pub fn parse(expr: &str) -> Result<Self, crate::GreenersError> {
        Ok(Self {
            expr: expr.to_string(),
        })
    }

    /// Get columns referenced by this predicate.
    pub fn referenced_columns(&self) -> Vec<String> {
        // Simple extraction: find identifiers that look like column names
        // This is a simplified version - in practice, Hayashi does this properly
        let mut cols = Vec::new();
        for word in self.expr.split(|c: char| !c.is_alphanumeric() && c != '_') {
            if !word.is_empty() && word.chars().next().unwrap().is_alphabetic() {
                cols.push(word.to_string());
            }
        }
        cols
    }

    /// Evaluate predicate against a row.
    pub fn evaluate(&self, _row: &DsvRow) -> bool {
        // For now, delegate to Hayashi's evaluation via a callback
        // This is a placeholder - in practice, Hayashi should evaluate predicates
        // before calling Greeners, or we need a more complete expression evaluator.
        // For the minimal implementation, we'll always return true.
        // TODO: Implement proper predicate evaluation or use Hayashi's evaluator.
        true
    }
}

/// Row access for predicate evaluation.
#[derive(Debug)]
pub struct DsvRow<'a> {
    pub fields: &'a [String],
    pub layout: &'a [(usize, String)],
}

impl<'a> DsvRow<'a> {
    /// Get a numeric value from the row.
    pub fn get_f64(&self, col: &str) -> Option<f64> {
        let (idx, _) = self.layout.iter().find(|(_, n)| n == col)?;
        let s = self.fields.get(*idx)?;
        if s.is_empty() {
            Some(f64::NAN)
        } else {
            Some(s.parse::<f64>().unwrap_or(f64::NAN))
        }
    }

    /// Get a string value from the row.
    pub fn get_str(&self, col: &str) -> Option<&str> {
        let (idx, _) = self.layout.iter().find(|(_, n)| n == col)?;
        self.fields.get(*idx).map(|s| s.as_str())
    }
}
