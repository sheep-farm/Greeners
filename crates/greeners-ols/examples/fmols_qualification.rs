//! Execute the real FMOLS estimator on externally generated Monte Carlo draws.
use greeners_ols::fmols::FMOLS;
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::io::{self, BufRead, Write};

#[derive(Deserialize)]
struct Draw {
    scenario: String,
    replication: usize,
    x: Vec<f64>,
    y: Vec<f64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let draw: Draw = serde_json::from_str(&line?)?;
        let x = Array2::from_shape_vec((draw.x.len(), 1), draw.x)?;
        let result = match FMOLS::fit(&Array1::from(draw.y), &x, None) {
            Ok(fit) => serde_json::json!({"scenario": draw.scenario,
                "replication": draw.replication, "estimate": fit.beta[0],
                "se": fit.beta_se[0], "n_obs": fit.n_obs}),
            Err(error) => serde_json::json!({"scenario": draw.scenario,
                "replication": draw.replication, "error": error.to_string()}),
        };
        serde_json::to_writer(&mut output, &result)?;
        writeln!(output)?;
    }
    output.flush()?;
    Ok(())
}
