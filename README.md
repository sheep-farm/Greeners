# Greeners 🦀

[![Crates.io](https://img.shields.io/crates/v/greeners.svg)](https://crates.io/crates/greeners)
[![Docs.rs](https://docs.rs/greeners/badge.svg)](https://docs.rs/greeners)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

**Greeners** is a high-performance, native Rust library for rigorous econometric analysis. 

It provides a comprehensive suite of estimators for linear, non-linear, and structural models, prioritizing numerical stability, type safety, and correctness. Unlike wrappers around R or Python, Greeners implements estimators from scratch using the `ndarray` ecosystem and LAPACK backends.

## ✨ Features

* **Linear Models:**
    * Ordinary Least Squares (OLS)
    * Instrumental Variables (IV / 2SLS)
    * Panel Data Fixed Effects (Within Estimator)
* **Robust Inference:**
    * Heteroskedasticity Consistent (HC1 / White)
    * Heteroskedasticity and Autocorrelation Consistent (HAC / Newey-West)
* **Discrete Choice (MLE):**
    * Logit (Logistic Regression via Newton-Raphson)
    * Probit (Normal CDF via Newton-Raphson)
* **Structural Models:**
    * Generalized Method of Moments (GMM) - Two-Step Efficient Estimator
    * Hansen's J-Test for Overidentification
* **Causal Inference:**
    * Difference-in-Differences (Canonical 2x2 Design)
* **Time Series Diagnostics:**
    * Augmented Dickey-Fuller (ADF) Unit Root Test
    * Durbin-Watson Statistic

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
greeners = "0.1.0"
ndarray = "0.15" # Greeners uses ndarray types for input