# Greeners: High-Performance Econometrics in Rust

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Version](https://img.shields.io/badge/version-0.1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Greeners** is a lightning-fast, type-safe econometrics library written in pure Rust. It provides a comprehensive suite of estimators for Cross-Sectional, Time-Series, and Panel Data analysis, leveraging linear algebra backends (LAPACK/BLAS) for maximum performance.

Designed for academic research, heavy simulations, and production-grade economic modeling.

## 🚀 Features

### Cross-Sectional & General
- **OLS & GLS:** Robust standard errors (White, Newey-West).
- **IV / 2SLS:** Instrumental Variables for endogeneity correction.
- **Quantile Regression:** Robust estimation via Iteratively Reweighted Least Squares (IRLS).
- **Discrete Choice:** Logit and Probit models (Newton-Raphson MLE).
- **Diagnostics:** R-squared, F-Test, T-Test, Confidence Intervals.

### Time Series (Macroeconometrics)
- **Unit Root Tests:** Augmented Dickey-Fuller (ADF).
- **VAR (Vector Autoregression):** Multivariate modeling with Information Criteria (AIC/BIC).
- **VARMA:** Hannan-Rissanen algorithm for ARMA structures.
- **VECM (Cointegration):** Johansen Procedure (Eigenvalue decomposition) for I(1) systems.
- **Impulse Response Functions (IRF):** Orthogonalized structural shocks.

### Panel Data
- **Fixed Effects (Within):** Absorbs individual heterogeneity.
- **Random Effects:** Swamy-Arora GLS estimator.
- **Between Estimator:** Long-run cross-sectional relationships.
- **Dynamic Panel:** Arellano-Bond (Difference GMM) to solve Nickell Bias.
- **Panel Threshold:** Hansen (1999) non-linear regime switching models.
- **Testing:** Hausman Test for FE vs RE.

### Systems of Equations
- **SUR:** Seemingly Unrelated Regressions (Zellner).
- **3SLS:** Three-Stage Least Squares (System IV).

### System Requirements (Pre-requisites)

## Debian / Ubuntu / Pop!_OS:

sudo apt-get update
sudo apt-get install gfortran libopenblas-dev liblapack-dev pkg-config build-essential

## Fedora / RHEL / CentOS:

sudo dnf install gcc-gfortran openblas-devel lapack-devel pkg-config

## Arch Linux / Manjaro:

sudo pacman -S gcc-fortran openblas lapack base-devel

## macOS:

brew install openblas lapack


## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
greeners = "0.1.0"
ndarray = "0.15"
# Note: You must have a BLAS/LAPACK provider installed on your system
ndarray-linalg = { version = "0.14", features = ["openblas"] }