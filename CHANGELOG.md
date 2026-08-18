# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0-proposal.1] - Unreleased

### Changed

- Workspace inheritance: centralized `version`, `rust-version`, `edition`, `authors`, `license`, `repository` and common dependency versions in the root `Cargo.toml`.
- Removed the catch-all `greeners-models` crate.
  - `gmm` and `nls` moved to `greeners-ols`.
  - `dfm` moved to `greeners-timeseries`.
  - `export` moved to the `greeners` facade (cross-cutting I/O utility).
- Reorganized the `greeners` facade to re-export all sub-crate modules and their public items, with `pub mod export` for the cross-cutting export module.
- Resolved test-only dependency cycles (e.g. removed `greeners-timeseries` from `greeners-panel` dependencies by splitting the `LocalLevel` test into `greeners-timeseries`).
- Added `ARCHITECTURE.md` documenting crate responsibilities, dependency rules and the facade design.

## [1.6.5-dev] - Unreleased

### Changed

- Portuguese user-facing output strings and comments translated to English.
- Bumped version to `1.6.5-dev`.

## [1.6.3] - 2026-08-17

### Fixed

- Gaussian-process training predictions and variance with observation noise (`greeners#13`).
  Posterior means now use latent covariance `K_f K_y^-1 y` and posterior variances use `diag(K_f - K_f K_y^-1 K_f)`.

## [1.6.0] - 2026-08-10

### Added

- Algebraic invariant tests for the remaining estimator families, bringing coverage to 127 families across panel/VAR/Markov-switching, ML/clustering/kernel, time-series, spatial, nonparametric, causal inference, diagnostics, GAM, beta/ordered/MNLogit and many others.
- New `tests/<estimator>_invariants.rs` integration tests (~100 new files).
- `PROOFS.md` documenting the 127-family test/invariant mapping.
- `AGENTS.md` with local verification commands for contributors.
- `deny.toml` for `cargo deny` checks.
- `CHANGELOG.md`.

### Changed

- Bumped version to `1.6.0`.
- `rust-version` set to `1.84.0` to match the MSRV of `faer 0.22`.
- Replaced `ndarray-rand` with `rand_distr` and removed the duplicate `ndarray 0.15.x` from the dependency graph.
- Reduced duplicate dependencies: downgraded `argmin` to `0.10.0`, `argmin-math` to `0.4`, `thiserror` to `1.0` and pinned `serde` to `1.0.228`. This removes duplicate major versions of `rand`, `thiserror` and the `syn 3.0` branch from the tree.
- Updated `ROADMAP.md` to mark recently implemented estimators as complete (`DynamicFactor`, `UnobservedComponents`, `MarkovAutoregression`, `MSTL`, `MultipleTests`, `Proportion`).
- Updated dependency lockfile, pulling in patched `crossbeam-epoch`, `rand` and other transitive crates.

### Fixed

- All non-documentation `unwrap()`/`expect()` calls in `src/` replaced with proper error handling, eliminating unexpected panic paths before the 1.6.0 release.
- All `cargo doc` warnings (broken intra-doc links and unclosed HTML tags).
- `cargo deny` now passes with an explicit license allowlist and documented ignores for the transitive `paste` and `instant` unmaintained advisories.

[1.6.5-dev]: https://github.com/sheep-farm/Greeners/compare/v1.6.4...develop
[1.6.3]: https://github.com/sheep-farm/Greeners/compare/v1.6.2...v1.6.3
[1.6.0]: https://github.com/sheep-farm/Greeners/compare/v1.5.3...v1.6.0
