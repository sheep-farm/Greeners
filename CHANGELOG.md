# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0] - 2026-08-10

### Added

- Algebraic invariant tests for the remaining estimator families, bringing coverage to 127 families across panel/VAR/Markov-switching, ML/clustering/kernel, time-series, spatial, nonparametric, causal inference, diagnostics, GAM, beta/ordered/MNLogit and many others.
- New `tests/<estimator>_invariants.rs` integration tests (~100 new files).
- `PROOFS.md` documenting the 127-family test/invariant mapping.
- `AGENTS.md` with local verification commands for contributors.
- `deny.toml` for `cargo deny` checks.
- `rust-version` set to `1.71` in `Cargo.toml`.
- `CHANGELOG.md`.

### Changed

- Bumped version to `1.6.0`.
- Replaced `ndarray-rand` with `rand_distr` and removed the duplicate `ndarray 0.15.x` from the dependency graph.
- Updated `ROADMAP.md` to mark recently implemented estimators as complete (`DynamicFactor`, `UnobservedComponents`, `MarkovAutoregression`, `MSTL`, `MultipleTests`, `Proportion`).
- Updated dependency lockfile, pulling in patched `crossbeam-epoch`, `rand` and other transitive crates.

### Fixed

- All `cargo doc` warnings (broken intra-doc links and unclosed HTML tags).
- `cargo deny` now passes with an explicit license allowlist and an ignored, documented, unmaintained advisory for the transitive `paste` crate.

[1.6.0]: https://github.com/sheep-farm/Greeners/compare/v1.5.3...v1.6.0
