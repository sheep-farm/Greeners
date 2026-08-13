# AGENTS.md — Project guidance for Greeners

## Local development setup

This repository uses `cargo` for all build / test / verification steps.

## Verification commands

```bash
# Formatting
cargo fmt
cargo fmt --check

# Linting (warnings are errors)
cargo clippy -- -D warnings

# Full test suite (unit tests, integration tests, doc tests)
cargo test

# Run a single integration test file
cargo test --test <file_stem>

# Documentation build (must be warning-free)
cargo doc

# Dependency / license / advisory audit
cargo deny check

# Release build sanity check
cargo build --release
```

## Test layout

- Unit tests live next to the implementation (`src/*.rs`) and run with `cargo test`.
- Algebraic invariant tests for estimators live in `tests/<estimator>_invariants.rs` and are also run with `cargo test`.
- The companion `PROOFS.md` document tracks the mathematical specification and the invariants checked by each test file.

## Hayashi smoke test

Greeners is consumed by the Hayashi interpreter in `../hayashi`. A quick
end-to-end check is:

```bash
cd ../hayashi
cargo test --test smoke
```

The Hayashi smoke suite contains many model-specific smoke tests that
exercise the same estimators through the `hay` language.

## Freeze mode (from 1.6.0)

Greeners is in **multi-year freeze** after the `v1.6.0` release. The goal is
for the numerical engine to remain stable while Hayashi matures on its side.

### What the freeze means

- `main` is release-only. Only critical bug fixes and security/dependency
  updates may be merged through short-lived `hotfix/*` branches and released
  as `1.6.x` patch versions.
- No new estimators, no breaking API changes, and no MSRV changes on the
  `1.6.x` line. This freeze is expected to last for years.
- Work that would require new mathematics in Greeners should be deferred to
  a future `2.0` release, not squeezed into `1.6.x`.

### What Hayashi can still do without unfreezing Greeners

Most remaining Hayashi work lives in the interpreter layer and does not need
Greeners changes:

- Add or fix Hayashi dispatch for estimators that already exist in Greeners
  (e.g. `be` / `BetweenEstimator` is already in `src/panel.rs`).
- Add validation cases, documentation, and smoke tests.
- Extend post-estimation commands that only re-use existing `*Result` fields.

### What would break the freeze

- New algorithms or new estimator families (e.g. non-Cholesky SVAR
  identification, richer structural Kalman models).
- Changes to public `struct`/`Result` signatures consumed by Hayashi.
- Raising or lowering `rust-version`.
- Removing or replacing existing dependencies in a way that changes the public
  API or MSRV.

If a change falls into the second list, open a `2.0` proposal instead of a
`1.6.x` pull request.
