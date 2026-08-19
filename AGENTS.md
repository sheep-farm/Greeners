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

- Unit tests live next to the implementation in each crate (`crates/*/src/*.rs`) and run with `cargo test`.
- Algebraic invariant tests for estimators live in `crates/<crate>/tests/*_invariants.rs` and are also run with `cargo test`.
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

## Release policy (2.0.0)

Greeners is released as a workspace of crates on crates.io. The `main` branch
tracks the current stable release (`2.0.0`).

- New estimators, breaking API changes, and dependency/MSRV updates target
  `2.x` releases.
- Patch releases (`2.0.x`) are reserved for bug fixes and security/dependency
  updates that do not break the public API.
- Hayashi consumes `greeners` from crates.io; `Cargo.lock` pins the exact
  version used in each Hayashi build.
