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
