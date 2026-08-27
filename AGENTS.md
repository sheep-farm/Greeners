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

## Release model (from 2.0.0)

The `1.6.x` freeze ended with the 2.0 workspace restructure. Greeners now moves
on its own release train and Hayashi follows it deliberately, so development
speed here never changes Hayashi's numbers by accident.

### Branches

- `develop`: integration branch, where all work lands. Ships as the next minor
  (`2.1.0`, `2.2.0`).
- `main`: release-only. Every release is tagged (`vX.Y.Z`) from its tip.
- `release/2.x`: maintenance line for whatever Hayashi is pinned to. Receives
  cherry-picks only, released as `2.x.Z` patches.

### Versioning rules

- **A change that alters numerical output is a minor, never a patch.** It keeps
  the API intact but breaks goldens and published results downstream, so it must
  never arrive through a `cargo update`. The Christiano-Fitzgerald filter fix is
  the reference case.
- Patches (`2.x.Z`) are for changes that cannot move a coefficient: build fixes,
  advisories, docs. Cherry-pick a numerical fix into a patch only when a
  consumer is actively hurt by the bug, and say so in the CHANGELOG.
- `rust-version` is part of the published contract: raise it in its own PR, with
  the new floor verified (`cargo +<msrv> check --workspace --locked`), and treat
  it as at least a minor. Current MSRV is `1.85.0` — the resolved graph pulls
  `edition2024` manifests, which Cargo 1.84 cannot parse.
- Inside `2.x`, do not change the signatures of the public `*Result` types and
  facade items Hayashi consumes; add fields/items instead. Breaking them is 3.0.

### Consumer contract (Hayashi)

- Hayashi pins the facade exactly (`greeners = "=2.0.0"`) with a committed
  `Cargo.lock` and `--locked` in CI, so nothing here reaches it until someone
  bumps the pin.
- To get a fix into Hayashi without touching its version: release the sub-crate
  patch from `release/2.x` and let Hayashi run
  `cargo update -p greeners-<crate> --precise <version>`.
- For QA against unreleased work, do not spend a version number: patch the
  source in Hayashi's checkout with
  `[patch.crates-io] greeners = { path = "../Greeners/crates/greeners" }` and run
  `cargo test --test smoke`. A `[patch]`-resolved crate cannot be published, so
  this is a testing tool only. Pre-releases (`2.1.0-rc.N`) are the option when
  Hayashi needs to consume unreleased work for longer; they are never selected by
  a caret requirement, and must never appear in a published Hayashi release.

### Reference values belong here

The mathematics lives in Greeners, so the reference vectors (statsmodels/R) live
next to the estimator, in `crates/<crate>/tests/`. Algebraic invariants alone do
not catch "implemented a different algorithm than documented" — that is exactly
how the CF filter bug survived. Hayashi's tests cover the bridge (formula →
design matrix → display), not numerical accuracy.
