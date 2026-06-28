# Workspace Rules

## Git Workflow
- **Development Branches**: All code modifications, bugfixes, and feature development must be performed strictly on development branches (`dev` for Hayashi, `develop` for Greeners).
- **Production Branches**: Do NOT merge, cherry-pick, commit, or push to production branches (`master` / `main`) unless explicitly requested by the user. Keep production branches strictly for formal releases.

## Project State (Memory)
- **Current State**: 
  - Hayashi version is `0.2.6-dev` on branch `dev` (version `0.2.5` published to crates.io).
  - Greeners version is `1.4.10-dev` on branch `develop` (version `1.4.9` published to crates.io).
  - All tests (544 in Hayashi, 94 in Greeners) pass with 100% success.
- **Last Fixes**:
  - Implemented dynamic type conversion to float (`to_float()`) for non-numeric (boolean, categorical) columns in Hayashi interpreter (`eval_col_expr` and `get_col_f64`).
  - Corrected EGARCH/GJRGARCH function signatures in Portuguese and English quick reference appendices.
- **Where to Resume**:
  - Continue audits and improvements on the `dev` branch of Hayashi and `develop` branch of Greeners.
  - Review remaining estimators or implement new user requests on dev branches.

