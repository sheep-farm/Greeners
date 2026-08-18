#!/usr/bin/env python3
"""Check that crates/greeners/src/lib.rs matches the generated curated facade.

Run from the repository root:

    python3 scripts/check_facade.py

The script rebuilds the facade from the public items used in the facade tests
and examples plus a base list of core types. If an item is re-exported at the
root of its sub-crate lib.rs, the facade uses `pub use crate::Item;`;
otherwise it falls back to `pub use crate::module::Item;`. The script then
compares the result with the current crates/greeners/src/lib.rs and exits
non-zero if they differ.
"""

import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE = {
    "ARDL", "ARIMA", "ArellanoBond", "AutoReg", "BSplineBasis", "BayesGaussMI",
    "BayesMixedGLM", "BetaLink", "BetaModel", "BetweenEstimator",
    "BinaryDiagnostics", "Bootstrap", "CUSUMTest", "CanCorr", "CategoricalColumn",
    "Column", "ColumnType", "ConditionalLogit", "ConditionalMNLogit",
    "ConditionalPoisson", "CorrStructure", "CovarianceType", "CoxPH", "DataFrame",
    "DataType", "Datasets", "Decomposition", "DescrStatsW", "Diagnostics",
    "DiffInDiff", "Equation", "ExponentialSmoothing", "FGLS", "FactorAnalysis",
    "Family", "FixedEffects", "Formula", "GEE", "GLM", "GLMGam", "GLSAR", "GMM",
    "GarchDist", "GarchModelType", "GenPoisson", "GreenersError", "HausmanTest",
    "HypothesisTest", "IV", "InferenceType", "Influence", "KDEMultivariate",
    "KDEUnivariate", "KalmanFilter", "KalmanSmoother", "KaplanMeier", "Kernel",
    "KernelReg", "Link", "Logit", "Lowess", "MANOVA", "MICE", "MNLogit",
    "MarkovSwitching", "MixedLM", "ModelSelection", "ModelSummary", "NegBin",
    "NegBinP", "NominalGEE", "OLS", "OlsResult", "OrderedLogit", "OrderedProbit",
    "OrdinalGEE", "PCA", "PanelDiagnostics", "PanelThreshold", "Poisson",
    "Probit", "QuantileReg", "RLM", "RandomEffects", "RecursiveLS", "RobustNorm",
    "RollingOLS", "RollingWLS", "Rotation", "SUR", "SVAR", "SVarIdentification",
    "SpecificationTests", "StateSpaceModel", "Stats", "SummaryCol", "SummaryStats",
    "SurEquation", "ThreeSLS", "TimeSeries", "TypeInferenceConfig", "VAR",
    "VARMA", "VECM", "WLS", "ZINB", "ZIP", "state_space_estimate",
}


def build_public_item_map():
    item_map = defaultdict(list)
    for crate in os.listdir(os.path.join(ROOT, "crates")):
        if crate == "greeners":
            continue
        src = os.path.join(ROOT, "crates", crate, "src")
        if not os.path.isdir(src):
            continue
        for f in os.listdir(src):
            if not f.endswith(".rs") or f == "lib.rs":
                continue
            mod = f[:-3]
            p = os.path.join(src, f)
            with open(p, "r", encoding="utf-8", errors="ignore") as fp:
                content = fp.read()
            content = re.sub(r"//.*", "", content)
            content = re.sub(r"/\*.*?\*/", "", content, flags=re.S)
            pattern = r"^pub\s+(?:struct|enum|type|fn|trait)\s+([A-Za-z_][A-Za-z0-9_]*)\b"
            for m in re.finditer(pattern, content, re.M):
                item_map[m.group(1)].append((crate, mod))
    return item_map


def collect_modules():
    mods = {}
    for crate in os.listdir(os.path.join(ROOT, "crates")):
        if crate == "greeners":
            continue
        lib = os.path.join(ROOT, "crates", crate, "src", "lib.rs")
        if not os.path.exists(lib):
            continue
        with open(lib, "r", encoding="utf-8") as f:
            content = f.read()
        mods[crate] = sorted(
            re.findall(r"^pub mod ([a-zA-Z_][a-zA-Z0-9_]*);", content, re.M)
        )
    return mods


def collect_crate_reexports():
    """Map item name -> crate for items re-exported at a sub-crate root."""
    crate_items = {}
    for crate in os.listdir(os.path.join(ROOT, "crates")):
        if crate == "greeners":
            continue
        lib = os.path.join(ROOT, "crates", crate, "src", "lib.rs")
        if not os.path.exists(lib):
            continue
        with open(lib, "r", encoding="utf-8") as f:
            content = f.read()
        # both `pub use module::Item;` and `pub use module::{A, B};`
        pattern_single = r"^pub\s+use\s+[a-zA-Z_][a-zA-Z0-9_]*::([A-Za-z_][A-Za-z0-9_]*)[;,]?\s*$"
        pattern_multi = r"^pub\s+use\s+[a-zA-Z_][a-zA-Z0-9_]*::\{([^}]+)\};"
        for m in re.finditer(pattern_single, content, re.M):
            crate_items[m.group(1)] = crate
        for m in re.finditer(pattern_multi, content, re.M):
            for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", m.group(1)):
                crate_items[name] = crate
    return crate_items


def collect_used_names():
    used = set()
    for sub in ["tests", "examples"]:
        for root, _, files in os.walk(os.path.join(ROOT, "crates", "greeners", sub)):
            for f in files:
                if not f.endswith(".rs"):
                    continue
                p = os.path.join(root, f)
                with open(p, "r", encoding="utf-8", errors="ignore") as fp:
                    content = fp.read()
                for m in re.finditer(r"\b([A-Z][A-Za-z0-9_]*)\b", content):
                    used.add(m.group(1))
    return used


def generate_facade():
    item_map = build_public_item_map()
    mods = collect_modules()
    crate_items = collect_crate_reexports()
    used = collect_used_names()

    selected = set(BASE)
    selected.update(used)

    # Determine where each selected item should be imported from.
    # Prefer crate root re-export; fall back to module path.
    root_items = []
    for name in selected:
        if name not in item_map:
            continue
        if len(item_map[name]) > 1:
            # Ambiguous across crates: only allow if a crate root re-exports it.
            if name in crate_items:
                root_items.append((name, crate_items[name], None))
            continue
        crate, mod = item_map[name][0]
        if name in crate_items and crate_items[name] == crate:
            root_items.append((name, crate, None))
        else:
            root_items.append((name, crate, mod))

    lines = ["//! greeners facade crate.", "", "pub mod export;", ""]
    lines.append("// Re-export all modules from sub-crates.")
    lines.append("")
    for crate in sorted(mods):
        cname = crate.replace("-", "_")
        for mod in mods[crate]:
            lines.append(f"pub use {cname}::{mod};")
    lines.append("")
    lines.append("// Curated root re-exports.")
    lines.append("")
    for name, crate, mod in sorted(root_items, key=lambda x: x[0]):
        cname = crate.replace("-", "_")
        if mod:
            lines.append(f"pub use {cname}::{mod}::{name};")
        else:
            lines.append(f"pub use {cname}::{name};")

    return "\n".join(lines) + "\n"


def reexports(text):
    """Extract the set of re-export statements, ignoring comments and blanks."""
    return set(
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("pub use ") or line.strip().startswith("pub mod ")
    )


def main():
    generated = generate_facade()
    facade_path = os.path.join(ROOT, "crates", "greeners", "src", "lib.rs")

    if len(sys.argv) > 1 and sys.argv[1] == "--fix":
        with open(facade_path, "w", encoding="utf-8") as f:
            f.write(generated)
        print("Facade regenerated. Run `cargo fmt` to format.")
        return 0

    with open(facade_path, "r", encoding="utf-8") as f:
        current = f.read()

    if reexports(generated) == reexports(current):
        print("Facade is up to date.")
        return 0

    print("Facade is out of date. Differences:")
    import difflib

    current_re = sorted(reexports(current))
    generated_re = sorted(reexports(generated))
    diff = difflib.unified_diff(
        [l + "\n" for l in current_re],
        [l + "\n" for l in generated_re],
        fromfile=facade_path,
        tofile="<generated>",
    )
    sys.stdout.writelines(diff)
    return 1


if __name__ == "__main__":
    sys.exit(main())
