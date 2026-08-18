#!/usr/bin/env python3
"""Check that crates/greeners/src/lib.rs matches the generated curated facade.

Run from the repository root:

    python3 scripts/check_facade.py

The script rebuilds the facade from the public items used in the facade tests
and examples plus a base list of core types. It then compares the result with
the current crates/greeners/src/lib.rs. If they differ, it exits with a
non-zero status and prints a diff.
"""

import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Base set of items that are always re-exported at the facade root, even if
# they are not used directly by the facade tests/examples. Add new names here
# when you want them exposed at the top level.
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
    used = collect_used_names()

    selected = set(BASE)
    selected.update(used)

    root_items = {}
    for name in selected:
        if name in item_map and len(item_map[name]) == 1:
            root_items[name] = item_map[name][0]

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
    for name in sorted(root_items):
        crate, mod = root_items[name]
        cname = crate.replace("-", "_")
        lines.append(f"pub use {cname}::{mod}::{name};")

    return "\n".join(lines) + "\n"


def main():
    generated = generate_facade()
    facade_path = os.path.join(ROOT, "crates", "greeners", "src", "lib.rs")
    with open(facade_path, "r", encoding="utf-8") as f:
        current = f.read()

    if generated == current:
        print("Facade is up to date.")
        return 0

    print("Facade is out of date. Differences:")
    import difflib

    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        generated.splitlines(keepends=True),
        fromfile=facade_path,
        tofile="<generated>",
    )
    sys.stdout.writelines(diff)
    return 1


if __name__ == "__main__":
    sys.exit(main())
