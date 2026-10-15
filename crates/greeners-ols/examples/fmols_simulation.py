"""Bounded, failure-aware FMOLS coverage diagnostic using the actual Rust example.

Run: python fmols_simulation.py <fmols_qualification executable> <output.json>
No coverage threshold is a pass gate. Reference agreement and coverage are separate.
"""

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

import arch
import numpy as np
import scipy
from arch.unitroot.cointegration import FullyModifiedOLS
from scipy.stats import norm

SCENARIOS = (
    ("iid_N300", 300, 0.0, 0.0, 2026101401),
    ("iid_N1000", 1000, 0.0, 0.0, 2026101402),
    ("endogenous_AR_N300", 300, 0.5, 0.5, 2026101403),
    ("endogenous_AR_N1000", 1000, 0.5, 0.5, 2026101404),
)
REPLICATIONS = 500
BURN = 300
BETA = 0.5
ALPHA = 0.25
CRITICAL = float(norm.ppf(0.975))


def generate(rng, n, rho, gamma):
    # Within each replication draw all dx, then all independent e, including burn-in.
    dx = rng.normal(size=n + BURN)
    e = rng.normal(size=n + BURN)
    x = dx.cumsum()  # x[-1]=0 before the first innovation.
    u = np.empty(n + BURN)
    previous = 0.0  # u[-1]=0 before the first innovation.
    for t in range(n + BURN):
        previous = rho * previous + gamma * dx[t] + e[t]
        u[t] = previous
    return x[BURN:], (ALPHA + BETA * x + u)[BURN:]


def binomial(successes, attempts):
    if attempts == 0:
        return {"successes": 0, "denominator": 0, "rate": None,
                "mc_standard_error": None, "wilson_95_interval": None}
    p = successes / attempts
    z2 = CRITICAL**2
    denominator = 1 + z2 / attempts
    centre = (p + z2 / (2 * attempts)) / denominator
    half = CRITICAL * np.sqrt(p * (1 - p) / attempts + z2 / (4 * attempts**2)) / denominator
    return {"successes": successes, "denominator": attempts, "rate": p,
            "mc_standard_error": float(np.sqrt(p * (1 - p) / attempts)),
            "wilson_95_interval": [float(centre - half), float(centre + half)]}


def aggregate(results):
    successful = [row for row in results if "error" not in row]
    for row in successful:
        if not np.isfinite(row["estimate"]) or not np.isfinite(row["se"]) or row["se"] <= 0:
            raise ValueError("Rust reported an invalid successful fit")
    estimates = np.array([row["estimate"] for row in successful])
    errors = estimates - BETA
    ses = np.array([row["se"] for row in successful])
    covered = int(np.count_nonzero(np.abs(errors) <= CRITICAL * ses))
    return {"attempted": len(results), "successful": len(successful),
            "failures": len(results) - len(successful),
            "failure_reasons": dict(Counter(row["error"] for row in results if "error" in row)),
            "nominal_coverage": 0.95,
            "coverage_all_attempts_failures_as_noncoverage": binomial(covered, len(results)),
            "coverage_successful_fits": binomial(covered, len(successful)),
            "successful_fit_statistics": {
                "bias": float(errors.mean()) if successful else None,
                "bias_mc_standard_error": float(errors.std(ddof=1) / np.sqrt(len(errors))) if len(errors) > 1 else None,
                "rmse": float(np.sqrt(np.mean(errors**2))) if successful else None,
                "empirical_standard_deviation": float(estimates.std(ddof=1)) if len(errors) > 1 else None,
                "average_se": float(ses.mean()) if successful else None,
                "average_interval_width": float(2 * CRITICAL * ses.mean()) if successful else None}}


def self_check():
    x1, y1 = generate(np.random.default_rng(9), 12, 0.5, 0.5)
    x2, y2 = generate(np.random.default_rng(9), 12, 0.5, 0.5)
    assert x1.shape == y1.shape == (12,)
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)
    assert np.isfinite(x1).all() and np.isfinite(y1).all()
    summary = aggregate([{"estimate": BETA, "se": 0.1}, {"error": "rank"}])
    assert summary["failures"] == 1
    assert summary["coverage_all_attempts_failures_as_noncoverage"]["rate"] == 0.5
    assert summary["coverage_successful_fits"]["rate"] == 1.0
    assert aggregate([])["coverage_successful_fits"]["rate"] is None
    try:
        aggregate([{"estimate": BETA, "se": 0.0}])
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid successful inference was accepted")


def run(executable, output):
    if (np.__version__, scipy.__version__, arch.__version__) != ("2.3.5", "1.15.3", "8.0.0"):
        raise RuntimeError("Use the declared numpy2.3.5/scipy1.15.3/arch8.0.0 reference environment")
    scenarios = []
    for name, n, rho, gamma, seed in SCENARIOS:
        rng = np.random.default_rng(seed)
        inputs, references = [], []
        for replication in range(REPLICATIONS):
            x, y = generate(rng, n, rho, gamma)
            inputs.append({"scenario": name, "replication": replication, "x": x.tolist(), "y": y.tolist()})
            bandwidth = max(1, int(4 * (n / 100)**(2 / 9)))
            native = FullyModifiedOLS(y, x[:, None], trend="c").fit(
                kernel="bartlett", bandwidth=bandwidth, force_int=True, diff=False, df_adjust=False)
            references.append({"estimate": float(native.params.iloc[0]), "se": float(native.std_errors.iloc[0])})
        payload = "".join(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n" for row in inputs)
        process = subprocess.run([str(executable)], input=payload, text=True, capture_output=True, check=True)
        results = [json.loads(line) for line in process.stdout.splitlines()]
        expected = [(name, i) for i in range(REPLICATIONS)]
        actual = [(row.get("scenario"), row.get("replication")) for row in results]
        if actual != expected:
            raise RuntimeError("Rust result ids/count/order do not match the submitted draws")
        for row in results:
            if "error" not in row and row.get("n_obs") != n - 1:
                raise RuntimeError("Rust returned an incorrect FM sample size")
        summary = aggregate(results)
        pairs = [(row, ref) for row, ref in zip(results, references) if "error" not in row]
        estimate_error = max((abs(row["estimate"] - ref["estimate"]) for row, ref in pairs), default=None)
        se_error = max((abs(row["se"] - ref["se"]) for row, ref in pairs), default=None)
        comparisons = [np.allclose([row["estimate"], row["se"]], [ref["estimate"], ref["se"]],
                                  atol=1e-10, rtol=1e-9) for row, ref in pairs]
        scenarios.append({"scenario": name, "N": n, "seed": seed, "rho": rho, "gamma": gamma,
                          "draw_payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                          "rust": summary, "native_arch_coverage_separate_reference": aggregate(references),
                          "native_agreement_separate_from_coverage": {
                              "compared": len(pairs), "agreement_count": sum(bool(v) for v in comparisons),
                              "max_absolute_estimate_error": estimate_error, "max_absolute_se_error": se_error,
                              "atol": 1e-10, "rtol": 1e-9}, "replication_results": results})
        print(f"{name}: {summary['successful']}/{REPLICATIONS} fits; "
              f"coverage={summary['coverage_all_attempts_failures_as_noncoverage']['rate']:.3f}", flush=True)
    source = Path(__file__)
    report = {"scope": "Bounded slope 95% Normal coverage diagnostic; no general coverage certification or pass threshold",
              "versions": {"Python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "arch": arch.__version__},
              "executable": str(executable), "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "dgp": {"alpha": ALPHA, "beta": BETA, "burn": BURN, "replications_per_scenario": REPLICATIONS,
                      "generator": "numpy.default_rng/PCG64; one independent seeded stream per scenario",
                      "draw_order": "For each replication, N+300 standard Normal dx draws then N+300 standard Normal e draws",
                      "initialisation": "x[-1]=u[-1]=0; x=cumsum(dx); u[t]=rho*u[t-1]+gamma*dx[t]+e[t]",
                      "trimming": "Discard indices0..299; retain indices300..N+299 with their random-walk levels",
                      "inference": "estimate +/- scipy.stats.norm.ppf(.975)*SE; nominal95%",
                      "mc_uncertainty": "95% Wilson intervals and plug-in binomial MC standard errors"},
              "scenarios": scenarios}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if not args.self_check:
        if args.executable is None or args.output is None:
            parser.error("executable and output are required")
        run(args.executable.resolve(), args.output.resolve())
