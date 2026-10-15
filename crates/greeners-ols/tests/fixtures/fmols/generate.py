"""Freeze native arch/cointReg diagnostics; never calls Greeners.

Run with numpy2.3.5, pandas2.3.3, arch8.0.0 and R/cointReg0.2.0.
The arch private joint-covariance API is deliberately version-pinned.
"""

import hashlib
import json
import platform
import subprocess
import tempfile
from pathlib import Path

import arch
import numpy as np
import pandas as pd
import scipy
from arch.unitroot.cointegration import FullyModifiedOLS
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent


def datasets():
    n = 300
    rng = np.random.RandomState(42)
    x = rng.normal(0, .1, n).cumsum()
    frames = {"iid": pd.DataFrame({"x": x, "y": .5*x + rng.normal(0, .5, n)})}
    for name, seed in (("endogeneity", 314159), ("serial_correlation", 271828)):
        rng = np.random.default_rng(seed)
        dx, e = rng.normal(size=n), rng.normal(size=n)
        x = dx.cumsum()
        u = .75*dx + e
        if name == "serial_correlation":
            u = np.empty(n)
            u[0] = e[0]/np.sqrt(1-.7**2)
            for t in range(1, n):
                u[t] = .7*u[t-1] + e[t]
        frames[name] = pd.DataFrame({"x": x, "y": .25 + .5*x + u})
    rng = np.random.default_rng(20261014)
    dx = rng.normal(size=(400, 2))
    e = rng.normal(size=400)
    u = np.empty(400)
    u[0] = e[0]
    for t in range(1, 400):
        u[t] = .4*u[t-1] + .5*dx[t, 0] - .3*dx[t, 1] + e[t]
    x = dx.cumsum(axis=0)
    frames["multivariate"] = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1],
                                           "y": .25 + x @ np.array([.5, -.75]) + u})
    return frames


def main():
    assert (np.__version__, pd.__version__, arch.__version__) == ("2.3.5", "2.3.3", "8.0.0")
    results = []
    for name, frame in datasets().items():
        path = ROOT / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.17g")
        df = pd.read_csv(path, float_precision="round_trip")
        names = [key for key in df.columns if key != "y"]
        n = len(df)-1
        bandwidth = max(1, int(4*(len(df)/100)**(2/9)))
        model = FullyModifiedOLS(df.y, df[names], trend="c")
        fit = model.fit(kernel="bartlett", bandwidth=bandwidth, force_int=True,
                        diff=False, df_adjust=False)
        kernel, eta, _ = model._common_fit("bartlett", bandwidth, True, False)
        assert eta.shape == (n, len(names)+1)
        omega = np.asarray(kernel.cov.long_run)
        lam = np.asarray(kernel.cov.one_sided)
        c = np.linalg.solve(omega[1:, 1:], omega[1:, 0])
        bias = lam[0, 1:] - c @ lam[1:, 1:]
        z = np.column_stack([np.ones(n), df[names].to_numpy()[1:]])
        params = fit.params[["const", *names]].to_numpy()
        predicted_r = params - np.linalg.solve(z.T@z, np.r_[0, bias])
        # R remains a native estimator; this transformation is a separate check.
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "reference.R"
            script.write_text('''args <- commandArgs(trailingOnly=TRUE)
stopifnot(as.character(packageVersion("cointReg")) == "0.2.0")
d <- read.csv(args[1]); x <- as.matrix(d[setdiff(names(d), "y")])
m <- cointReg::cointRegFM(x,d$y,deter=matrix(1,nrow(d),1),kernel="ba",
                         bandwidth=as.integer(args[2])+1L,demeaning=FALSE)
cat(jsonlite::toJSON(list(params=unname(m$theta),se=unname(m$sd.theta),
                         R=as.character(getRversion())),digits=NA,auto_unbox=TRUE))
''', encoding="utf-8")
            result = subprocess.run(["Rscript", "--vanilla", str(script), str(path), str(bandwidth)],
                                    check=True, text=True, capture_output=True)
        native_r = json.loads(result.stdout)
        se = fit.std_errors[["const", *names]].to_numpy()
        np.testing.assert_allclose(native_r["params"], predicted_r, rtol=0, atol=1e-10)
        np.testing.assert_allclose(native_r["se"], se, rtol=0, atol=1e-10)
        results.append({"name": name, "N": len(df), "n": n, "bandwidth": bandwidth,
                        "names": names, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "params": params.tolist(), "se": se.tolist(), "omega": omega.tolist(),
                        "normal_probabilities": (2*norm.sf(np.abs(params/se))).tolist(),
                        "bias": bias.tolist(), "native_R": native_r,
                        "predicted_R_params": predicted_r.tolist(),
                        "R_prediction_max_error": float(np.max(np.abs(predicted_r-native_r["params"])))})
    output = {"versions": {"Python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "scipy": scipy.__version__, "arch": arch.__version__, "cointReg": "0.2.0"},
              "contract": "intercept; Bartlett; no HAC centring/df adjustment; bias multiplier N-1; Normal inference",
              "sources": ["https://github.com/bashtage/arch/blob/v8.0.0/arch/unitroot/cointegration.py",
                          "https://github.com/cran/cointReg/blob/3874b40d1035c73b436480e8d0d54771163be544/R/cointReg-FM-OLS.R"],
              "cases": results}
    (ROOT / "reference.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print("Frozen four native-reference fixtures; R convention reconciliation passed")


if __name__ == "__main__":
    main()
