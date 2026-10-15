"""Regenerate the frozen STL fixtures with statsmodels 0.14.6 only."""

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
import statsmodels
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stl import _stl


ROOT = Path(__file__).resolve().parent
COUNTS = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
]
COUNT_HASH = "8c999fa9d67e1dff475b9d0d82996f06cc5b7a2598ab359814fbb1c2f4644afd"
CASES = [
    ("outliers_even", 96, 12, 7, 23, {47: 2.5, 79: -1.75}),
    ("unequal_odd", 53, 7, 9, 15, {17: 1.2}),
    ("shortest", 8, 4, 7, 9, {3: 0.4}),
    ("wide_spans", 29, 5, 31, 41, {12: -0.8}),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if statsmodels.__version__ != "0.14.6":
        raise RuntimeError("Fixture oracle must be statsmodels 0.14.6")
    assert sha("".join(f"{v}\n" for v in COUNTS).encode()) == COUNT_HASH
    inputs = [("airpassengers", [math.log(v) for v in COUNTS], 12, 7, 23, {})]
    for name, n, p, sw, tw, outliers in CASES:
        y = [2 + 0.02*i + 0.0003*i*i + 0.3*math.sin(2*math.pi*i/p)
             + 0.08*math.cos(4*math.pi*i/p) + 0.03*math.sin(0.73*i)
             for i in range(n)]
        for i, amount in outliers.items():
            y[i] += amount
        inputs.append((name, y, p, sw, tw, outliers))
    manifest = {
        "reference": "statsmodels 0.14.6",
        "source_url": "https://github.com/statsmodels/statsmodels/blob/v0.14.6/statsmodels/tsa/stl/_stl.pyx",
        "source_sha256": "9b24f6fc0aff80bae0b93712853cc1499b99c5c0722be84c387ad0c3b4088c15",
        "licence_sha256": "1ca78e1dec9dcebc55f3b96a862317f23e76422c5fc943568d955aad1f6b5fad",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform(),
                        "reference_binary_sha256": sha(Path(_stl.__file__).read_bytes())},
        "airpassengers_count_sha256": COUNT_HASH,
        "airpassengers_origin": "R datasets::AirPassengers, monthly airline passengers 1949-1960; counts retained from Hayashi validation/cases/stl_airpassengers/data/data.csv",
        "synthetic_formula": "2+0.02*i+0.0003*i*i+0.3*sin(2*pi*i/p)+0.08*cos(4*pi*i/p)+0.03*sin(0.73*i)",
        "component_atol": 1e-8, "component_rtol": 0, "reconstruction_atol": 1e-12,
        "cases": [],
    }
    for name, y, p, sw, tw, outliers in inputs:
        settings = dict(period=p, seasonal=sw, trend=tw, low_pass=tw,
                        seasonal_deg=1, trend_deg=1, low_pass_deg=1,
                        seasonal_jump=1, trend_jump=1, low_pass_jump=1, robust=True)
        fit = STL(np.asarray(y), **settings).fit(inner_iter=2, outer_iter=1)
        assert np.isfinite(np.concatenate([fit.trend, fit.seasonal, fit.resid, fit.weights])).all()
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["index", "observed", "trend", "seasonal", "remainder", "refit_weight"])
        for i in range(len(y)):
            writer.writerow([i] + [format(v, ".17g") for v in
                                   (y[i], fit.trend[i], fit.seasonal[i], fit.resid[i], fit.weights[i])])
        data = stream.getvalue().encode()
        (ROOT / f"{name}.csv").write_bytes(data)
        manifest["cases"].append(dict(name=name, n=len(y), settings=settings,
                                      inner_iter=2, outer_iter=1, outliers=outliers,
                                      csv_sha256=sha(data),
                                      downweighted_count=int(np.sum(fit.weights < 0.99)),
                                      minimum_refit_weight=float(np.min(fit.weights))))
    assert any(case["downweighted_count"] > 0 for case in manifest["cases"]
               if case["name"] != "airpassengers"), "Synthetic cases must exercise robust refitting"
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
