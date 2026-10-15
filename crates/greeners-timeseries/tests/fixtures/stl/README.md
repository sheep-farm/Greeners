# Frozen STL regression fixtures

Run `python generate.py` in an environment with statsmodels **0.14.6**, NumPy
and SciPy. The generator rejects other statsmodels versions. Rust tests read
the checked-in CSV files with the existing csv dependency and need no Python.
`manifest.json` records the generation environment, binary/source/licence
hashes, inputs, complete configuration and CSV hashes. Byte-identical
regeneration is checked in that recorded environment, not promised for arbitrary
numerical stacks. No random seeds, fitting searches or tolerance tuning occur.

All vectors include every endpoint in original order. Column `refit_weight`
records the one weight update used by the robust refit. The synthetic cases
include non-trivial updates; the shortest case happens to produce all-one
weights with the pinned oracle and is retained unchanged.

Contract: local-linear degree 1 and jump 1 throughout, two inner iterations,
one robust refit (`outer_iter=1`). The low-pass window equals the effective
trend window, unlike the reference default. Full-vector comparisons use
atol=1e-8 and rtol=0; reconstruction uses 1e-12. The AirPassengers component
tolerance was historically selected after exploration and was not changed
for this repair. These checks do not establish equivalence to R's robust STL.

AirPassengers contains the canonical 144 monthly passenger counts (1949-1960),
from R's `datasets::AirPassengers`, retained from Hayashi's public STL case.
The generator embeds the counts and verifies the pre-existing canonical hash
of one decimal integer per line before taking natural logarithms. No new data
licence is asserted. Source adaptation attribution and the complete applicable
statsmodels licence ship in this crate's `THIRD_PARTY_NOTICES.md`.

Synthetic formula and the exact lengths, periods, windows and outliers are
fixed in `generate.py` and the manifest. The shortest case has n=2*period;
unequal phase lengths, odd/even periods, seasonal spans beyond phase length
and a trend span beyond n are covered.

Separate analytic controls (not Python robust golden fixtures) use n=48,
period=12, windows=7/23: zeros, constant 2, and 0.25+0.03125*i. Trend equals
input and seasonal/remainder equal zero to 1e-12. Tiny residual arithmetic is
not forced to zero; exact median, zero-scale reset and positive 1e-100-scale
weight controls live in the private-helper Rust tests.

Non-finite inputs, oversized dimensions/windows and non-finite intermediate
arithmetic return an error. Finite inputs across the entire f64 range are not
qualified. The alternating +/-MAX/2 test accepts a finite reconstructing result
or an explicit error; the definite robustness-scale overflow [1e308;5] must
error. Zero support alone uses the observation (interior) or adjacent smoothed
endpoint (extrapolation), never a global mean.
