"""
Generate reference values from statsmodels/scipy for numerical accuracy tests in Greeners.
Output: tests/reference_values.csv (key-value format)

Requirements: pip install numpy statsmodels linearmodels
"""

import csv
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, Probit
from statsmodels.regression.quantile_regression import QuantReg
from linearmodels.panel import PanelOLS, RandomEffects

np.random.seed(42)

rows = []


def add(key, value):
    rows.append((key, f"{value:.10f}"))


# ============================================================================
# DATASET 1: OLS (n=50, k=3 regressors + intercept)
# ============================================================================
n = 50
x1 = np.random.randn(n)
x2 = np.random.uniform(0, 10, n)
x3 = np.random.exponential(2, n)
eps = np.random.randn(n) * 1.5
y = 3.0 + 1.5 * x1 - 0.8 * x2 + 0.5 * x3 + eps

# Save raw data so Rust tests can reconstruct the exact same dataset
for i in range(n):
    add(f"ols_data.y.{i}", y[i])
    add(f"ols_data.x1.{i}", x1[i])
    add(f"ols_data.x2.{i}", x2[i])
    add(f"ols_data.x3.{i}", x3[i])

X = sm.add_constant(np.column_stack([x1, x2, x3]))

# OLS NonRobust
res = sm.OLS(y, X).fit()
for i, name in enumerate(["const", "x1", "x2", "x3"]):
    add(f"ols_nonrobust.params.{name}", res.params[i])
    add(f"ols_nonrobust.se.{name}", res.bse[i])
    add(f"ols_nonrobust.t.{name}", res.tvalues[i])
    add(f"ols_nonrobust.p.{name}", res.pvalues[i])
add("ols_nonrobust.r_squared", res.rsquared)
add("ols_nonrobust.adj_r_squared", res.rsquared_adj)
add("ols_nonrobust.f_statistic", res.fvalue)
add("ols_nonrobust.log_likelihood", res.llf)
add("ols_nonrobust.aic", res.aic)
add("ols_nonrobust.bic", res.bic)
add("ols_nonrobust.sigma", np.sqrt(res.mse_resid))
add("ols_nonrobust.n_obs", float(res.nobs))
add("ols_nonrobust.df_resid", float(res.df_resid))

# OLS HC1
res_hc1 = sm.OLS(y, X).fit(cov_type="HC1")
for i, name in enumerate(["const", "x1", "x2", "x3"]):
    add(f"ols_hc1.se.{name}", res_hc1.bse[i])
    add(f"ols_hc1.t.{name}", res_hc1.tvalues[i])
    add(f"ols_hc1.p.{name}", res_hc1.pvalues[i])

# OLS HC3
res_hc3 = sm.OLS(y, X).fit(cov_type="HC3")
for i, name in enumerate(["const", "x1", "x2", "x3"]):
    add(f"ols_hc3.se.{name}", res_hc3.bse[i])
    add(f"ols_hc3.t.{name}", res_hc3.tvalues[i])
    add(f"ols_hc3.p.{name}", res_hc3.pvalues[i])

# OLS with Normal inference (z-values instead of t-values)
res_norm = sm.OLS(y, X).fit(use_t=False)
for i, name in enumerate(["const", "x1", "x2", "x3"]):
    add(f"ols_normal.z.{name}", res_norm.tvalues[i])
    add(f"ols_normal.p.{name}", res_norm.pvalues[i])

# OLS HC2
res_hc2 = sm.OLS(y, X).fit(cov_type="HC2")
for i, name in enumerate(["const", "x1", "x2", "x3"]):
    add(f"ols_hc2.se.{name}", res_hc2.bse[i])
    add(f"ols_hc2.t.{name}", res_hc2.tvalues[i])
    add(f"ols_hc2.p.{name}", res_hc2.pvalues[i])

# Note: HC4 is not supported by statsmodels, so no reference values for it.

# OLS Newey-West (HAC)
res_nw = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
for i, name in enumerate(["const", "x1", "x2", "x3"]):
    add(f"ols_nw.se.{name}", res_nw.bse[i])
    add(f"ols_nw.t.{name}", res_nw.tvalues[i])
    add(f"ols_nw.p.{name}", res_nw.pvalues[i])

# OLS Clustered (create cluster IDs: 10 clusters of 5 obs each)
cluster_ids = np.repeat(np.arange(10), 5)
res_cl = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": cluster_ids})
for i, name in enumerate(["const", "x1", "x2", "x3"]):
    add(f"ols_clustered.se.{name}", res_cl.bse[i])
    add(f"ols_clustered.t.{name}", res_cl.tvalues[i])
    add(f"ols_clustered.p.{name}", res_cl.pvalues[i])
# Save cluster IDs for Rust test
for i in range(n):
    add(f"ols_data.cluster.{i}", float(cluster_ids[i]))

# ============================================================================
# DATASET 2: IV / 2SLS (n=100)
# ============================================================================
n = 100
z1 = np.random.randn(n)
z2 = np.random.randn(n)
u = np.random.randn(n)
x_endog = 0.5 * z1 + 0.3 * z2 + 0.5 * u  # endogenous regressor
y_iv = 2.0 + 1.0 * x_endog + u

for i in range(n):
    add(f"iv_data.y.{i}", y_iv[i])
    add(f"iv_data.x_endog.{i}", x_endog[i])
    add(f"iv_data.z1.{i}", z1[i])
    add(f"iv_data.z2.{i}", z2[i])

from statsmodels.sandbox.regression.gmm import IV2SLS

res_iv = IV2SLS(
    y_iv,
    sm.add_constant(x_endog),
    sm.add_constant(np.column_stack([z1, z2])),
).fit()
for i, name in enumerate(["const", "x_endog"]):
    add(f"iv_2sls.params.{name}", res_iv.params[i])
    add(f"iv_2sls.se.{name}", res_iv.bse[i])
add("iv_2sls.r_squared", res_iv.rsquared)

# ============================================================================
# DATASET 3: Logit & Probit (n=200)
# ============================================================================
n = 200
x1_bin = np.random.randn(n)
x2_bin = np.random.uniform(-2, 2, n)
latent = -1.0 + 0.8 * x1_bin + 1.2 * x2_bin + np.random.logistic(0, 1, n)
y_bin = (latent > 0).astype(float)

for i in range(n):
    add(f"binary_data.y.{i}", y_bin[i])
    add(f"binary_data.x1.{i}", x1_bin[i])
    add(f"binary_data.x2.{i}", x2_bin[i])

X_bin = sm.add_constant(np.column_stack([x1_bin, x2_bin]))

# Logit
res_logit = Logit(y_bin, X_bin).fit(disp=0)
for i, name in enumerate(["const", "x1", "x2"]):
    add(f"logit.params.{name}", res_logit.params[i])
    add(f"logit.se.{name}", res_logit.bse[i])
    add(f"logit.z.{name}", res_logit.tvalues[i])
    add(f"logit.p.{name}", res_logit.pvalues[i])
add("logit.log_likelihood", res_logit.llf)
add("logit.pseudo_r2", res_logit.prsquared)

# Probit
res_probit = Probit(y_bin, X_bin).fit(disp=0)
for i, name in enumerate(["const", "x1", "x2"]):
    add(f"probit.params.{name}", res_probit.params[i])
    add(f"probit.se.{name}", res_probit.bse[i])
    add(f"probit.z.{name}", res_probit.tvalues[i])
    add(f"probit.p.{name}", res_probit.pvalues[i])
add("probit.log_likelihood", res_probit.llf)
add("probit.pseudo_r2", res_probit.prsquared)

# Logit: average marginal effects
mfx = res_logit.get_margeff(at="overall")
for i, name in enumerate(["x1", "x2"]):
    add(f"logit.ame.{name}", mfx.margeff[i])
    add(f"logit.ame_se.{name}", mfx.margeff_se[i])

# ============================================================================
# DATASET 4: Panel Data - Fixed & Random Effects (N=20, T=5)
# ============================================================================
N_panel, T_panel = 20, 5
n_panel = N_panel * T_panel
entity_ids = np.repeat(np.arange(N_panel), T_panel)
time_ids = np.tile(np.arange(T_panel), N_panel)
alpha_i = np.repeat(np.random.randn(N_panel) * 2, T_panel)  # entity effects
x1_panel = np.random.randn(n_panel)
x2_panel = np.random.uniform(0, 5, n_panel)
eps_panel = np.random.randn(n_panel) * 0.5
y_panel = 1.0 + 2.0 * x1_panel - 0.5 * x2_panel + alpha_i + eps_panel

for i in range(n_panel):
    add(f"panel_data.y.{i}", y_panel[i])
    add(f"panel_data.x1.{i}", x1_panel[i])
    add(f"panel_data.x2.{i}", x2_panel[i])
    add(f"panel_data.entity.{i}", float(entity_ids[i]))
    add(f"panel_data.time.{i}", float(time_ids[i]))

import pandas as pd

panel_df = pd.DataFrame(
    {
        "y": y_panel,
        "x1": x1_panel,
        "x2": x2_panel,
        "entity": entity_ids,
        "time": time_ids,
    }
)
panel_df = panel_df.set_index(["entity", "time"])

# Fixed Effects (within estimator)
fe_mod = PanelOLS(panel_df["y"], panel_df[["x1", "x2"]], entity_effects=True)
fe_res = fe_mod.fit()
for i, name in enumerate(["x1", "x2"]):
    add(f"panel_fe.params.{name}", fe_res.params[name])
    add(f"panel_fe.se.{name}", fe_res.std_errors[name])
    add(f"panel_fe.t.{name}", fe_res.tstats[name])
    add(f"panel_fe.p.{name}", fe_res.pvalues[name])
add("panel_fe.r_squared", fe_res.rsquared)
add("panel_fe.n_obs", float(fe_res.nobs))

# Random Effects
re_mod = RandomEffects(panel_df["y"], sm.add_constant(panel_df[["x1", "x2"]]))
re_res = re_mod.fit()
for name in ["const", "x1", "x2"]:
    add(f"panel_re.params.{name}", re_res.params[name])
    add(f"panel_re.se.{name}", re_res.std_errors[name])
    add(f"panel_re.t.{name}", re_res.tstats[name])
    add(f"panel_re.p.{name}", re_res.pvalues[name])
add("panel_re.r_squared", re_res.rsquared)

# ============================================================================
# DATASET 5: Quantile Regression (n=100, tau=0.25, 0.50, 0.75)
# ============================================================================
n = 100
x1_q = np.random.randn(n)
x2_q = np.random.uniform(0, 5, n)
# Heteroskedastic errors to make quantiles differ
eps_q = np.random.randn(n) * (1 + 0.5 * np.abs(x1_q))
y_q = 2.0 + 1.0 * x1_q - 0.3 * x2_q + eps_q

for i in range(n):
    add(f"quantile_data.y.{i}", y_q[i])
    add(f"quantile_data.x1.{i}", x1_q[i])
    add(f"quantile_data.x2.{i}", x2_q[i])

X_q = sm.add_constant(np.column_stack([x1_q, x2_q]))

for tau in [0.25, 0.50, 0.75]:
    res_qr = QuantReg(y_q, X_q).fit(q=tau)
    tau_str = f"{tau:.2f}"
    for i, name in enumerate(["const", "x1", "x2"]):
        add(f"quantile_{tau_str}.params.{name}", res_qr.params[i])
        add(f"quantile_{tau_str}.se.{name}", res_qr.bse[i])
        add(f"quantile_{tau_str}.t.{name}", res_qr.tvalues[i])
        add(f"quantile_{tau_str}.p.{name}", res_qr.pvalues[i])

# ============================================================================
# DATASET 6: WLS (n=50, known heteroskedastic weights)
# ============================================================================
n = 50
x1_w = np.random.randn(n)
# Variance proportional to |x1| + 1
sigma_i = np.sqrt(np.abs(x1_w) + 1.0)
eps_w = np.random.randn(n) * sigma_i
y_w = 5.0 - 2.0 * x1_w + eps_w
weights = 1.0 / (sigma_i ** 2)

for i in range(n):
    add(f"wls_data.y.{i}", y_w[i])
    add(f"wls_data.x1.{i}", x1_w[i])
    add(f"wls_data.weights.{i}", weights[i])

X_w = sm.add_constant(x1_w)
res_wls = sm.WLS(y_w, X_w, weights=weights).fit()
for i, name in enumerate(["const", "x1"]):
    add(f"wls.params.{name}", res_wls.params[i])
    add(f"wls.se.{name}", res_wls.bse[i])
    add(f"wls.t.{name}", res_wls.tvalues[i])
    add(f"wls.p.{name}", res_wls.pvalues[i])
add("wls.r_squared", res_wls.rsquared)

# ============================================================================
# Write CSV
# ============================================================================
with open("tests/reference_values.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["key", "value"])
    writer.writerows(rows)

print(f"Written {len(rows)} reference values to tests/reference_values.csv")
