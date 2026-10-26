import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

np.random.seed(42)
N = 30_000
s = 0.5
rate = 2.0
q_levels = [0.1, 0.5, 0.9]

# 1. Exponential Model: Exp(2) 
# SciPy uses reciprocal rate: scale = 1 / rate
orig_exp = stats.expon.rvs(scale=1.0 / rate, size=N)
rem_exp = orig_exp[orig_exp > s] - s

# Empirical quantiles using inverted CDF and theoretical counterparts
orig_q = np.quantile(orig_exp, q_levels, method="inverted_cdf")
rem_q = np.quantile(rem_exp, q_levels, method="inverted_cdf")
theo_q = stats.expon.ppf(q_levels, scale=1.0 / rate)

quantiles_df = pd.DataFrame({
    "Quantile": q_levels,
    "Original Emp": orig_q,
    "Remaining Emp": rem_q,
    "Theoretical Exp(2)": theo_q
})
print("Exp(2) Quantiles Table:")
print(quantiles_df.to_string(index=False))

# Empirical survival function P(T > t)
def empirical_survival(sample, grid):
    return np.mean(sample[:, None] > grid[None, :], axis=0)

grid_exp = np.linspace(0, 3, 301)
surv_orig_exp = empirical_survival(orig_exp, grid_exp)
surv_rem_exp = empirical_survival(rem_exp, grid_exp)
theo_surv_exp = np.exp(-rate * grid_exp)

# Maximum absolute discrepancy on grid [0, 3]
max_err_orig = np.max(np.abs(surv_orig_exp - theo_surv_exp))
max_err_rem = np.max(np.abs(surv_rem_exp - theo_surv_exp))
print(f"\nMax discrepancy on [0, 3] (Original): {max_err_orig:.5f}")
print(f"Max discrepancy on [0, 3] (Remaining): {max_err_rem:.5f}")

# 2. Uniform Model: Unif(0, 1)
orig_unif = stats.uniform.rvs(loc=0.0, scale=1.0, size=N)
rem_unif = orig_unif[orig_unif > s] - s

grid_unif = np.linspace(0, 1, 201)
surv_orig_unif = empirical_survival(orig_unif, grid_unif)
surv_rem_unif = empirical_survival(rem_unif, grid_unif)
theo_surv_rem_unif = np.maximum(0.0, 1.0 - 2.0 * grid_unif)

# 3. Visualization 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot for Exp(2)
ax1.plot(grid_exp, theo_surv_exp, "k--", label=r"Theory: $e^{-2t}$", linewidth=2)
ax1.plot(grid_exp, surv_orig_exp, label="Original waits", alpha=0.8)
ax1.plot(grid_exp, surv_rem_exp, ":", label=r"Remaining ($T-s \mid T>s$)", linewidth=2)
ax1.set_title(r"Exp(2) Survival Curves ($s=0.5$)")
ax1.set_xlabel("t")
ax1.set_ylabel(r"$\mathbb{P}(T > t)$")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

# Plot for Unif(0, 1)
ax2.plot(grid_unif, np.maximum(0.0, 1.0 - grid_unif), "k--", label=r"Original Theory: $1-t$", linewidth=2)
ax2.plot(grid_unif, theo_surv_rem_unif, "r--", label=r"Remaining Theory: $(1-2t)_+$", linewidth=2)
ax2.plot(grid_unif, surv_orig_unif, label="Original waits", alpha=0.8)
ax2.plot(grid_unif, surv_rem_unif, ":", label="Remaining waits", linewidth=2)
ax2.set_title(r"Unif(0, 1) Survival Curves ($s=0.5$)")
ax2.set_xlabel("t")
ax2.set_ylabel(r"$\mathbb{P}(T > t)$")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()