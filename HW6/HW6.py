import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

np.random.seed(42)
N, s, rate = 30_000, 0.5, 2.0
q_levels = [0.1, 0.5, 0.9]

# 1. Exponential Model: Exp(2)
orig_exp = stats.expon.rvs(scale=1.0 / rate, size=N)
rem_exp = orig_exp[orig_exp > s] - s

# Quantiles table
print("Exp(2) Quantiles Table:")
print(pd.DataFrame({
    "Quantile": q_levels,
    "Original Emp": np.quantile(orig_exp, q_levels, method="inverted_cdf"),
    "Remaining Emp": np.quantile(rem_exp, q_levels, method="inverted_cdf"),
    "Theoretical": stats.expon.ppf(q_levels, scale=1.0 / rate)
}).to_string(index=False))

# Empirical survival & discrepancies
surv = lambda sample, grid: np.mean(sample[:, None] > grid[None, :], axis=0)

grid_exp = np.linspace(0, 3, 301)
theo_exp = np.exp(-rate * grid_exp)
print(f"\nMax discrepancy [0, 3] (Original):  {np.max(np.abs(surv(orig_exp, grid_exp) - theo_exp)):.5f}")
print(f"Max discrepancy [0, 3] (Remaining): {np.max(np.abs(surv(rem_exp, grid_exp) - theo_exp)):.5f}")

# 2. Uniform Model: Unif(0, 1)
orig_unif = stats.uniform.rvs(0.0, 1.0, size=N)
rem_unif = orig_unif[orig_unif > s] - s
grid_unif = np.linspace(0, 1, 201)

# 3. Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(grid_exp, theo_exp, "k--", lw=2, label=r"Theory: $e^{-2t}$")
ax1.plot(grid_exp, surv(orig_exp, grid_exp), label="Original", alpha=0.8)
ax1.plot(grid_exp, surv(rem_exp, grid_exp), ":", lw=2, label="Remaining")
ax1.set_title(r"Exp(2) Survival ($s=0.5$)")

ax2.plot(grid_unif, 1.0 - grid_unif, "k--", lw=2, label="Orig Theory: $1-t$")
ax2.plot(grid_unif, np.maximum(0.0, 1.0 - 2.0 * grid_unif), "r--", lw=2, label=r"Rem Theory: $(1-2t)_+$")
ax2.plot(grid_unif, surv(orig_unif, grid_unif), label="Original", alpha=0.8)
ax2.plot(grid_unif, surv(rem_unif, grid_unif), ":", lw=2, label="Remaining")
ax2.set_title(r"Unif(0, 1) Survival ($s=0.5$)")

for ax in (ax1, ax2):
    ax.set(xlabel="t", ylabel=r"$\mathbb{P}(T > t)$")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()

plt.tight_layout()
plt.show()