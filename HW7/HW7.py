import matplotlib.pyplot as plt
import numpy as np

# Set seed
rng = np.random.default_rng(42)

# True parameters for X ~ Exp(1)
mu_true = 1.0
var_true = 1.0

n_values = [20, 200, 2000]
B = 2000

mean_errors = {}
var_errors = {}

print("1. Monte Carlo Simulation Results (B = 2000)")
print(
    f"{'n':<6} | {'Mean Bias':<11} | {'Mean RMSE':<11} | {'Var Bias':<11} |"
    f" {'Var RMSE':<11}"
)
print("-" * 62)

for n in n_values:
    # Generate B independent samples of size n: shape (B, n)
    samples = rng.exponential(scale=1.0, size=(B, n))

    # Compute sample mean and unbiased sample variance (ddof=1) for each simulation
    sample_means = np.mean(samples, axis=1)
    sample_vars = np.var(samples, axis=1, ddof=1)

    # Errors
    err_mean = sample_means - mu_true
    err_var = sample_vars - var_true

    mean_errors[n] = err_mean
    var_errors[n] = err_var

    # Empirical Bias and RMSE
    bias_mean = np.mean(err_mean)
    rmse_mean = np.sqrt(np.mean(err_mean**2))

    bias_var = np.mean(err_var)
    rmse_var = np.sqrt(np.mean(err_var**2))

    print(
        f"{n:<6} | {bias_mean:<11.5f} | {rmse_mean:<11.5f} | {bias_var:<11.5f} |"
        f" {rmse_var:<11.5f}"
    )

# 2 Figure Comparing Error Distributions Across n
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

axes[0].boxplot(
    [mean_errors[n] for n in n_values],
    tick_labels=[f"n = {n}" for n in n_values],
    showmeans=True,
    meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 5},
)
axes[0].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
axes[0].set_title(r"Sample Mean Error Distribution ($\hat{\mu} - 1$)")
axes[0].set_ylabel("Estimation Error")
axes[0].grid(axis="y", linestyle=":", alpha=0.6)

axes[1].boxplot(
    [var_errors[n] for n in n_values],
    tick_labels=[f"n = {n}" for n in n_values],
    showmeans=True,
    meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 5},
)
axes[1].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
axes[1].set_title(r"Sample Variance Error Distribution ($S^2 - 1$)")
axes[1].set_ylabel("Estimation Error")
axes[1].grid(axis="y", linestyle=":", alpha=0.6)

fig.suptitle(
    r"Monte Carlo Error Distributions for $X \sim \mathrm{Exp}(1)$ ($B = 2000$)",
    fontsize=13,
)
plt.tight_layout()
plt.show()

# 3 Contamination Experiment (n = 200)
sample_200 = rng.exponential(scale=1.0, size=200)

# Before contamination
mean_before = np.mean(sample_200)
median_before = np.median(sample_200)
var_before = np.var(sample_200, ddof=1)

# Replace one observation (index 0) with 100
sample_contaminated = sample_200.copy()
sample_contaminated[0] = 100.0

# After contamination
mean_after = np.mean(sample_contaminated)
median_after = np.median(sample_contaminated)
var_after = np.var(sample_contaminated, ddof=1)

print("\n2. Contamination Experiment (n = 200)")
print(f"{'Statistic':<12} | {'Before':<12} | {'After':<12} | {'Change':<12}")
print("-" * 56)
print(
    f"{'Mean':<12} | {mean_before:<12.5f} | {mean_after:<12.5f} |"
    f" {mean_after - mean_before:<+12.5f}"
)
print(
    f"{'Median':<12} | {median_before:<12.5f} | {median_after:<12.5f} |"
    f" {median_after - median_before:<+12.5f}"
)
print(
    f"{'Variance':<12} | {var_before:<12.5f} | {var_after:<12.5f} |"
    f" {var_after - var_before:<+12.5f}"
)