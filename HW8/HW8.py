import matplotlib.pyplot as plt
import numpy as np

# Set seed for reproducibility
rng = np.random.default_rng(42)
n = 2000

# Model A: X, Y ~ iid N(0, 1)
X_A = rng.standard_normal(n)
Y_A = rng.standard_normal(n)

# Model B: X ~ Unif(-1, 1), Y = X^2 + 0.05 * eps, eps ~ N(0, 1)
X_B = rng.uniform(-1.0, 1.0, size=n)
eps = rng.standard_normal(n)
Y_B = X_B**2 + 0.05 * eps

# Sample covariance (ddof=1) and Pearson correlation
cov_A = np.cov(X_A, Y_A, ddof=1)[0, 1]
corr_A = np.corrcoef(X_A, Y_A)[0, 1]

cov_B = np.cov(X_B, Y_B, ddof=1)[0, 1]
corr_B = np.corrcoef(X_B, Y_B)[0, 1]

print("Sample Covariance and Correlation (n = 2000)")
print(f"Model A: Cov(X, Y) = {cov_A:+.5f},  Corr(X, Y) = {corr_A:+.5f}")
print(f"Model B: Cov(X, Y) = {cov_B:+.5f},  Corr(X, Y) = {corr_B:+.5f}")


def binned_means(x, y, num_bins=10):
	edges = np.linspace(np.min(x), np.max(x), num_bins + 1)
	centers = 0.5 * (edges[:-1] + edges[1:])
	bin_idx = np.clip(np.digitize(x, edges) - 1, 0, num_bins - 1)
	means = np.array([
		y[bin_idx == i].mean() if np.any(bin_idx == i) else np.nan
		for i in range(num_bins)
	])
	return centers, means


centers_A, means_A = binned_means(X_A, Y_A, num_bins=10)
centers_B, means_B = binned_means(X_B, Y_B, num_bins=10)

# 1. Side-by-Side plots 
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

axes1[0].scatter(X_A, Y_A, alpha=0.35, s=12, color="steelblue")
axes1[0].set_title(f"Model A: $X, Y \\sim_{{iid}} \\mathcal{{N}}(0,1)$\n(r = {corr_A:+.4f})")
axes1[0].set_xlabel("X")
axes1[0].set_ylabel("Y")
axes1[0].grid(True, linestyle=":", alpha=0.6)

axes1[1].scatter(X_B, Y_B, alpha=0.35, s=12, color="darkorange")
axes1[1].set_title(f"Model B: $Y = X^2 + 0.05\\varepsilon$\n(r = {corr_B:+.4f})")
axes1[1].set_xlabel("X")
axes1[1].set_ylabel("Y")
axes1[1].grid(True, linestyle=":", alpha=0.6)

fig1.suptitle("Side-by-Side Scatterplots (n = 2000)", fontsize=13)
plt.tight_layout()
plt.show()

# 2. Mean of Y within 10 Bins of X 
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

axes2[0].plot(centers_A, means_A, marker="o", linewidth=2, color="steelblue", label="Binned Sample Mean")
axes2[0].axhline(0.0, color="black", linestyle="--", linewidth=1, label=r"True $\mathbb{E}[Y \mid X] = 0$")
axes2[0].set_title("Model A: Mean of Y in 10 Bins of X")
axes2[0].set_xlabel("X (Bin Center)")
axes2[0].set_ylabel("Mean of Y")
axes2[0].set_ylim(-0.5, 1.2)
axes2[0].legend()
axes2[0].grid(True, linestyle=":", alpha=0.6)

x_grid = np.linspace(-1, 1, 200)
axes2[1].plot(centers_B, means_B, marker="o", linewidth=2, color="darkorange", label="Binned Sample Mean")
axes2[1].plot(x_grid, x_grid**2, color="black", linestyle="--", linewidth=1, label=r"True $\mathbb{E}[Y \mid X] = X^2$")
axes2[1].set_title("Model B: Mean of Y in 10 Bins of X")
axes2[1].set_xlabel("X (Bin Center)")
axes2[1].set_ylabel("Mean of Y")
axes2[1].set_ylim(-0.5, 1.2)
axes2[1].legend()
axes2[1].grid(True, linestyle=":", alpha=0.6)

fig2.suptitle("Conditional Mean of Y Across 10 Bins of X", fontsize=13)
plt.tight_layout()
plt.show()