import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")
SEED = 20260811
rng = np.random.default_rng(SEED)

G = np.linspace(0.0, 4.0, 1001)
F_theory = np.sqrt(G) / 2.0
u_levels = np.array([0.1, 0.5, 0.9])

fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.plot(G, F_theory, color="black", linewidth=2.3, label=r"Theoretical Fy(y) = sqrt(y)/2")

print(f"Reported random seed: {SEED}\n")
quantile_results = {"Theoretical": 4.0 * (u_levels ** 2)}

for n, color in [(500, "tab:orange"), (10_000, "tab:blue")]:
    y = rng.uniform(0.0, 2.0, size=n) ** 2
    
    ecdf_on_G = np.searchsorted(np.sort(y), G, side="right") / n
    d_n = np.max(np.abs(ecdf_on_G - F_theory))
    print(f"Maximum absolute difference D_{n:<5} = {d_n:.6f}")
    
    ax.step(G, ecdf_on_G, where="post", color=color, alpha=0.8, label=f"Empirical, n={n:,}")
    
    quantile_results[f"n = {n:,}"] = np.quantile(y, u_levels, method="inverted_cdf")

ax.set(xlabel="$y$", ylabel="Cumulative probability", xlim=(0, 4), ylim=(0, 1.02),
       title="Empirical CDFs approach the theoretical CDF on [0, 4]")
ax.legend()
plt.tight_layout()
plt.show()

print("\nQuantile comparison:")
print(f"{'Level u':<10} | {'Theoretical':<14} | {'n = 500':<12} | {'n = 10,000':<12}")
print("-" * 56)
for i, u in enumerate(u_levels):
    print(f"{u:<10.1f} | {quantile_results['Theoretical'][i]:<14.4f} | "
          f"{quantile_results['n = 500'][i]:<12.4f} | {quantile_results['n = 10,000'][i]:<12.4f}")
