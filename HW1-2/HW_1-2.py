import numpy as np
import matplotlib.pyplot as plt


plt.style.use("seaborn-v0_8-whitegrid")

faces = np.arange(1, 7)
probs = np.array([0.08, 0.10, 0.12, 0.15, 0.20, 0.35])
assert np.isclose(probs.sum(), 1.0)

A = {2, 4, 6}
p_A = sum(probs[k - 1] for k in A)  

n = 5000
seeds = [202601, 202602]
m_values = np.arange(1, n + 1)
checkpoints = [50, 500, 5000]

fig, ax = plt.subplots(figsize=(10, 5))

for seed in seeds:
    rng = np.random.default_rng(seed)
    
    rolls = rng.choice(faces, size=n, p=probs)
    
    in_A = np.isin(rolls, list(A))
    
    p_hat_m = np.cumsum(in_A) / m_values
    
    ax.plot(m_values, p_hat_m, lw=1.2, label=f"Seed {seed}")
    
    print(f"--- Результати для seed = {seed} ---")
    for m in checkpoints:
        val = p_hat_m[m - 1]
        dev = abs(val - p_A)
        print(f"m = {m:4d} | p_hat_m(A) = {val:.4f} | |p_hat_m(A) - P(A)| = {dev:.4f}")

ax.axhline(p_A, color="black", ls="--", lw=1.5, label=rf"Model value $\mathbb{{P}}(A) = {p_A:.2f}$")

ax.set(
    xlabel=r"Number of rolls $m$",
    ylabel=r"Relative frequency $\widehat{p}_m(A)$",
    ylim=(0.4, 0.8),
    title=r"Running relative frequency $\widehat{p}_m(A)$ vs. $\mathbb{P}(A)$ for $n=5000$"
)
ax.legend()
plt.tight_layout()
plt.show()