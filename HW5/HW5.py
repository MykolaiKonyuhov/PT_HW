import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.style.use("seaborn-v0_8-whitegrid")

def tv_distance(n, lam):
    p, k = lam / n, np.arange(n + 1)
    binom, pois = stats.binom(n, p), stats.poisson(lam)
    return 0.5 * (np.sum(np.abs(binom.pmf(k) - pois.pmf(k))) + pois.sf(n))

lambdas = [1, 5, 10]
n_grid = [21, 51, 101, 501, 1000]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

for lam in lambdas:
    n_vals = [lam] + [n for n in n_grid if n > lam]
    c_lam = 2 * lam
    pois = stats.poisson(lam)
    
    dtv = [tv_distance(n, lam) for n in n_vals]
    tail_err = [abs(stats.binom(n, lam / n).sf(c_lam - 1) - pois.sf(c_lam - 1)) for n in n_vals]
    
    ax1.plot(n_vals, dtv, marker="o", label=rf"$\lambda = {lam}$")
    ax2.plot(n_vals, tail_err, marker="s", label=rf"$\lambda = {lam}$")

ax1.set(xscale="log", xlabel="Sample size $n$ (log scale)",
        ylabel=r"$d_{\mathrm{TV}}(p_{X_n}, p_Y)$", title="Total-Variation Distance vs $n$")
ax1.legend()

ax2.set(xscale="log", xlabel="Sample size $n$ (log scale)",
        ylabel=r"$|\mathbb{P}(X_n \geq 2\lambda) - \mathbb{P}(Y \geq 2\lambda)|$",
        title=r"Tail Error at $c_\lambda = 2\lambda$")
ax2.legend()

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

for ax, (lam, n) in zip(axes, [(10, 10), (1, 1000)]):
    binom, pois = stats.binom(n, lam / n), stats.poisson(lam)
    
    if lam == 1:
        k = np.arange(7)
        ax.set_xlim(-0.8, 6.8)
        ax.set_xticks(range(7))
    else:
        k = np.arange(15)
        ax.set_xlim(-0.8, 14.8)
        ax.set_xticks(range(0, 15, 2))
        
    w = 0.35
    ax.bar(k - w / 2, binom.pmf(k), width=w, label=f"Binomial({n}, {lam/n:.3f})",
           color="tab:blue", alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.bar(k + w / 2, pois.pmf(k), width=w, label=f"Poisson({lam})",
           color="tab:orange", alpha=0.85, edgecolor="black", linewidth=0.8)
    
    ax.set(xlabel="Count $k$", ylabel="Probability",
           title=rf"$(\lambda={lam}, n={n})$: $d_{{\mathrm{{TV}}}} = {tv_distance(n, lam):.4f}$")
    ax.legend()

plt.tight_layout()
plt.show()