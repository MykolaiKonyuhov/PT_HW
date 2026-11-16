import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson

# Parameters
lambdas = [1, 5, 10]
multipliers = [1, 2, 5, 10, 50, 100]

def compute_tv_distance(n, lam):
    #Exact total variation distance between Binomial(n, lam/n) and Poisson(lam)
    p = lam / n
    k = np.arange(0, n + 1)
    pmf_binom = binom.pmf(k, n, p)
    pmf_pois = poisson.pmf(k, lam)
    
    # Sum differences over [0, n] + full Poisson tail probability P(Y > n)
    diff_sum = np.sum(np.abs(pmf_binom - pmf_pois))
    tail_pois = poisson.sf(n, lam)
    
    return 0.5 * (diff_sum + tail_pois)

def compute_tail_error(n, lam):
    #Absolute difference |P(X_n >= 2*lam) - P(Y >= 2*lam)| using sf at c_lam - 1
    p = lam / n
    c_lam = 2 * lam
    
    prob_binom = binom.sf(c_lam - 1, n, p)
    prob_pois = poisson.sf(c_lam - 1, lam)
    
    return np.abs(prob_binom - prob_pois)

# Parts a, b 
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for lam in lambdas:
    n_values = [m * lam for m in multipliers]
    tv_dists = [compute_tv_distance(n, lam) for n in n_values]
    tail_errs = [compute_tail_error(n, lam) for n in n_values]
    
    # Plot (a): Total Variation Distance
    axes[0].plot(n_values, tv_dists, marker='o', linewidth=2, label=rf'$\lambda = {lam}$')
    
    # Plot (b): Tail Probability Error
    axes[1].plot(n_values, tail_errs, marker='s', linewidth=2, label=rf'$\lambda = {lam}$ ($c_\lambda = {2*lam}$)')

# Formatting Plot (a)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_title(r'(a) Total Variation Distance $d_{\mathrm{TV}}(p_{X_n}, p_Y)$ vs. $n$', fontsize=12)
axes[0].set_xlabel(r'$n$ (log scale)', fontsize=11)
axes[0].set_ylabel(r'$d_{\mathrm{TV}}(p_{X_n}, p_Y)$ (log scale)', fontsize=11)
axes[0].grid(True, which='both', linestyle='--', alpha=0.5)
axes[0].legend()

# Formatting Plot (b)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_title(r'(b) Tail Error $|\mathbb{P}(X_n \geq 2\lambda) - \mathbb{P}(Y \geq 2\lambda)|$ vs. $n$', fontsize=12)
axes[1].set_xlabel(r'$n$ (log scale)', fontsize=11)
axes[1].set_ylabel('Absolute Tail Error (log scale)', fontsize=11)
axes[1].grid(True, which='both', linestyle='--', alpha=0.5)
axes[1].legend()

plt.tight_layout()
plt.show()

# c) PMF Comparison for large vs. small TV Distance 
# P1 Large distance -> (lambda=5, n=5) [i.e., n = lambda, p = 1]
# P2 Small distance -> (lambda=5, n=500) [i.e., n = 100*lambda, p = 0.01]
pairs = [(5, 5), (5, 500)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (lam, n) in enumerate(pairs):
    p = lam / n
    tv = compute_tv_distance(n, lam)
    
    # Plot range covering the bulk of the Poisson(5) distribution
    k_max = int(lam * 2.5)
    k = np.arange(0, k_max + 1)
    pmf_binom = binom.pmf(k, n, p)
    pmf_pois = poisson.pmf(k, lam)
    
    width = 0.38
    axes[idx].bar(k - width/2, pmf_binom, width=width, label=rf'$\mathrm{{Binomial}}({n}, {p:.2f})$', alpha=0.85)
    axes[idx].bar(k + width/2, pmf_pois, width=width, label=rf'$\mathrm{{Poisson}}({lam})$', alpha=0.85)
    
    axes[idx].set_title(rf'$(\lambda={lam}, n={n}) \Rightarrow p={p:.2f},\ d_{{\mathrm{{TV}}}} = {tv:.4f}$', fontsize=12)
    axes[idx].set_xlabel('Count $k$', fontsize=11)
    axes[idx].set_ylabel('Probability Mass', fontsize=11)
    axes[idx].set_xticks(k)
    axes[idx].grid(axis='y', linestyle='--', alpha=0.5)
    axes[idx].legend()

plt.tight_layout()
plt.show()