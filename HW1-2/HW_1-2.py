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

print("Lecture 2 starts here")

import itertools
import math
import numpy as np

# -------------------------------------------------------------------------
# Part 1: Exact enumeration using itertools.combinations
# -------------------------------------------------------------------------
# Represent the 15 rows: 1 = flagged (4 rows), 0 = unflagged (11 rows)
dataset = [1] * 4 + [0] * 11
sample_size = 5

all_combinations = list(itertools.combinations(dataset, sample_size))
total_test_sets = len(all_combinations)

# Count how many test sets have at least 2 flagged rows
favorable_cases = sum(1 for combo in all_combinations if sum(combo) >= 2)

prob_itertools = favorable_cases / total_test_sets

print(f"Total test sets enumerated: {total_test_sets}")
print(f"Favorable test sets (>= 2 flagged): {favorable_cases}")
print(f"Exact Probability (itertools): {prob_itertools:.7f} ({favorable_cases}/{total_test_sets})")


# -------------------------------------------------------------------------
# Part 2: Verification using the Combinatorial Formula
# -------------------------------------------------------------------------
N, K, n = 15, 4, 5

comb_total = math.comb(N, n)
prob_0 = math.comb(K, 0) * math.comb(N - K, n - 0) / comb_total
prob_1 = math.comb(K, 1) * math.comb(N - K, n - 1) / comb_total

prob_theoretical = 1 - (prob_0 + prob_1)

print(f"Theoretical Probability (formula): {prob_theoretical:.7f}")


# -------------------------------------------------------------------------
# Part 3: Verification via Monte Carlo Simulation
# -------------------------------------------------------------------------
num_simulations = 1_000_000
rng = np.random.default_rng(seed=42)

# Generate uniform draws without replacement
# We shuffle an array of row labels/flags across iterations
simulated_draws = rng.choice(dataset, size=(num_simulations, sample_size), replace=False)
flagged_counts = simulated_draws.sum(axis=1)

prob_monte_carlo = np.mean(flagged_counts >= 2)

print(f"Monte Carlo Probability ({num_simulations:,} trials): {prob_monte_carlo:.7f}")