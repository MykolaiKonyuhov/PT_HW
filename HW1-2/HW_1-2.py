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
import random

dataset = [1] * 4 + [0] * 11

all_combinations = list(itertools.combinations(dataset, 5))
favorable_combinations = [c for c in all_combinations if sum(c) >= 2]
exact_prob_itertools = len(favorable_combinations) / len(all_combinations)

total_ways = math.comb(15, 5)
ways_0 = math.comb(4, 0) * math.comb(11, 5)
ways_1 = math.comb(4, 1) * math.comb(11, 4)
exact_prob_formula = 1 - (ways_0 + ways_1) / total_ways

trials = 1000000
successes = sum(sum(random.sample(dataset, 5)) >= 2 for _ in range(trials))
mc_prob = successes / trials

print(f"Itertools probability: {exact_prob_itertools:.6f}")
print(f"Formula probability:   {exact_prob_formula:.6f}")
print(f"Monte Carlo result:    {mc_prob:.6f}")