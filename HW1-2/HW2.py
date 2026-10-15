import itertools
import math
import random

dataset = [1] * 4 + [0] * 11

combs = list(itertools.combinations(dataset, 5))
p_itertools = sum(sum(c) >= 2 for c in combs) / len(combs)

p_formula = 1 - (math.comb(4, 0) * math.comb(11, 5) + math.comb(4, 1) * math.comb(11, 4)) / math.comb(15, 5)

trials = 100000
p_monte_carlo = sum(sum(random.sample(dataset, 5)) >= 2 for r in range(trials)) / trials

print(f"Itertools:   {p_itertools:.6f}")
print(f"Formula:     {p_formula:.6f}")
print(f"Monte Carlo: {p_monte_carlo:.6f}")