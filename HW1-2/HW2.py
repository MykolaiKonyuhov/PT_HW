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