#4
import numpy as np
import matplotlib.pyplot as plt

sensitivity = 0.95
specificity = 0.99
fp_rate = 1 - specificity  # P(T^+|D^c)

p_grid = np.linspace(0.001, 0.5, 500)
posterior_exact = (sensitivity * p_grid) / (sensitivity * p_grid + fp_rate * (1 - p_grid))
p_threshold = fp_rate / (2 * sensitivity - (sensitivity - fp_rate))  #1/96

#Monte Carlo
eval_prevalences = [0.001, 0.005, 0.010417, 0.05, 0.1, 0.2, 0.35, 0.5]
N = 200000  
mc_estimates = []
np.random.seed(42)

for p in eval_prevalences:
    has_disease = np.random.rand(N) < p

    test_positive = np.zeros(N, dtype=bool)
    test_positive[has_disease] = np.random.rand(np.sum(has_disease)) < sensitivity
    test_positive[~has_disease] = np.random.rand(np.sum(~has_disease)) < fp_rate
    
    positives_count = np.sum(test_positive)
    if positives_count > 0:
        emp_fraction = np.sum(has_disease & test_positive) / positives_count
    else:
        emp_fraction = 0.0
    mc_estimates.append(emp_fraction)


plt.figure(figsize=(9, 5.5))
plt.plot(p_grid, posterior_exact, label="Exact Posterior P(D|T^+)", color="royalblue", lw=2)
plt.scatter(eval_prevalences, mc_estimates, color="red", zorder=5, label="Monte Carlo Estimates ($N=200\,000$)")

plt.axvline(x=p_threshold, color="gray", linestyle="--", alpha=0.7)
plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7)
plt.scatter([p_threshold], [0.5], color="black", s=60, zorder=6, label=f"Threshold: p≈{p_threshold:.4f}")

plt.title("Posterior Probability P(D|T^+) vs Prevalence", fontsize=13)
plt.xlabel("Prevalence P(D)", fontsize=11)
plt.ylabel("Posterior P(D|T^+)", fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.show()