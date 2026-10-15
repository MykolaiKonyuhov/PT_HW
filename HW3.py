import numpy as np
import matplotlib.pyplot as plt

# 1. Параметри тесту
sensitivity = 0.95
specificity = 0.99
fp_rate = 1 - specificity  # 0.01

# 2. Теоретична крива
p_grid = np.linspace(0.001, 0.5, 500)
posterior_exact = (sensitivity * p_grid) / (sensitivity * p_grid + fp_rate * (1 - p_grid))

# Порогове значення для posterior = 0.5
p_threshold = fp_rate / (2 * sensitivity - (sensitivity - fp_rate))  # 1/96 ≈ 0.010417

# 3. Симуляція Монте-Карло для вибраних рівнів prevalence
eval_prevalences = [0.001, 0.005, 0.010417, 0.05, 0.1, 0.2, 0.35, 0.5]
N = 200_000  # розмір модельованої популяції
mc_estimates = []

np.random.seed(42)
for p in eval_prevalences:
    # 1 - хворий (D), 0 - здоровий (D^c)
    has_disease = np.random.rand(N) < p
    
    # Генерація результатів тестів
    test_positive = np.zeros(N, dtype=bool)
    test_positive[has_disease] = np.random.rand(np.sum(has_disease)) < sensitivity
    test_positive[~has_disease] = np.random.rand(np.sum(~has_disease)) < fp_rate
    
    # Емпірична частка реально хворих серед тих, у кого тест позитивний
    positives_count = np.sum(test_positive)
    if positives_count > 0:
        emp_fraction = np.sum(has_disease & test_positive) / positives_count
    else:
        emp_fraction = 0.0
    mc_estimates.append(emp_fraction)

# 4. Побудова графіка
plt.figure(figsize=(9, 5.5))
plt.plot(p_grid, posterior_exact, label="Exact Posterior $\mathbb{P}(D \mid T^+)$", color="#1f77b4", lw=2)
plt.scatter(eval_prevalences, mc_estimates, color="#d62728", zorder=5, label="Monte Carlo Estimates ($N=200\,000$)")

# Позначення точки threshold = 0.5
plt.axvline(x=p_threshold, color="gray", linestyle="--", alpha=0.7)
plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7)
plt.scatter([p_threshold], [0.5], color="black", s=60, zorder=6, label=f"Threshold: $p \\approx {p_threshold:.4f}$")

plt.title("Posterior Probability $\mathbb{P}(D \mid T^+)$ vs Prevalence", fontsize=13)
plt.xlabel("Prevalence $\mathbb{P}(D)$", fontsize=11)
plt.ylabel("Posterior $\mathbb{P}(D \mid T^+)$", fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.show()