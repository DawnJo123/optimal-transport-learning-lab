
import numpy as np
from scipy.optimize import linprog

# Probability distributions
a = np.array([0.5, 0.5])
b = np.array([0.5, 0.5])

# Cost matrix
C = np.array([
    [1.0, 3.0],
    [1.0, 1.0]
])

# Flatten the cost matrix
c = C.flatten()

# Equality constraints: source mass
A_eq = np.array([
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1]
], dtype=float)

# Required marginal values
b_eq = np.concatenate([a, b])

# Non-negativity constraints
bounds = [(0, None)] * 4

# Solve the Linear Programming problem
result = linprog(
    c,
    A_eq=A_eq,
    b_eq=b_eq,
    bounds=bounds,
    method="highs"
)

# Display results
if result.success:
    T = result.x.reshape(2, 2)

    print("Optimal transport plan:")
    print(T)

    print("Minimum transport cost:")
    print(result.fun)

else:
    print("Optimization failed:", result.message)
