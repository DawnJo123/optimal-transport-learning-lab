import numpy as np
from scipy.optimize import linprog
import ot

# Probability distributions
a = np.array([0.5, 0.5])
b = np.array([0.5, 0.5])

# Cost matrix
C = np.array([
    [1.0, 3.0],
    [1.0, 1.0]
])

T=ot.emd(a,b,C)

min_cost=ot.emd2(a,b,C)

print("Optimal transport plan: \n", T)

print("Minimum cost:\n",min_cost)

