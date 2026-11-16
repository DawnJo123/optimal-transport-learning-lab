import numpy as np
from scipy.spatial.distance import cdist
import ot

# Source points
X = np.array([
    [0.0],
    [2.0],
    [4.0]
])

# Target points
Y = np.array([
    [1.0],
    [3.0],
    [5.0]
])

#Probability distribution
a=np.array([0.2,0.5,0.3])
b=np.array([0.4,0.3,0.3])

#Cost matrix Calculation
C=cdist(X,Y, metric="euclidean")

#Optimal Transport plan
T=ot.emd(a,b,C)

#Minimum transport Cost
min_cost=ot.emd2(a,b,C)

print("Cost matrix: \n", C)
print("Transport plan: \n", T)
print("Minimum Cost: ",min_cost)