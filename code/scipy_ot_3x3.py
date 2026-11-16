import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linprog

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

C=cdist(X,Y, metric="euclidean")

print("Cost Matrix: \n", C)

#Transport plan
# T=[[T11, T12, T13],
#    [T21, T22, T23],
#    [T31, T32, T33]]

# Tflat=[T11, T12, T13, T21, T22, T23, T31, T32, T33]

# T11+T12+T13=0.2
# T21+T22+T23=0.5
# T31+T32+T33=0.3

# T11+T21+T31=0.4
# T12+T22+T32=0.3
# T13+T23+T33=0.3 

A_eq=np.array([
    [1,1,1,0,0,0,0,0,0],
    [0,0,0,1,1,1,0,0,0],
    [0,0,0,0,0,0,1,1,1],
    [1,0,0,1,0,0,1,0,0],
    [0,1,0,0,1,0,0,1,0],
    [0,0,1,0,0,1,0,0,1]
],dtype=float)

B_eq=np.concatenate([a,b])

c=C.flatten()

bounds=[(0,None)]*9

result=linprog(c, A_eq=A_eq, b_eq=B_eq, bounds=bounds, method="highs")

if result.success:
    T=result.x.reshape(3,3)

    print("Cost matrix: \n", C)

    print("\n optimal Transport plan: \n", T)

    print("Minimum transport cost: \n",result.fun)