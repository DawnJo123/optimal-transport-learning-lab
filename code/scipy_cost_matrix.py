import numpy as np
from scipy.spatial.distance import cdist

#source points
X=np.array([[0.0],[2.0]])

#Target points
Y=np.array([[1.0],[3.0]])

#Probability distribution
a=np.array([0.5,0.5])
b=np.array([0.5,0.5])

#Euclidian distance
C=cdist(X,Y,metric="euclidean")

print("Source points:", X)
print("Target points: ", Y)

print("Source distribution: ",a)
print("Target distribution: ", b)

print("Cost matrix: ", C)