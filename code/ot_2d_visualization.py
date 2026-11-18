
import numpy as np
import matplotlib.pyplot as plt

# Source point cloud
X = np.array([
    [0.0, 0.0],
    [1.0, 2.0],
    [2.0, 0.0],
    [3.0, 2.0]
])

# Target point cloud
Y = np.array([
    [1.0, 0.0],
    [2.0, 2.0],
    [3.0, 0.0],
    [4.0, 2.0]
])

# Equal probability mass
a = np.ones(len(X)) / len(X)
b = np.ones(len(Y)) / len(Y)

# Visualize the point clouds
plt.scatter(X[:, 0], X[:, 1], label="Source", marker="o")
plt.scatter(Y[:, 0], Y[:, 1], label="Target", marker="x")

plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("Source and Target Point Clouds")
plt.legend()
plt.grid(True)
plt.show()

from scipy.spatial.distance import cdist

C = cdist(X, Y, metric="euclidean")

print("Cost matrix:")
print(C)

import ot

epsilon = 0.5

T = ot.sinkhorn(a, b, C, reg=epsilon)

print("Transport plan:")
print(T)

print("\nRow sums:", T.sum(axis=1))
print("Column sums:", T.sum(axis=0))

print("\nTransport cost:", np.sum(T * C))


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

# Plot source and target points
plt.scatter(
    X[:, 0], X[:, 1],
    color="blue", marker="o", s=100, label="Source"
)

plt.scatter(
    Y[:, 0], Y[:, 1],
    color="red", marker="x", s=100, label="Target"
)

# Draw transport connections
for i in range(len(X)):
    for j in range(len(Y)):

        if T[i, j] > 0.005:
            plt.plot(
                [X[i, 0], Y[j, 0]],
                [X[i, 1], Y[j, 1]],
                color="black",
                alpha=0.6,
                linewidth=5 * T[i, j]
            )

plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("2D Optimal Transport")
plt.legend()
plt.grid(True)
plt.show()
