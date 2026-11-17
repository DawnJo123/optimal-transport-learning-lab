import numpy as np
import ot

a=np.array([0.7,0.3])
b=np.array([0.4,0.6])

C=np.array([
    [0.0,0.1],
    [1.0,0.0]
])

epsilon=1.0

#Sinkhorn 
T=ot.sinkhorn(a,b,C,reg=epsilon)

transport_cost=np.sum(T*C)

print("Transport plan:")
print(T)

print("\nRow sums:", T.sum(axis=1))
print("Required source:", a)

print("\nColumn sums:", T.sum(axis=0))
print("Required target:", b)

print("\nTransport cost:", transport_cost)
