import numpy as np

a=np.array([0.7,0.3])
b=np.array([0.4,0.6])

C=np.array([
    [0.0,1.0],
    [1.0,0.0]
])

for epsilon in [0.5,1,1.5,5]:
    print("Epsilon: ",epsilon)

# epsilon =1

    K=np.exp(-C/epsilon)

    print("Kernal matrix: ")
    print(K)

    v=np.ones(2)


    max_iterations = 1000
    tolerance = 1e-9

    for iteration in range(max_iterations):

    # Update u: match row sums to a
        u = a / (K @ v)

    # Update v: match column sums to b
        v = b / (K.T @ u)

    # Construct transport plan
        T = np.diag(u) @ K @ np.diag(v)

    # Check marginal errors
        row_error = np.max(np.abs(T.sum(axis=1) - a))
        col_error = np.max(np.abs(T.sum(axis=0) - b))

        if max(row_error, col_error) < tolerance:
            break

    print("Iterations:", iteration + 1)
    print("\nTransport plan:")
    print(T)

    print("\nRow sums:", T.sum(axis=1))
    print("Required source:", a)

    print("\nColumn sums:", T.sum(axis=0))
    print("Required target:", b)

    print("\nRow error:", row_error)
    print("Column error:", col_error)