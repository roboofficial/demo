import numpy as np
import matplotlib.pyplot as plt

states = ["A", "B", "C"]

P = np.array([
    [0.2, 0.6, 0.2],
    [0.3, 0.0, 0.7],
    [0.5, 0.3, 0.2]  
])

def power_method(P, num_iterations=1000):
    n = P.shape[0]
    v = np.random.rand(n)
    v /= np.sum(v)  # Normalize to get a probability distribution

    for _ in range(num_iterations):
        v = np.dot(v, P)

    return v
def power_method_2(P, tol=1e-8, max_iter=1000):
    n = P.shape[0]
    v = np.random.rand(n)
    v /= np.sum(v)

    for _ in range(max_iter):
        v_next = np.dot(v, P)
        if np.linalg.norm(v_next - v) < tol:
            break
        v = v_next

    return v

def algebraic_method(P):
    n = P.shape[0]
    
    A = P.T - np.eye(n)
    A[-1] = np.ones(n)  # replace last row with normalization condition
    
    b = np.zeros(n)
    b[-1] = 1
    
    pi = np.linalg.solve(A, b)
    return pi

def eigenvector_method(P):
    eigvals, eigvecs = np.linalg.eig(P.T)
    stationary_vector = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    stationary_vector /= np.sum(stationary_vector)  # Normalize
    return stationary_vector.flatten()

import numpy as np
import matplotlib.pyplot as plt

def plot_stationary_distributions(power, algebraic, eigen, states=["A","B","C"]):
    power = np.array(power)
    algebraic = np.array(algebraic)
    eigen = np.array(eigen)

    x = np.arange(len(states))
    width = 0.25

    plt.figure()

    plt.bar(x - width, power, width, label="Power Method")
    plt.bar(x, algebraic, width, label="Algebraic Method")
    plt.bar(x + width, eigen, width, label="Eigenvector Method")

    plt.xticks(x, states)
    plt.xlabel("States")
    plt.ylabel("Probability")
    plt.title("Comparison of Stationary Distributions")
    plt.legend()

    plt.show()

stationary_distribution_power = power_method(P)
stationary_distribution_algebraic = algebraic_method(P)
stationary_distribution_eigenvector = eigenvector_method(P)

print("Stationary Distribution (Power Method):", stationary_distribution_power)
print("Stationary Distribution (Algebraic Method):", stationary_distribution_algebraic)
print("Stationary Distribution (Eigenvector Method):", stationary_distribution_eigenvector)
plot_stationary_distributions(stationary_distribution_power, stationary_distribution_algebraic, stationary_distribution_eigenvector)
