import random
import math
import matplotlib.pyplot as plt
import numpy as np

def get_neighbor(point):
    new_point = point + random.uniform(-0.1, 0.1)
    return max(-3, min(3, new_point))

def simulated_annealing(T_min=1e-6, alpha=0.99, T=1000):
    #initial point = 0
    current_point = random.uniform(-3, 3)
    current_cost = objective_function(current_point)
    best_point = current_point
    best_cost = current_cost

    while T > T_min:
        for i in range(100):
            new_point = get_neighbor(current_point)
            new_cost = objective_function(new_point)
            delta = new_cost - current_cost

            if delta < 0 or random.random() < math.exp(-delta / T):
                current_point = new_point
                current_cost = new_cost
                if current_cost < best_cost:
                    best_point = current_point
                    best_cost = current_cost
        T *= alpha

    return best_point, best_cost

def objective_function(x):
    return (x**3 - 2* x**2 - 10*x + 10)

point, cost = simulated_annealing()

def plot_graph():
    x= np.linspace(-3,3,100)
    y = objective_function(x)
    plt.plot(x,y)
    plt.scatter(point, cost, color='red')
    plt.show()

print("Best Point:", point)
print("Best Cost:", cost)
plot_graph()