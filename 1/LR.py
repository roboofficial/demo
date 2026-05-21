import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def generate_synthetic_data(n_points=50, slope=2.5, intercept=10, noise_std=5, random_seed=42):
    """Generate synthetic data with controlled Gaussian noise"""
    np.random.seed(random_seed)
    x = np.linspace(0, 20, n_points)
    noise = np.random.normal(0, noise_std, n_points)
    y = slope * x + intercept + noise   
    return x, y

def fit_linear_regression(x, y):
    """Calculate the slope (m) and intercept (c) using the OLS formula"""
    n = len(x)
    mean_x, mean_y = np.mean(x), np.mean(y)
    m = np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x) ** 2)
    c = mean_y - m * mean_x
    return m, c

def plot_data_and_fit(x, y, m, c):
    """Visualize the original data points and the fitted line"""
    plt.scatter(x, y, color='blue', label='Data Points')
    y_pred = m * x + c
    plt.plot(x, y_pred, color='red', label=f'Fitted Line: y = {m:.2f}x + {c:.2f}')
    plt.xlabel('Independent Variable (x)')
    plt.ylabel('Dependent Variable (y)')
    plt.title('Simple Linear Regression Fit')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # Step 1: Generate synthetic data
    x, y = generate_synthetic_data()

    # Step 2: Fit linear regression
    m, c = fit_linear_regression(x, y)

    # Step 3: Plot data and fitted line
    plot_data_and_fit(x, y, m, c)

    # Step 4: Evaluate goodness of fit
    y_pred = m * x + c
    r2 = r2_score(y, y_pred)
    print(f"Slope (m): {m:.2f}, Intercept (c): {c:.2f}, R² Score: {r2:.4f}")   

if __name__ == "__main__":
    main()