import numpy as np
from sklearn.datasets import load_wine

# Load the wine dataset
wine = load_wine()
X = wine.data  # Input features
y = wine.target  # Target variable

# Normalize the input features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Add bias term to X
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Initialize hyperprior parameters
a = 1.0
b = 1.0
c = 1.0
d = 1.0

# Initialize variables
N, M = X.shape
sigma_beta = np.ones(M)
sigma_y = 1.0

# EM algorithm
converged = False
while not converged:
    # E-step
    Sigma_beta = np.diag(sigma_beta)
    C_beta = np.linalg.inv((1 / sigma_y) * X.T @ X + np.linalg.inv(Sigma_beta))

    mu_beta = 1 / sigma_y * C_beta @ X.T @ y

    # Update parameters
    sigma_beta_new = (2 * b + mu_beta ** 2 + np.diag(C_beta)) / (2 * a + 1)
    sigma_y_new = (2 * d + np.linalg.norm(y - X @ mu_beta) ** 2 + np.trace(X @ C_beta @ X.T)) / (2 * c + N)

    # Check for convergence
    if np.all(np.abs(sigma_beta_new - sigma_beta) < 1e-6) and np.abs(sigma_y_new - sigma_y) < 1e-6:
        converged = True

    # Update variables
    sigma_beta = sigma_beta_new
    sigma_y = sigma_y_new

# Model prediction
y_hat = X @ mu_beta

# Print the estimated coefficients
print("Estimated Coefficients:")
for i, coef in enumerate(mu_beta[1:]):
    print(f"β{i+1}: {coef}")

# Print mean and covariance of β
print("Mean of β:")
print(mu_beta)

print("Covariance of β:")
print(C_beta)

# Print predicted values
print("Predicted values (y_hat):")
print(y_hat)

print("Actual y values:")
print(y)
