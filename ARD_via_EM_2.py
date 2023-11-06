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

# Create a model instance
class LinearRegressionModel:
    def __init__(self, weights):
        self.weights = np.insert(weights, 0, 0)  # Insert a zero for the bias term

    def predict(self, X):
        return X @ self.weights

# Create multiple model instances from the learned distribution
num_samples = 10
model_instances = []
for _ in range(num_samples):
    sample_weights = np.random.multivariate_normal(mu_beta, C_beta)
    model = LinearRegressionModel(sample_weights[1:])  # Exclude the bias term
    model_instances.append(model)

# Compare predictions of the model instances
losses = []
for i, model in enumerate(model_instances):
    y_pred = model.predict(X)
    loss = np.mean((y_pred - y) ** 2)
    losses.append(loss)
    print(f"Model {i+1} loss: {loss}")

# Print the model with the lowest loss
min_loss_index = np.argmin(losses)
min_loss_model = model_instances[min_loss_index]
print(f"\nModel with the lowest loss (Model {min_loss_index+1})")
