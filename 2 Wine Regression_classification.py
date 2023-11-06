from sklearn.datasets import load_wine
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Load Wine dataset
wine_dataset = load_wine()
X = wine_dataset.data
y = wine_dataset.target

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

# Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create DataLoader for each dataset
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

no_features = X_train.shape[1]
no_classes = len(torch.unique(y))

print(no_features)
print(no_classes)

model = torch.nn.Sequential(
    torch.nn.Linear(no_features, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, no_classes)  # Output size should match the number of classes
)

opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification

loss_curve = []
val_loss_curve = []

for epoch in range(200):
    model.train()
    for batch_X, batch_y in train_loader:
        opt.zero_grad()
        y_hat = model(batch_X)
        loss_val = loss(y_hat, batch_y)  # Use batch_y directly as class labels
        loss_val.backward()
        opt.step()
        loss_curve.append(loss_val.item())

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X_val, batch_y_val in val_loader:
            y_hat_val = model(batch_X_val)
            val_loss += loss(y_hat_val, batch_y_val).item()
    val_loss /= len(val_loader)
    val_loss_curve.append(val_loss)

    print(f"Epoch [{epoch + 1}/100], Training Loss: {loss_curve[-1]:.4f}, Validation Loss: {val_loss:.4f}")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_curve)
plt.xlabel('Iterations')
plt.ylabel('Training Loss')
plt.title('Training Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(val_loss_curve, color='orange')
plt.xlabel('Iterations')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Curve')

plt.tight_layout()
plt.show()

# Testing
model.eval()
test_loss = 0
correct_predictions = 0
total_samples = 0

with torch.no_grad():
    for batch_X_test, batch_y_test in test_loader:
        y_hat_test = model(batch_X_test)
        test_loss += loss(y_hat_test, batch_y_test).item()

        # Calculate accuracy
        predicted_labels = y_hat_test.argmax(dim=1)  # Get the index of the class with highest probability
        correct_predictions += (predicted_labels == batch_y_test).sum().item()
        total_samples += batch_y_test.size(0)

test_loss /= len(test_loader)
accuracy = correct_predictions / total_samples * 100

print(f"Testing Loss: {test_loss:.4f}")
print(f"Testing Accuracy: {accuracy:.2f}%")
