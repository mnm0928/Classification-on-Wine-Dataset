import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Load Wine dataset
wine_dataset = load_wine()
X = wine_dataset.data
y = wine_dataset.target

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)  # Change the data type to long for class labels

# Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create DataLoader for each dataset
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

no_features = X_train.shape[1]
no_classes = len(torch.unique(y))

print(no_features)
print(no_classes)

model = torch.nn.Sequential(
    torch.nn.Linear(no_features, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, no_classes)
)

opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss = torch.nn.CrossEntropyLoss()

# Set up TensorBoard writer with modified log directory name
log_dir = "logs/" + datetime.now().strftime("%m.%d-%H_%M_%S")
writer = SummaryWriter(log_dir)

loss_curve = []
val_loss_curve = []

for epoch in range(200):
    model.train()
    epoch_loss = 0.0
    epoch_accuracy = 0.0

    for batch_X, batch_y in train_loader:
        opt.zero_grad()
        y_hat = model(batch_X)
        loss_val = loss(y_hat, batch_y)
        loss_val.backward()
        opt.step()

        # Calculate accuracy
        predicted_labels = y_hat.argmax(dim=1)
        epoch_accuracy += torch.sum(predicted_labels == batch_y).item()
        epoch_loss += loss_val.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    avg_epoch_accuracy = epoch_accuracy / len(train_loader) / batch_size

    # Log metrics to TensorBoard
    writer.add_scalar("epoch_loss", avg_epoch_loss, epoch)
    writer.add_scalar("epoch_accuracy", avg_epoch_accuracy, epoch)
    writer.flush()

    print(f"Epoch [{epoch + 1}/200], Training Loss: {avg_epoch_loss:.4f}, Training Accuracy: {avg_epoch_accuracy:.2%}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X_val, batch_y_val in val_loader:
            y_hat_val = model(batch_X_val)
            val_loss += loss(y_hat_val, batch_y_val).item()
    avg_val_loss = val_loss / len(val_loader)
    val_loss_curve.append(avg_val_loss)

    print(f"Validation Loss: {avg_val_loss:.4f}")

# ... (Testing and final metrics)

# Close the TensorBoard writer
writer.close()
