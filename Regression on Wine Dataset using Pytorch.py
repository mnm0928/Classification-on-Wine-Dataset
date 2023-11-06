from sklearn.datasets import load_wine
import torch
import matplotlib.pyplot as plt

# Load Wine dataset
wine_dataset = load_wine()
X = wine_dataset.data
y = wine_dataset.target

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

y = torch.unsqueeze(y, dim=1)

print("shape of X: ", X.shape)
print("shape of y: ", y.shape)

no_features = X.shape[1]

model = torch.nn.Sequential(
    torch.nn.Linear(no_features, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 8),
    torch.nn.Tanh(),
    torch.nn.Linear(8, 1)
)


opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss = torch.nn.MSELoss()

loss_curve = []

for epoch in range(1000):

    opt.zero_grad()

    y_hat = model(X)
    print(y_hat.shape)
    loss_val = loss(y_hat, y)
    loss_val.backward()
    opt.step()
    loss_curve.append(loss_val.detach())


plt.plot(loss_curve)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.show()
