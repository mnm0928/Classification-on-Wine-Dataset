import torch
import matplotlib.pyplot as plt


X = torch.arange(0.0, 1.0, step=0.01)
X2 = torch.randint(2, size=(len(X),))
Y = (X * 0.7 + X2 * 0.2 - 0.3) + torch.normal(0, 0.1, size=(len(X),))

print(X2)


a = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)
c = torch.rand(1, requires_grad=True)
print(a)

"""
loss_curve = []

opt = torch.optim.SGD([a, b, c], lr=0.02)
loss = torch.nn.MSELoss()

for epoch in range(50):
    opt.zero_grad()
    y_hat = X * a + X2 * b + c
    loss_val = loss(y_hat, Y)
    loss_val.backward()
    opt.step()
    loss_curve.append(loss_val.detach())

plt.plot(loss_curve)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

"""