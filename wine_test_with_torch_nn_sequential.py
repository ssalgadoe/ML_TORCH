import torch
import numpy as np
from torch.autograd import Variable
from torch import Tensor as T
from sklearn.datasets import load_wine

features, labels  = load_wine(return_X_y=True)

x = Variable(T(features), requires_grad=False)
y = Variable(T(labels), requires_grad=False)

N = len(labels)
d_in = len(features[0])
H = 100
d_out = 1

model = torch.nn.Sequential(
    torch.nn.Linear(d_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, d_out)
)

loss_fn = torch.nn.MSELoss(size_average=False)

lr = 1e-8
for i in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(i, loss.data[0])
    model.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.data -= lr * param.grad.data

print(loss.data[0])


