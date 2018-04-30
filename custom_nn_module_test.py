import torch
from torch.autograd import Variable
from torch import Tensor as T
import numpy as np
from sklearn.datasets import load_wine


class myNN(torch.nn.Module):
    def __init__(self, d_in, H, d_out):
        super(myNN, self).__init__()
        self.linear1 = torch.nn.Linear(d_in, H)
        self.linear2 = torch.nn.Linear(H, d_out)

    def forward(self, x):
        relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(relu)

        return (y_pred)


N, d_in, H, d_out = 50, 10, 100, 5
x = Variable(torch.randn(N, d_in), requires_grad=False)
y = Variable(torch.randn(N, d_out), requires_grad=False)

loss_fn = torch.nn.MSELoss(size_average=False)

model = myNN(d_in, H, d_out)
lr = 1e-3
for t in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(loss.data[0])
    model.zero_grad()
    loss.backward()
    for par in model.parameters():
        par.data -= lr * par.grad.data

