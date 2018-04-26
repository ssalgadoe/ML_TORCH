import torch
import numpy as np
from torch.autograd import Variable
from torch import Tensor as T
from sklearn.datasets import load_wine



dtype = torch.FloatTensor
N, d_in, H, d_out = 50, 10,100,5

x = Variable(torch.randn(N,d_in), requires_grad=False)
y = Variable(torch.randn(N,d_out), requires_grad=False)

model = torch.nn.Sequential(
    torch.nn.Linear(d_in,H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,d_out)
)


loss_fn = torch.nn.MSELoss(size_average=False)

lr = 1e-4
for i in range(100):
    y_pred = model(x)
    loss = loss_fn(y_pred,y)
    print(i, loss.data[0])
    model.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.data -= lr * param.grad.data
