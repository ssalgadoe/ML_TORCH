import torch
import numpy as np
from torch.autograd import Variable
from torch import Tensor as T
from sklearn.datasets import load_iris


features, target = load_iris(return_X_y=True)


N = len(target)
d_in = len(features[0])
d_out = 1
H = 100

x = Variable(T(features), requires_grad=False)
y = Variable(T(target), requires_grad=False)



model = torch.nn.Sequential(
        torch.nn.Linear(d_in,H),
        torch.nn.ReLU(),
        torch.nn.Linear(H,d_out)

)
loss_fn = torch.nn.MSELoss(size_average=False)

lr = 1e-4
for i in range(3000):
    y_pred = model(x)
    loss= loss_fn(y_pred, y)
    print(loss.data[0])
    model.zero_grad()
    loss.backward()
    for par in model.parameters():
        par.data -= lr*par.grad.data