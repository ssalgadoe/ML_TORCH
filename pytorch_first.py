import torch

from torch.autograd import Variable



dtype = torch.FloatTensor
N, d_in, H, d_out = 50, 10,100,5

x = Variable(torch.randn(N,d_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N,d_out).type(dtype), requires_grad=False)

w1 = Variable(torch.randn(d_in,H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H,d_out), requires_grad=True)


lr = 0.0001
for i in range(50):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    #print(y_pred.shape, y_pred[0], y[0])
    loss = (y-y_pred).pow(2).sum()
    print(loss.data[0])
    loss.backward()
    w1.data -= lr*w1.grad.data
    w2.data -= lr*w2.grad.data
    w1.grad.data.zero_()
    w2.grad.data.zero_()


