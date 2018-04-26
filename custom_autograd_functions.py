import torch
from torch.autograd import Variable


class MyRelu(torch.autograd.Function):
    def forward(self, input):
        self.save_for_backward(input)
        return input.clamp(min=0)

    def backward(self, grad_out):
        grad_input = grad_out.clone()
        input, = self.saved_tensors
        grad_input[input < 0] = 0
        return grad_input



dtype = torch.FloatTensor
N, d_in, H, d_out = 50, 10,100,5

x = Variable(torch.randn(N,d_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N,d_out).type(dtype), requires_grad=False)

w1 = Variable(torch.randn(d_in,H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H,d_out), requires_grad=True)


lr = 0.0001

for i in range(50):
    relu = MyRelu()
    y_pred = relu(x.mm(w1)).mm(w2)
    #print(y_pred.shape, y_pred[0], y[0])
    loss = (y-y_pred).pow(2).sum()
    print('l',loss.data[0])
    loss.backward()
    w1.data -= lr*w1.grad.data
    w2.data -= lr*w2.grad.data
    w1.grad.data.zero_()
    w2.grad.data.zero_()
print(loss)