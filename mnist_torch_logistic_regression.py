import torch
import numpy as np
from torch.autograd import Variable
from torch import Tensor as T
from sklearn.datasets import load_boston
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import dataset, dataloader


class LinearModel(torch.nn.Module):
    def __init__(self, nr_features, nr_classes):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(nr_features, nr_classes)

    def forward(self, x):
        out = self.linear(x)
        return (out)



batch_sz = 100
learning_rate = 1e-3
nr_epochs = 2
nr_features = 784
nr_classes = 10

train_set = dsets.MNIST('../ML_SCIKIT/data', train=True, transform=transforms.ToTensor())
test_set = dsets.MNIST('../ML_SCIKIT/data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_sz, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_sz, shuffle=False)


model = LinearModel(nr_features, nr_classes)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for ep in range(nr_epochs):
    for i, (img, lbl) in enumerate(train_loader):
        images = Variable(img.view(-1, 28 * 28))
        labels = Variable(lbl)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                  % (ep + 1, nr_epochs, i + 1, len(train_set) // batch_sz, loss.data[0]))

total = 0
correct = 0

for img, labels in test_loader:
    images = Variable(img.view(-1, 28 * 28))
    result = model(images)
    _, prediction = torch.max(result.data, 1)

    total += labels.size(0)
    correct += (prediction == labels).sum()

print("accuracy %d" % (100 * correct / total))