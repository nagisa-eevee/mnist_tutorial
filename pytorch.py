import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
# from matplotlib import pyplot as plt
# import math
import time

BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.01

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# download and load the data
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)


class SimpleNet(nn.Module):
    # TODO:define model
    def __init__(self):
        super(SimpleNet, self).__init__()
        # input size 28*28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # self.dropout1 = nn.Dropout2d(p=0.25)
        # self.dropout2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(in_features=64*12*12, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        # x = self.dropout2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


model = SimpleNet()

# TODO:define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# train and evaluate
# n = 0
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_loader)):
        # TODO:forward + backward + optimize
        images, labels = data
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 117 == 116:
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/117))
            # n += 1
            # plt.scatter(n, math.log10(running_loss/117), color='r', marker='.')
            running_loss = 0.0
# plt.show()

# evaluate
# TODO:calculate the accuracy using traning and testing dataset
model.eval()

correct = 0
total = 0
print_signal = True
with torch.no_grad():
    for images, labels in tqdm(train_loader):
        output = model(images)
        # takes the indices as the prediction and discards it's value
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy on the training set: %.2f %%' % (100*correct/total))

correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy on the testing set: %.2f %%' % (100*correct/total))

