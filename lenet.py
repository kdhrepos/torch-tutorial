import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # input channel: 1, output channel: 6, conv: 5x5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # input channel: 6, output channel: 16, conv: 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # affine operation: y = Ax + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        # Conv layer c1: input channel 1, output channel 6
        # 5x5 conv, act func: ReLU
        # returns (N, 6, 28, 28), N is batch size
        c1 = F.relu(self.conv1(input))
        # max pooling with 2x2 kernel
        # take c1 as input
        s2 = F.max_pool2d(c1, (2, 2))
        # Conv layer c3: input channel 6, output channel 16
        # 5x5 conv, act func: ReLU
        # returns (N, 16, 10, 10), N is batch size
        c3 = F.relu(self.conv2(s2))
        # max pooling with 2x2 kernel
        # take c3 as input
        s4 = F.max_pool2d(c3, (2, 2))
        # flatten operation, returns (N, 400) tensor
        s4 = torch.flatten(s4, 1)
        # fully connected layer
        # takes (N, 400) tensor as input
        # returns (N, 120) tensor
        f5 = F.relu(self.fc1(s4))
        # fully connected layer
        # takes (N, 120) tensor as input
        # returns (N, 84) tensor
        f6 = F.relu(self.fc2(f5))
        # takes (N, 84) as input
        # returns (N, 10) tensor
        output = self.fc3(f6)
        return output

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

# parameters
params = list(net.parameters())
print(len(params))
print(params[0].size())

# dummy input
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# dummy back prop
net.zero_grad(0)
out.backward(torch.randn(1, 10))

# loss func
output = net(input)
target = torch.randn(10)  # dummy label
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# back prop - graident check
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
