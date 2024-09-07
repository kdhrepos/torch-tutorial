import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  # y = xA^T + b
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
# print(model)

# Softmax
X = torch.rand(1, 28, 28, device=device)
# print(f"Input: {X}")
logits = model(X)
pred_prob = nn.Softmax(dim=1)(logits)
y_pred = pred_prob.argmax(1)
# print(f"Predicted class {y_pred}")

# example image
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# make dimension flat
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# linear layer
layer = nn.Linear(in_features=28*28, out_features=20)
hidden = layer(flat_image)
print(hidden.size())

# ReLU
print(f"Before ReLU: {hidden}")
hidden = nn.ReLU()(hidden)
print(f"After ReLU: {hidden}")

# Sequential
seq_modules = nn.Sequential(
    flatten,
    layer,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)
print(f"Logits: {logits}")

# Softmax
softmax = nn.Softmax(dim=1)  # makes the sum of probs as "1"
pred_prob = softmax(logits)
print(f"Softmax: {pred_prob}")

# Params of nn model
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
