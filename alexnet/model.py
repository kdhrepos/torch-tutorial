import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(AlexNet, self).__init__()
        # Conv layer
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(
                in_channels=256, out_channels=384, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=384, out_channels=384, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=384, out_channels=256, kernel_size=3),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # Fully conntected layer
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=9216, out_features=4096),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(True),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, input):
        out = self.conv(input)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
