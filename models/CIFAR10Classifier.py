import torch
import torch.nn as nn

class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Flatten image dimensions
        self.flatten = nn.Flatten()

        # First hidden layer
        self.fc1 = nn.Linear(3 * 32 * 32, 128)
        self.relu1 = nn.ReLU()

        # Second hidden layer
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

        # Output layer
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x