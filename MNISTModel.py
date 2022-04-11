import torch.nn as nn
import torch


class Feedforward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 32)
        self.fc2 = nn.Linear(32, 10)
        self.sigmoid = nn.Sigmoid()

        # xavier_(self.fc1.parameters)
        # xavier_(self.fc2.parameters)
        # self.register_parameters([self.fc1, self.fc2])

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.sigmoid(z1)
        z2 = self.fc2(a1)
        return z2
