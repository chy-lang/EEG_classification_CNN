import torch
from InceptionA_model import InceptionA
import torch.nn.functional as F
class ComplexConvNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(ComplexConvNeuralNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1)
        self.incep1 = InceptionA(in_channel=10)
        self.conv2 = torch.nn.Conv2d(in_channels=88, out_channels=20, kernel_size=3, padding=1)
        self.incep2 = InceptionA(in_channel=20)
        self.pool = torch.nn.MaxPool2d(2)
        self.linear1 = torch.nn.Linear(in_features=88 * 4 * 4, out_features=128)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=3)
        self.conv3 = torch.nn.Conv2d(in_channels=88, out_channels=88, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size = 32
        x = x.view(batch_size, 1, 8, 8)
        x = F.relu(self.conv1(x))
        x = self.incep1(x)
        x1 = x
        x = F.relu(self.conv2(x))
        x = self.incep2(x)
        x = F.relu(x + x1)
        x = self.pool(self.conv3(x))
        x = x.view(batch_size, -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
