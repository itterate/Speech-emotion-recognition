import torch.nn as nn
import torch

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 256, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, stride=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x
