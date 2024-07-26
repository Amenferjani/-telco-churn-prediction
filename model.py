import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, inputData, h1=46, h2=46, outData=1):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(inputData,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,outData)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

