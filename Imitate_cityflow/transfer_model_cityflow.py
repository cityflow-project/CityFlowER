import os
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = "cuda" #NVIDI A GPU
elif torch.backends.mps.is_available():
    device = "mps" #Apple GPU
else:
    device = "cpu"


class LaneChangeNetwork(nn.Module):
    def __init__(self):
        super(LaneChangeNetwork, self).__init__()
        # Layers for feature extraction and classification
        self.layers = nn.Sequential(
            nn.Linear(17, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Output layer for 3 classes
        )

    def forward(self, x):
        # Forward pass through the network
        return self.layers(x)   # Apply softmax to output



class RegressionModelBKerner(nn.Module):
    def __init__(self):
        super(RegressionModelBKerner, self).__init__()
        self.fc1 = nn.Linear(4, 50)  # 10 input features
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)   # Output one value: speed

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x) 
        return x


class RegressionModelIDM(nn.Module):
    def __init__(self):
        super(RegressionModelIDM, self).__init__()
        self.fc1 = nn.Linear(4, 50)  # 10 input features
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)   # Output one value: speed

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x) 
        return x


model_load = RegressionModelIDM()



model_load.load_state_dict(torch.load("PATH.pth"))

#save as torch script
model_scripted = torch.jit.script(model_load)
model_scripted.save('PATH.pt')
