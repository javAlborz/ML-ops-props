# Import necessary packages

import numpy as np
import torch

import helper

import matplotlib.pyplot as plt

from torchvision import datasets, transforms



## Your solution here

import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hidden1 = nn.Linear(784, 128)
        
        self.hidden2 = nn.Linear(128, 64)

        self.output = nn.Linear(64, 10)
        
    def forward(self, x):
        x = F.ReLU(self.hidden1(x))
        
        x = F.ReLU(self.hidden2(x))
        
        x = F.softmax(self.output(x), dim=1)
        
        return x



# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)

# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])
helper.view_classify(images[0].view(1, 28, 28), ps)

