from torch.nn import Module
from torch import nn 
import os
import torch
import sys
sys.path.append(os.getcwd())
from model.KAN import KAN
class CKAN(Module):
    def __init__(self, num_class):
        super(CKAN, self).__init__()
        self.num_class = num_class
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = KAN([4096, 128, num_class])
    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.pool(x) 
        x = self.conv2(x)
        x = self.act(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) #8*8*64
        x = self.fc(x)
        return x
if __name__ == "__main__":
# Create the model
    x = torch.randn(8, 3, 32, 32)
    model = CKAN(num_class=100)  # For CIFAR-100
    out = model(x)
    print(out.size())