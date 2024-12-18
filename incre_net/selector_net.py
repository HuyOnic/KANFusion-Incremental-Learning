import torch.nn as nn
import torch.nn.functional as F
import sys
import os 
os.path.append(os.getcwd())
from model.KAN import KAN

class SelectorNet(nn.Module):
    def __init__(self, num_task):
        super(SelectorNet, self).__init__()
        self.kan = KAN([128, 32, num_task])

    def forward(self, x):
        return self.kan(x)
    