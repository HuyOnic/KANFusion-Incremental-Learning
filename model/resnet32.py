import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
import os 
sys.path.append(os.getcwd())
from model.KAN import KAN as KAN
# Basic Residual Block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        out = F.relu(out)
        return out

# ResNet-32 Model
class ResNet32(nn.Module):
    def __init__(self):
        super(ResNet32, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Residual layers
        self.layer1 = self._make_layer(16, 16, 5, stride=1)  # 5 blocks in layer 1
        self.layer2 = self._make_layer(16, 32, 5, stride=2)  # 5 blocks in layer 2
        self.layer3 = self._make_layer(32, 64, 5, stride=2)  # 5 blocks in layer 3
        
        # Fully connected layers
        # self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # Initial convolution
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)  # Global Average Pooling
        out = out.view(out.size(0), -1)  # Flatten the output
        return out

if __name__ == "__main__":
# Create the model
    x = torch.randn(8, 3, 32, 32)
    model = ResNet32(num_classes=100)  # For CIFAR-100
    out = model(x)
    print(out.size())
