import torch.nn as nn
import torch.nn.functional as F
import sys
import os 
os.path.append(os.getcwd())
from model.KAN import KAN
from model.resnet32 import ResNet32
from incre_net.basenet import BaseNet

class SelectorNet(nn.Module):
    def __init__(self, num_task):
        super(SelectorNet, self).__init__()
        self.kan = KAN([128, 32, num_task])

    def forward(self, x):
        if len(x.size())>2:
            x = x.view(x.size(0), -1)
        return self.kan(x)
    
class NodeModel(BaseNet):
    def __init__(self, model_name, pretrain=False):
        super().__init__(model_name, pretrain)
        self.model = ResNet32()

    def update_fc(self, num_classes):
        del self.fc 
        self.fc = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.model(x)
        out = self.fc(x)
        return out
    