import os 
import sys 
sys.path.append(os.getcwd())
from incre_net.basenet import BaseNet
from torch import nn
from model.resnet32 import ResNet32

class NodeModel(BaseNet):
    def __init__(self, model_name, pretrain=False):
        super().__init__(model_name, pretrain)
        self.model = ResNet32()

    def reset_paramerters(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def update_fc(self, num_classes):
        del self.fc 
        self.fc = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.model(x)
        out = self.fc(x)
        return out



    
    
