import os 
import sys 
import copy
sys.path.append(os.getcwd())
from incre_net.basenet import BaseNet
from model.KANLinear import KANLinear
from torch import nn


class IncrementalNet(BaseNet):
    def __init__(self, model_name, pretrain=False):
        super().__init__(model_name, pretrain)

    def update_fc(self, num_classes):
        new_fc = self.generate_fc(out_dim=num_classes)
        if self.fc is not None:
            num_outputs = self.fc.out_features 
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            new_fc.weight.data[:num_outputs] = weight
            new_fc.bias.data[:num_outputs] = bias
        del self.fc 
        self.fc = new_fc

    def generate_fc(self, out_dim, in_dim=64):
        fc = nn.Linear(in_dim, out_dim)
        return fc
    
    # def unfreeze(self):
    #     for param in self.parameters():
    #         param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        out = self.fc(x)
        return out

class IncrementalKANNet(BaseNet):
    def __init__(self, model_name, pretrain=False):
        super().__init__(model_name, pretrain)

    def update_fc(self, num_classes):
        new_fc = self.generate_fc(out_dim=num_classes)
        if self.fc is not None:
            num_outputs = self.fc.out_features 
            base_weight = copy.deepcopy(self.fc.base_weight.data)
            spline_weight = copy.deepcopy(self.fc.scaled_spline_weight.data)
            new_fc.base_weight.data[:num_outputs] = base_weight
            new_fc.spline_weight.data[:num_outputs] = spline_weight
        del self.fc 
        self.fc = new_fc

    def generate_fc(self, out_dim, in_dim=32):
        fc = KANLinear(in_dim, out_dim)
        return fc
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        out = self.fc(x)
        return out

    
    
