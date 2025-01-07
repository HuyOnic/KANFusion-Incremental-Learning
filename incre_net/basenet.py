
import torch 
from torch.nn import Module
import copy
import os
import sys
sys.path.append(os.getcwd())
from model.KAN import KAN
from model.resnet32 import ResNet32
class BaseNet(Module):
    def __init__(self, model_name, pretrain=False) -> None:
        super().__init__()
        self.pretrain = pretrain
        self.fc = None 
    
    def extract_vector(self, x):
        return self.model(x)
    
    def forward(self, x):
        pass
    
    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def get_model(self, model_name: str):
        if model_name.lower() == "kan" or model_name.lower() == "kan":
            return KAN()
        elif model_name.lower() == "restnet32":
            return ResNet32()
        # elif model_name.lower() == "kanlinear":
        #     return KANLinear()
        # elif model_name.lower ==  "mlp":
        #     return MLP()
    
    
    


            




    



