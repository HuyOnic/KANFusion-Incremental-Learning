
import torch 
from torch.nn import Module
import copy
from utils.get_model import get_model 
import os
import sys
sys.path.append(os.getcwd())
from model.KAN import KAN
from model.resnet32 import ResNet32
class BaseNet(Module):
    def __init__(self, model_name, pretrain=False) -> None:
        super().__init__()
        self.model = ResNet32() #fix_this #feature extractor layers
        self.pretrain = pretrain
        self.fc = None # ouput layer

    @property
    def feature_dim(self):
        return self.model.out_dim
    
    def extract_vector(self, x):
        return self.model(x)
    
    def forward(self, x):
        pass
    
    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
 
    def eval(self):   
        return self.train(False)

    def load_checkpoints(self, args):
        checkpoint_name = f"checkpoints/finetune_{args["incre_method"]}.pkl"
        model_info = torch.load(checkpoint_name)
        self.model.load_state_dict(model_info['model'])
        self.fc.load_state_dict(model_info['model'])
        test_acc = model_info["test_acc"]
        return test_acc
    
    def get_model(self, model_name: str):
        if model_name.lower() == "kan" or model_name.lower() == "kan":
            return KAN()
        elif model_name.lower() == "restnet32":
            return ResNet32()
        # elif model_name.lower() == "kanlinear":
        #     return KANLinear()
        # elif model_name.lower ==  "mlp":
        #     return MLP()
    
    
    


            




    



