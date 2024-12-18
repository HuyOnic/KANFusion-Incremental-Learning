import numpy as np 
import torch 
from torch import nn
import math
import sys
import os
sys.path.append(os.getcwd())
from model.KAN import KAN 
from model.MLP import MLP
from model.resnet32 import ResNet32
from model.KANLinear import KANLinear

def get_model(model_name: str):
    if model_name.lower() == "kan" or model_name.lower() == "kan":
        return KAN()
    elif model_name.lower() == "kanlinear":
        return KANLinear()
    elif model_name.lower() ==  "mlp":
        return MLP()
    elif model_name.lower() == "resnet32":
        return ResNet32()

def confusion_matrix():
    pass



if __name__=="__main__":
    y_pred = np.array([1,2,3,1,2,3,4,5,7,5])
    y_true = np.array([5,8,5,0,2,3,4,5,6,5])
    increment = 2
    num_task = math.ceil((max(y_true)+1)/increment)
    print(num_task)
