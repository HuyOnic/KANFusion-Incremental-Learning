import torch 
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from torchvision.datasets import CIFAR10, CIFAR100
class Capture_128(Dataset):
    def __init__(self, root, isTrain=True, transform=None):
        super(Capture_128, self).__init__()
        self.root = root
        self.transform = transform
        self.isTrain = isTrain
        self.samples, self.labels = self._get_data()

    def _get_data(self):
        data_frame = pd.read_feather(self.root) 
        samples = np.array(data_frame.iloc[:,1:-1])
        labels = np.array(data_frame.iloc[:,-1])
        samples = samples/255
        return samples, labels 
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return torch.Tensor(self.samples[index]), int(self.labels[index])
    
if __name__=="__main__":
    dataset = Capture_128('dataset/Capture_test_128.feather', isTrain=False)
    print(max(dataset.samples[:,0]))
    freqs = Counter(np.sort(dataset.labels))
    print(freqs)


