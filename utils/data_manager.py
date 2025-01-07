import sys
import os
import numpy as np
import torch
from collections import Counter
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
sys.path.append(os.getcwd())
from utils.dataset import Capture_128


class DataManager(object):
    def __init__(self, dataset_name, shuffle, init_cls, increment, seed) -> None:
        self.dataset_name = dataset_name
        self.shuffle = shuffle
        self.seed = seed
        self.train_dataset, self.test_dataset, self._class_order = self._setup_data()
        self.increments = [init_cls] #store number of classes for each task
        #Linear Increment
        while sum(self.increments) + increment < len(self._class_order):
            self.increments.append(increment)
        offset = len(self._class_order) - sum(self.increments)
        if offset > 0:
            self.increments.append(offset)
        print("Data Manager Created Successfully!")
    
    @property
    def num_task(self):
        return len(self.increments)
    
    def get_task_size(self,task_id):
        return self.increments[task_id]
    
    def get_total_num_classes(self):
        return len(self._class_order)
    
    def _setup_data(self):
        name = self.dataset_name.lower()
        transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                                std=[0.247, 0.243, 0.261])
        ])

        if name == "capture_128":
            train_dataset = Capture_128(root="dataset/Capture_train_128.feather", isTrain=False)
            test_dataset =  Capture_128(root="dataset/Capture_test_128.feather", isTrain=False)
        elif name == "cifar10":
            train_dataset = CIFAR10(root="./dataset", train=True, transform=transformations)
            test_dataset =  CIFAR10(root="./dataset", train=False, transform=transformations)
        elif name == "cifar100":
            train_dataset = CIFAR100(root="./dataset", train=True, transform=transformations, download=True)
            test_dataset =  CIFAR100(root="./dataset", train=False, transform=transformations, download=True)
        elif name == "mnist":
            train_dataset = MNIST(root="./dataset", train=True, transform=transformations, download=True)
            test_dataset =  MNIST(root="./dataset", train=False, transform=transformations, download=True)
        labels = sorted([data[1] for data in train_dataset])
        order = [label for label in range(len(np.unique(labels)))]
        if self.shuffle:
            np.random.seed(self.seed)
            order = np.random.permutation(len(order)).tolist()
        print("Class Order: ",order)
        return train_dataset, test_dataset, order
    
    def get_data(self, num_known_classes, total_classes, isTrain, appendent=None): 
        all_classes_of_task = self._class_order[num_known_classes:total_classes]
        if isTrain:
            curr_dataset = self.train_dataset
        else: 
            curr_dataset = self.test_dataset
        indices = [i for i, (_, label) in enumerate(curr_dataset) if label in all_classes_of_task]
        data_of_task = [curr_dataset[i][0] for i in indices]
        labels_of_task = [curr_dataset[i][1] for i in indices]
        if appendent is not None:
            for sample, label in appendent:
                data_of_task.append(sample)
                labels_of_task.append(label)
        return DummyDataset(data_of_task, labels_of_task)

class DummyDataset(Dataset): 
    def __init__(self, samples, labels) -> None:
        assert len(samples) == len(labels), "Data size error!"
        self.samples = samples
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

if __name__=="__main__":
    data_manager = DataManager(dataset_name="capture_128", shuffle=True, init_cls=2, increment=2, seed=2024)
    _, _, dataloader = data_manager.get_data_of_task(0,2, isTrain=False)
    print(dataloader[0])


