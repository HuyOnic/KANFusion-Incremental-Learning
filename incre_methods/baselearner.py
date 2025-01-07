import os 
import sys
sys.path.append(os.getcwd())
import torch 
from torch.nn import Module
import numpy as np
import copy 
from torch import nn
from torch.utils.data import DataLoader
from utils.data_manager import DummyDataset
EPSILON=1e-8
class BaseIncremnetalMethod(object):
    def __init__(self, args) -> None:
        self.args = args
        self._cur_task = -1
        self._know_class = 0
        self._total_class = 0
        self._samples_memory, self._labels_memory =[], []
        self.workers = args["num_workers"]
        self._memory_size = args["memory_size"]
        self._device = args["device"]
    @property
    def examplar_size(self):
        assert len(self._samples_memory) == len(self._labels_memory), "Size Error!"
        return len(self._labels_memory)

    @property
    def memory(self):
        if len(self._samples_memory) == 0:
            return None
        else:
            return (self._samples_memory, self._labels_memory)
    
    def save_checkpoint(self, test_acc): ###
        checkpoint_name = f'exps/checkpoints/pretrained_{self.args["csv_name"]}'
        _checkpoint_cpu = copy.deepcopy(self._network)
        if isinstance(_checkpoint_cpu, nn.DataParallel):
            _checkpoint_cpu = _checkpoint_cpu.module
        _checkpoint_cpu.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "convnet": _checkpoint_cpu.convnet.state_dict(),
            "fc":_checkpoint_cpu.fc.state_dict(),
            "test_acc": test_acc
        }
        torch.save(save_dict, "{}_{}.pt".format(checkpoint_name, self._cur_task))
    
    def after_task(self):
        self._know_class = self._total_class
        print(f'Examplar size: {self.examplar_size}')

    def eval_task(self, data_manager, save_conf=False):
        cls_order = data_manager._class_order[:self._total_class]
        all_acc = {}
        #Evaluate on all seen classes
        for cls in cls_order:
            _ , _, test_dataset = data_manager.get_data_of_task( 
                                num_known_classes = cls, 
                                total_classes = cls+1,
                                isTrain = False,
                                appendent = None
                        ) 
            test_loader = DataLoader(
                        test_dataset, 
                        batch_size=self.args["batch_size"], 
                        shuffle=False, 
                        )
            #Classify
            correct = 0
            num_samples = 0
            for batch_idx, (samples, labels) in enumerate(test_loader):
                pred = self._incre_net(samples)
                pred_value = torch.argmax(pred, dim=1)
                correct += (pred_value==labels).sum().item()
                num_samples+=len(labels)
            all_acc[cls]=correct/num_samples
        all_acc['avg'] = sum([all_acc[cls] for cls in all_acc.keys()])/self._total_class
        return all_acc
    
    def building_examplar(self):
        pass


    
    
                

