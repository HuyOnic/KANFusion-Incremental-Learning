import os 
import sys
sys.path.append(os.getcwd())
import torch 
from torch.nn import Module
import numpy as np
import copy 
from torch import nn
from torch.utils.data import DataLoader
from utils.tool import accuracy, tensor2numpy
from utils.data_manager import DummyDataset
EPSILON=1e-8
class BaseIncremnetalMethod(object):
    def __init__(self, args) -> None:
        self.args = args
        self._cur_task = -1
        self._know_class = 0
        self._total_class = 0
        self._incre_net = None
        self.workers = args["num_workers"] if torch.cuda.is_available() else None
        self._samples_memory, self._labels_memory =[], []
        self._memory_size = args["memory_size"]
        self._device = args["device"]
        self._init_lr = args["init_lr"]
        self._incre_lr = args["incre_lr"]   
        self._weight_decay = args["weight_decay"] 
        self._mile_stone = args["milestones"]
    @property
    def examplar_size(self):
        assert len(self._samples_memory) == len(self._labels_memory), "Size Error!"
        return len(self._labels_memory)
    
    @property
    def num_samples_per_class(self):
        assert self._total_class!=0, "Total classes is 0!"
        return self._memory_size//self._total_class

    @property
    def memory(self):
        if len(self._samples_memory) == 0:
            return None
        else:
            return (self._samples_memory, self._labels_memory)
    
    def save_checkpoint(self, test_acc):
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
        torch.save(save_dict, "{}_{}.pkl".format(checkpoint_name, self._cur_task))
    
    def after_task():
        pass
    
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

    def _building_examplar(self, data_manager, mem_per_class: int):
        '''
        args:
            (Object) data_manager
            int mem_per_class
        '''
        print(f"Constructing examplar set ...{mem_per_class} per class")
        for class_id in range(self._know_class, self._total_class):
            data, labels, dataset_id = data_manager.get_data_of_task(class_id, class_id+1, isTrain=True)
            data_loader = DataLoader(dataset_id, batch_size=self.args["batch_size"], shuffle=False)
            vector, _ = self._extract_vectors(data_loader)
            vectors = (vector.T / (np.linalg.norm(vector.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0) 
            examplar_vectors = []
            S = np.sum(examplar_vectors, axis = 0)
            for k in range(1, mem_per_class+1):
                mu_p = (vectors+S)/k
                idx = np.argmin(np.sum((class_mean-mu_p)**2, axis=1))
                self._samples_memory.append(torch.tensor(data[idx]))
                self._labels_memory.append(torch.tensor(labels[idx]))
                examplar_vectors.append[np.array(vectors[idx])]
                data = np.delete(data, idx, axis=0)
                labels = np.delete(labels, idx, axis=0)
                vectors = np.delete(vectors, idx, axis=0)
                if len(vectors)==0:
                    break

    def _extract_vectors(self, loader):
        self._incre_net.eval()
        vectors, targets = [], []
        for _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._incre_net, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._incre_net.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._incre_net.extract_vector(_inputs.to(self._device))
                )
            vectors.append(_vectors)
            targets.append(_targets)
        return np.concatenate(vectors), np.concatenate(targets)
    
    
                

