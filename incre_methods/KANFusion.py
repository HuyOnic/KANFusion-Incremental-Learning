import os 
import sys
sys.path.append(os.getcwd())
from torch import optim
import numpy as np
from tqdm import tqdm
from incre_methods.baselearner import BaseIncremnetalMethod
from incre_net.incre_net import IncrementalNet
from torch.utils.data import DataLoader
from utils.data_manager import DummyDataset
from torch import nn
import copy
import torch
from model.CKAN import CKAN
import logging
class KANFusion(BaseIncremnetalMethod):
    def __init__(self, args) -> None:
        super().__init__(args)
        self._incre_net =  IncrementalNet(model_name=args["model"], pretrain=False)
        self._task_weight = []
        self._samples_memory = []
        self._labels_memory = []
        
    def after_task(self):
        self._know_class = self._total_class
        print(f'Examplar size: {self.examplar_size}')

    def incremental_train(self, data_manager):
        self._cur_task+=1
        num_classes = data_manager.get_task_size(self._cur_task)
        self._total_class = self._know_class + num_classes
        self._selector_net = CKAN(num_class=self._total_class)
        self._incre_net.update_fc(self._total_class)
        self._incre_net.to(self._device)
        print(f'Task {self._cur_task}: Learning on class {self._know_class} - {self._total_class-1}')
        #Prepare Dataset
        _, _, train_dataset = data_manager.get_data_of_task( 
            num_known_classes = self._know_class, 
            total_classes = self._total_class,
            isTrain = True,
            appendent = None #change this
        ) 
        train_loader = DataLoader(
                            train_dataset, 
                            batch_size=self.args["batch_size"], 
                            shuffle=True
                            )
        
        #Train Model
        if self._cur_task>0:
            optimizer = optim.SGD(self._incre_net.parameters(), momentum=0.9, lr=self.args["incre_lr"], weight_decay=self._weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, 
                milestones=self._mile_stone,
                gamma=self.args["lr_decay"]
            )
            self._update_presentation(train_loader, optimizer, scheduler)
            self._building_examplar(data_manager, 1000)
            #Train Selector Network
            self._train_selector_net(self._samples_memory, self._labels_memory) #Train selector after task 2

        else:
            optimizer = optim.SGD(
                                self._incre_net.parameters(), 
                                momentum=0.9, 
                                lr=self.args["init_lr"], 
                                weight_decay=self.args["weight_decay"])
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, 
                milestones=self._mile_stone,
                gamma=self.args["lr_decay"]
            )
            #Load checkpoints from a pretrain model
            self._init_train(train_loader, optimizer, scheduler)
            self._building_examplar(data_manager, 1000)
        self._task_weight.append(copy.deepcopy(self._incre_net))

    def _init_train(self, train_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epochs"]))
        self._incre_net.model.to(self._device)
        criterion = nn.CrossEntropyLoss()
        for epoch in prog_bar:
            total_loss = 0
            # correct = 0
            # num_samples = 0
            for batch_idx, (samples, labels) in enumerate(train_loader):
                samples.to(self._device)
                labels.to(self._device)
                logits = self._incre_net(samples)
                labels = labels.to(torch.long)
                running_loss = criterion(logits, labels) 
                optimizer.zero_grad()
                running_loss.backward()
                optimizer.step()
                total_loss+= running_loss.item()
                # predict = torch.argmax(logits, dim=1)
                # correct += (predict==labels).sum().item()
                # total+=labels.size(0)
            scheduler.step()
        # training_acc = correct/num_samples*100
            training_loss = total_loss/(batch_idx+1)
            if epoch%5==0:
                prog_bar.write(f'Epoch {epoch} Loss {training_loss}')
        
    def _update_presentation(self, train_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["incre_epochs"]))
        criterion = nn.CrossEntropyLoss()
        for epoch in prog_bar:
            total_loss = 0
            # correct = 0
            # num_samples = 0
            for batch_idx, (samples, labels) in enumerate(train_loader):
                samples.to(self._device)
                labels.to(self._device)
                logits = self._incre_net(samples)
                labels = labels.to(torch.long)
                running_loss = criterion(logits, labels) 
                optimizer.zero_grad()
                running_loss.backward()
                optimizer.step()
                total_loss+= running_loss.item()
                # predict = torch.argmax(logits, dim=1)
                # correct += (predict==labels).sum().item()
                # total+=labels.size(0)
            scheduler.step()
                # training_acc = correct/num_samples*100
            if epoch%5==0:
                training_loss = total_loss/(batch_idx+1)
                print(f'Epoch {epoch} Loss {training_loss}')
            
    def _train_selector_net(self, train_data, train_labels):
        dataset = DummyDataset(train_data, train_labels)
        dataloader = DataLoader(dataset, batch_size=self.args["batch_size"], shuffle=True)
        self._selector_net.to(self._device)
        optimizer = optim.SGD(self._selector_net.parameters(), lr=self.args["init_lr"], momentum=0.9, weight_decay=self.args["weight_decay"])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self._mile_stone, gamma=self.args["lr_decay"])
        criterion = nn.CrossEntropyLoss()
        prog_bar = tqdm(range(self.args["selector_epochs"]))
        logging.info(f'Training Selector Network ...')
        for epoch in prog_bar:
            total_loss = 0
            for batch_idx, (samples, labels) in enumerate(dataloader):
                samples.to(self._device)
                labels.to(self._device)
                logits = self._selector_net(samples)
                labels = labels.to(torch.long)
                running_loss = criterion(logits, labels) 
                optimizer.zero_grad()
                running_loss.backward()
                optimizer.step()
                total_loss += running_loss.item()
            scheduler.step()
            if epoch%5==0:
                training_loss = total_loss/(batch_idx+1)
                logging.info(f'Epoch {epoch} Loss {training_loss}')
    
    def _building_examplar(self, data_manager, mem_per_class: int):
        '''
        args:
            (Object) data_manager
            int mem_per_class
        '''
        print(f"Constructing examplar set ...({mem_per_class}) per class")
        for class_id in range(self._know_class, self._total_class):
            data, labels, dataset_id = data_manager.get_data_of_task(class_id, class_id+1, isTrain=True)
            data_loader = DataLoader(dataset_id, batch_size=self.args["batch_size"], shuffle=False)
            vector, _ = self._extract_vectors(data_loader)
            vectors = (vector.T / (np.linalg.norm(vector.T, axis=0) + 1e-8)).T
            class_mean = np.mean(vectors, axis=0) 
            examplar_vectors = []
            S = np.sum(examplar_vectors, axis = 0) 
            for k in range(1, mem_per_class+1):
                mu_p = (vectors+S)/k
                idx = np.argmin(np.sum((class_mean-mu_p)**2, axis=1))
                self._samples_memory.append(torch.tensor(data[idx]))
                self._labels_memory.append(torch.tensor(self._cur_task))
                examplar_vectors.append(np.array(vectors[idx]))
                data = np.delete(data, idx, axis=0)
                labels = np.delete(labels, idx, axis=0)
                vectors = np.delete(vectors, idx, axis=0)
                if len(vectors)==0:
                    break

    def eval_task(self, data_manager, save_conf=False):
        cls_order = data_manager._class_order[:self._total_class]
        correct = 0
        all_acc = {}
        all_models = [model.eval() for model in self._task_weight]
        #Evaluate on all seen classes
        logging.info(f'Evaluate on all seen classes ...')
        with torch.no_grad():
            if self._cur_task>0: 
                for cls in cls_order: # Load each class sequentially
                    _ , _, test_dataset = data_manager.get_data_of_task( 
                                        num_known_classes = cls, 
                                        total_classes = cls+1,
                                        isTrain = False
                                        ) 
                    test_loader = DataLoader(
                                test_dataset, 
                                batch_size=1, 
                                shuffle=False, 
                                )
                    #Select model and Classify
                    for batch_idx, (samples, labels) in enumerate(test_loader):
                        samples.to(self._device), labels.to(self._device)
                        for sample, label in zip(samples, labels):
                            selected_classifier = np.argmax(self._selector_net.predict(sample.view(1,-1)))
                            pred = all_models[selected_classifier](samples)
                            pred_value = torch.argmax(pred, dim=1)
                            correct += 1 if pred_value==label else 0
                    all_acc[cls]=correct/len(test_dataset)
                    logging.info(f'Accuracy of class {cls}: {all_acc[cls]}')
            else:
                for cls in cls_order:
                    _ , _, test_dataset = data_manager.get_data_of_task( 
                                        num_known_classes = cls, 
                                        total_classes = cls+1,
                                        isTrain = False,
                                        ) 
                    test_loader = DataLoader(
                                test_dataset, 
                                batch_size=1, 
                                shuffle=False, 
                                )
                    #Select model and Classify
                    for batch_idx, (samples, labels) in enumerate(test_loader):
                        samples.to(self._device), labels.to(self._device)
                        pred = all_models[0](samples)
                        pred_value = torch.argmax(pred, dim=1)
                        correct += (pred_value==labels).sum().item()
                    all_acc[cls]=correct/len(test_dataset)
                    logging.info(f'Accuracy of class {cls}: {all_acc[cls]}')

            all_acc['avg'] = sum([all_acc[cls] for cls in all_acc.keys()])/self._total_class
            logging.info(f'Average Accuracy: {all_acc['avg']}')

    
    def eval_selector(self, predict, labels): #Evaluate selector network
        return np.sum(predict==labels).item()/len(labels) 
    
    def save_task_weights(self, time_str):
        task_name = f"exps/kanfusion/{self.args["incre_method"]}_task_{self._cur_task}"
        print(f'Saving task weights {task_name}')
        results = {f'Task{k}_Net':v.state_dict() for k,v in enumerate(self._task_weight)}
        results['Selector_Net'] = self._selector_net.state_dict()
        torch.save(results, f"exps/kanfusion_{time_str}/model.pt")







        

        




                    



