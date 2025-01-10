import os 
import sys
sys.path.append(os.getcwd())
from torch import optim
import numpy as np
from tqdm import tqdm
from incre_methods.baselearner import BaseIncremnetalMethod
from incre_net.node_model import NodeModel
from torch.utils.data import DataLoader
from utils.data_manager import DummyDataset
from torch import nn
import copy
import torch
from model.CKAN import CKAN
from model.KAN import KAN
import logging
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
class NexusiKAN(BaseIncremnetalMethod):
    def __init__(self, args) -> None:
        super().__init__(args)
        self._incre_net =  NodeModel(model_name=args["model"], pretrain=False)
        self._task_weight = []
        self._selector_net = GaussianNB()

    def incremental_train(self, data_manager):
        self._cur_task+=1
        num_classes = data_manager.get_task_size(self._cur_task)
        self._total_class = self._know_class + num_classes
        self._incre_net.update_fc(self._total_class)
        self._incre_net.to(self._device)
        print(f'Task {self._cur_task}: Learning on class {self._know_class} - {self._total_class-1}')
        #Prepare Dataset
        train_dataset = data_manager.get_data( 
            num_known_classes = self._know_class, 
            total_classes = self._total_class,
            isTrain = True
        ) 
        train_loader = DataLoader(
                            train_dataset, 
                            batch_size=self.args["batch_size"], 
                            shuffle=True
                            )
        
        #Train Model
        if self._cur_task>0:
            optimizer = optim.SGD(self._incre_net.parameters(), momentum=0.9, lr=self.args["incre_lr"], weight_decay=self.args["weight_decay"])
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, 
                milestones=self.args["milestones"],
                gamma=self.args["lr_decay"]
            )
            self._update_presentation(train_loader, optimizer, scheduler)
            self._building_examplar(data_manager, self.args["memory_size"])
            #Train Selector Network
            # self._selector_net.update_fc(self._cur_task+1)
            # self._train_selector_net(self._samples_memory, self._labels_memory) #Train selector after task 2
            x_train, x_test, y_train, y_test = train_test_split(self._samples_memory, self._labels_memory, test_size=0.2, random_state=42)
            self._selector_net.fit(x_train, y_train)
            #Evaluate Selector Network
            pred = self._selector_net.predict(x_test)
            acc = self.eval_selector(pred, y_test)
            print(f'Accuracy of Selector Network: {acc}')
        else:
            optimizer = optim.SGD(
                                self._incre_net.parameters(), 
                                momentum=0.9, 
                                lr=self.args["init_lr"], 
                                weight_decay=self.args["weight_decay"])
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, 
                milestones=self.args["milestones"], 
                gamma=self.args["lr_decay"]
            )
            #Load checkpoints from a pretrain model
            self._init_train(train_loader, optimizer, scheduler)
            self._building_examplar(data_manager, self._memory_size)
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
                samples = samples.to(self._device)
                labels = labels.to(self._device)
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
                samples = samples.to(self._device)
                labels = labels.to(self._device)
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
        self._selector_net.reset_paramerters()
        dataset = DummyDataset(train_data, train_labels)
        dataloader = DataLoader(dataset, batch_size=self.args["batch_size"], shuffle=True)
        self._selector_net.to(self._device)
        optimizer = optim.SGD(self._selector_net.parameters(), lr=self.args["init_lr"], momentum=0.9, weight_decay=self.args["weight_decay"])
        criterion = nn.CrossEntropyLoss()
        prog_bar = tqdm(range(self.args["selector_epochs"]))
        logging.info(f'Training Selector Network ...')
        for epoch in prog_bar:
            total_loss = 0
            for batch_idx, (samples, labels) in enumerate(dataloader):
                samples = samples.to(self._device)
                labels = labels.to(self._device)
                optimizer.zero_grad()
                # samples = samples.view(samples.size(0), -1)
                logits = self._selector_net(samples)
                labels = labels.to(torch.long)
                running_loss = criterion(logits, labels) 
                running_loss.backward()
                optimizer.step()
                total_loss += running_loss.item()
            training_loss = total_loss/(batch_idx+1)
            print(f'Head model - Epoch {epoch} - Loss {training_loss}')

    def _extract_vectors(self, loader):
        self._incre_net.eval()
        vectors, targets = [], []
        for _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._incre_net, nn.DataParallel):
                _vectors = self._incre_net.module.extract_vector(_inputs.to(self._device)).cpu().data.numpy()
            else:
                _vectors = self._incre_net.extract_vector(_inputs.to(self._device)).data.numpy()
            vectors.append(_vectors)
            targets.append(_targets)
        return torch.tensor(np.concatenate(vectors)), torch.tensor(np.concatenate(targets))

    def _building_examplar(self, data_manager, mem_per_class: int):
        '''
        args:
            (Object) : data_manager
            int : mem_per_class
        '''
        print(f"Constructing examplar set ...({mem_per_class})")
        for class_id in range(self._know_class, self._total_class):
            dataset = data_manager.get_data(class_id, class_id+1, isTrain=True)
            data_loader = DataLoader(dataset, batch_size=self.args["batch_size"], shuffle=False)
            vectors, _ = self._extract_vectors(data_loader)
            mean = torch.sum(vectors)/len(dataset)
            distances = torch.stack([torch.norm(vector-mean, p=2) for vector in vectors])
            _, indices = torch.topk(distances, mem_per_class, largest=False)
            for idx in indices:
                self._samples_memory.append(dataset[idx][0].view(-1))
                self._labels_memory.append(self._cur_task)

    def eval_task(self, data_manager, save_conf=False):
        cls_order = data_manager._class_order[:self._total_class]
        cls_order = cls_order[::-1]
        all_acc = {}
        all_models = [model.eval() for model in self._task_weight]
        #Evaluate on all seen classes
        logging.info(f'Evaluate on all seen classes ...')
        with torch.no_grad():
            if self._cur_task>0: 
                for cls in cls_order: # Load each class sequentially
                    correct = 0
                    test_dataset = data_manager.get_data( 
                                        num_known_classes = cls, 
                                        total_classes = cls+1,
                                        isTrain = False
                                        ) 
                    test_loader = DataLoader(
                                test_dataset, 
                                batch_size=self.args["batch_size"], 
                                shuffle=False, 
                                )
                    #Select model and Classify
                    for batch_idx, (samples, labels) in enumerate(test_loader):
                        samples.to(self._device), labels.to(self._device)
                        for sample, label in zip(samples, labels):
                            selected_classifier = np.argmax(self._selector_net.predict(sample.view(-1).unsqueeze(0)))
                            print(f"Class {cls} Selected Classifier: {selected_classifier}")
                            pred = all_models[selected_classifier](sample.unsqueeze(0))
                            pred_value = torch.argmax(pred, dim=1)
                            correct += 1 if pred_value==label else 0
                    all_acc[cls]=correct/len(test_dataset)
                    logging.info(f'Accuracy of class {cls}: {all_acc[cls]}')
            else:
                for cls in cls_order:
                    correct = 0
                    test_dataset = data_manager.get_data( 
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

            all_acc["avg"] = sum([all_acc[cls] for cls in all_acc.keys()])/self._total_class
            logging.info(f'Average Accuracy: {all_acc["avg"]}')
    
    def eval_selector(self, predict, labels): #Evaluate selector network
        return np.sum(predict==labels).item()/len(labels) 
    
    def save_task_weights(self, time_str):
        task_name = f'exps/kanfusion/{self.args["incre_method"]}_task_{self._cur_task}'
        print(f'Saving task weights {task_name}')
        results = {f'Task{k}_Net':v.state_dict() for k,v in enumerate(self._task_weight)}
        results['Selector_Net'] = self._selector_net
        torch.save(results, f'exps/kanfusion_{time_str}/model.pt')







        

        




                    



