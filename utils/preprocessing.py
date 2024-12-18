import numpy as np
import pandas as pd
import sys
import os
import random
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())
class Preprocessing():
    def __init__(self) -> None:
        pass
    def fit_transform(self, x, y, method: str):
        self.x = x
        self.y = y
        freq = {}
        # Count Freqency of labels in dataset 
        for label in y:
            if label in freq.keys():
                freq[label]+=1
                continue
            freq[label] = 1
        sorted_freq = [freq[key] for key in sorted(freq.keys())]
        if method.lower() == "mean_sampling":
            self.balance_data(sorted_freq)
        elif method.lower() == "under_sampling":
            self.under_sampling(sorted_freq)
        return self.x, self.y

    def balance_data(self, sorted_freq):
        num_balance = int(sum(sorted_freq)/len(sorted_freq))
        print("Balance Threshold:", num_balance)
        for class_id in range(len(sorted_freq)):
            idx_list = [idx for idx, label in enumerate(self.y) if label==class_id]
            len_idx_list = len(idx_list)
            if len_idx_list>=num_balance:
                deleted = idx_list[num_balance:len_idx_list]
                self.x = np.delete(self.x, deleted, axis=0)
                self.y = np.delete(self.y, deleted)
                continue
            class_data = [self.x[idx] for idx in idx_list] #get every samples have class is idx
            num_generate = int(num_balance-len_idx_list)
            generated_data = []
            for _ in range(num_generate):
                random_2_points = random.sample(range(len(class_data)),2)
                generated_data.append((class_data[random_2_points[0]]+class_data[random_2_points[1]])/2)
            self.x = np.concatenate((self.x,np.array(generated_data)),axis=0)
            self.y = np.concatenate((self.y,np.array([class_id for _ in range(num_generate)])))

    def under_sampling(self, sorted_freq):
        num_balance = min(sorted_freq)
        print("Balance threshold: ", num_balance)
        for class_id in range(len(sorted_freq)):
            idx_list = [idx for idx, label in enumerate(self.y) if label==class_id]
            if sorted_freq[class_id]==num_balance:
                continue
            deleted = idx_list[num_balance:]
            self.x = np.delete(self.x, deleted, axis=0)
            self.y = np.delete(self.y, deleted)



    
            

            
            
        

