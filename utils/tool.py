import numpy as np
#Calculate all accuracy (all seen classes, for each classes, new classes and old classes)
def accuracy(y_pred , y_true, num_known_classes, incremental):
    '''
    args:
    - y_pred: predicted values
    - y_true: true values
    - incremental_list: a list includes number of class per incremental task.
    - incremental_
    '''
    assert len(y_pred)==len(y_true), "Data lenght error!"
    all_acc = {}
    #Accuracy for all seen classes
    all_acc["total"] = (y_pred==y_true).sum().item()*100/len(y_true)
    #Accuracy for each classes
    for class_id in range(0, max(y_true), incremental):
        label = f"{class_id}-{class_id+incremental}" 
        idx = np.where(np.logical_and(class_id<=y_true, y_true<class_id+incremental))[0]
        all_acc[label] = (y_true==y_pred).sum().item()*100/len(y_true)
    #Accuracy  classes on old classes 
    # idxes = np.where(y_true<num_known_classes)[0]
    # all_acc["old"] = (y_true[idxes]==y_pred[idxes]).sum().item()*100/len()
    #Accuracy of old classes( the task before )
    # idxes = np.where(y_true>=num_known_classes)[0]
    # all_acc["new"] = (y_true[idxes]==y_pred[idxes]).sum().item()
    return all_acc

def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()