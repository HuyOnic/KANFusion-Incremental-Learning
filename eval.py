import torch
from utils.dataset import Capture_128
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def get_class_names(l):
    tmp = []
    for i in l:
        if i not in tmp:
            tmp.append(i)
    return tmp
        
device = "cuda" if torch.cuda.is_available() else "cpu"
test_dataset = Capture_128(root='dataset/Capture_test_128.feather', isTrain=False)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
model = torch.load('exps/kan2.pt', weights_only=False)
mlp = torch.load('exps/mlp1.pt', weights_only=False)
model.eval()
mlp.eval()
total_loss = 0
correct_predictions = 0
mlp_correct_predictions = 0 #mlp

num_sample = 0
predict_list = []
criterion = torch.nn.CrossEntropyLoss()
with torch.no_grad():
    for batch_idx, (samples, labels) in enumerate(test_loader):
        samples.to(device)
        labels.to(device)
        predict = model(samples)
        mlp_predict = mlp(samples) #mlp
        num_sample+=samples.size(0)
        loss = criterion(predict, labels)
        total_loss += loss.item()
        predicted_labels = torch.argmax(predict,dim=1)
        mlp_predicted_labels = torch.argmax(mlp_predict,dim=1) #mlp
        predict_list+=predicted_labels.tolist()
        correct_predictions += (predicted_labels==labels).sum().item()
        mlp_correct_predictions += (mlp_predicted_labels==labels).sum().item() #mlp

        # correct_predictions += (predicted == labels).sum().item()
avg_loss = total_loss/len(test_loader)
accuracy = correct_predictions/ num_sample
print(f"Test Loss {avg_loss}")
print(f"Test Accuracy {accuracy}")
mlp_accuracy = mlp_correct_predictions/ num_sample
print(f"MLP Test Accuracy {mlp_accuracy}")

y_true = test_dataset.labels.tolist()
y_pred =  predict_list
conf_matrix = confusion_matrix(y_true, y_pred)
class_names = [i for i in range(13)]
plt.figure(figsize=(10,10))
# sns.heatmap(conf_matrix, fmt='d', annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
# plt.title('Confusion Matrix')
# plt.ylabel('Predicted Label')
# plt.xlabel('True Label')
model_names = ["MLP", "KAN"]
colors = ["red", "blue"]
plt.bar(model_names, [mlp_accuracy, accuracy], color=colors)
plt.title("MLP vs KAN (Capture 128 Dataset)")
plt.show()


