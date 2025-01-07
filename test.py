import torch 
a = torch.tensor([-9,-4,2],dtype=torch.float32)
b = torch.nn.Softmax(dim=0)(a)
print(torch.argmax(b))
print(torch.sum(b))