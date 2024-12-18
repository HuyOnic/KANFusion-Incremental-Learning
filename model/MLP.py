from torch.nn import Module
import torch 
from torch import nn
class MLP(Module):
    def __init__(self, width) -> None:
        super(MLP, self).__init__()
        self.SiLU = nn.SiLU()
        self.logits = nn.Softmax()
        self.width = width
        self.layers = self._create_network()

    def _create_network(self):
        layers = []
        for infeature, outfeature in zip(self.width[0:-1], self.width[1:]):
            layers += [nn.Linear(in_features=infeature, out_features=outfeature)]
            if outfeature!=self.width[-1]:
                layers += [self.SiLU]
        layers += [self.logits]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
if __name__=="__main__":
    x = torch.rand((4, 128))
    model = MLP(width=[128,32,13])
    output = model(x)
    print(torch.argmax(output,dim=1))


