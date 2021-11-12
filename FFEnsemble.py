import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

class FFEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(FFEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(4, 2)
        
    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        return x