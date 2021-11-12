import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

class FFNet(nn.Module):
    def __init__(self):
        super(FFNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size = (3,3), stride = (1,1))

        self.conv2 = nn.Conv2d(32, 32, kernel_size = (5,5), stride = (1,1), padding = 2)
        self.pool1 = nn.MaxPool2d(kernel_size = (3,3), stride= (2,2))

        self.conv3 = nn.Conv2d(32, 64, kernel_size = (3,3), stride = (1,1), padding = 1)

        self.conv4 = nn.Conv2d(64, 64, kernel_size = (5,5), stride = (1,1), padding = 2)
        self.pool2 = nn.MaxPool2d(kernel_size = (3,3), stride= (2,2))

        self.conv5 = nn.Conv2d(64, 128, kernel_size = (3,3), stride = (1,1), padding = 1)

        self.conv6 = nn.Conv2d(128, 128, kernel_size = (5,5), stride = (1,1), padding = 2)
        self.pool3 = nn.MaxPool2d(kernel_size = (3,3), stride=(2,2))

        self.hidden1 = nn.Linear(30*30*128, 256)

        self.drop1 = nn.Dropout(0.5)

        self.hidden2 = nn.Linear(256, 256)

        self.drop2 = nn.Dropout(0.5)

        self.hidden3 = nn.Linear(256, 2) ##num classes for 2 

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.hidden2(x)
        x = self.drop2(x)
        x = self.hidden3(x)
        x = self.sigmoid(x)
        return x


        


