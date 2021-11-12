
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from FFNet import FFNet
import torch
# from torchsummary import summary
from dataset_manager import CooccurenceDataset
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
# import matplotlib.pyplot as plt
from models import model_selection
# from FFEnsemble import FFEnsemble

import math
import numpy as np
import os
from glob import glob
# import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device we are using is:", device)

def train(model, device, train_dl, optimiser, epoch):
    
    model.train()
    train_total = 0
    train_correct = 0
    print("Starting Training Epoch:", epoch)

    # Loop training iterations
    for i, (inputs, targets) in enumerate(train_dl):
        # Load the input features and labels from the training dataset
        targets = targets.to(device)
        cooccurence_inputs = inputs[0].to(device)
        face_border_inputs = inputs[1].to(device)

        # Reset the gradients to 0 for all learnable weight parameters
        optimiser.zero_grad()

        # output = model(cooccurence_inputs.float(), face_border_inputs)
        # output = model(cooccurence_inputs.float())
        output = model(face_border_inputs)

        criterion = CrossEntropyLoss()

        loss = criterion(output, targets)

        scores, predictions = torch.max(output.data, 1)
        train_total += targets.size(0)
        train_correct += int(sum(predictions == targets))

        # Backward pass: compute the gradients of the loss w.r.t. the model's parameters
        loss.backward()

        # Update the neural network weights
        optimiser.step()

    acc = round((train_correct / train_total) * 100, 2)
    print('Training Epoch [{}], Loss: {}, Accuracy: {}'.format(epoch, loss.item(), acc))

def validate(model, device, val_dl, epoch):
    model.eval()
    val_total = 0
    val_correct = 0
    print("Starting Validation Epoch:", epoch)

    # Loop training iterations
    for i, (inputs, targets) in enumerate(val_dl):
        
        # Load the input features and labels from the training dataset
        targets = targets.to(device)
        cooccurence_inputs = inputs[0].to(device)
        face_border_inputs = inputs[1].to(device)

        # output = model(cooccurence_inputs.float())
        output = model(face_border_inputs)

        criterion = CrossEntropyLoss()

        loss = criterion(output, targets)

        scores, predictions = torch.max(output.data, 1)
        val_total += targets.size(0)
        val_correct += int(sum(predictions == targets))

    acc = round((val_correct / val_total) * 100, 2)
    print('Validation Epoch [{}], Loss: {}, Accuracy: {}'.format(epoch, loss.item(), acc))

if __name__ == '__main__':

    print('-'*100)
    print('Processing')
    print('-'*100)

    n_epochs = 40                                                     # Number of epochs

    batch_size = 5                                                    # Training batch size
    val_batch_size = 5
    test_batch_size = 5

    train_folder = '../final_data_smaller/train/'
    val_folder = '../final_data_smaller/validation/'
    test_folder = '../final_data_smaller/test/'

    # ------------------------------------------------------------------------------------------------------------------
    # Prepare a list of training and validation images
    # ------------------------------------------------------------------------------------------------------------------

    train_dataset = CooccurenceDataset(train_folder)
    val_dataset = CooccurenceDataset(val_folder)
    test_dataset = CooccurenceDataset(test_folder)

    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))

    # tensor_fb = train_dataset[0][0][1]
    # plt.imshow(tensor_image.permute(1, 2, 0))
    # plt.show()

    # print(train_dataset[0])
    # print(val_dataset[0])
    # print(test_dataset[0])

    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers = 0)
    val_dl = DataLoader(val_dataset, val_batch_size, shuffle=True, num_workers = 0)
    test_dl = DataLoader(test_dataset, test_batch_size, shuffle=True, num_workers = 0)
    
    # ------------------------------------------------------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------------------------------------------------------
    
    # model_cooccurence = FFNet().to(device)
    # model_cooccurence = model_cooccurence.float()
    # summary(model_cooccurence, (6, 256, 256))
    
    # decay = learning_rate/n_epochs

    model_xception, *_ = model_selection(modelname='xception', num_out_classes=2)
    model_xception = model_xception.to(device)
    # summary(model_xception, (3, 299, 299))

    # model_ensemble = FFEnsemble(model_cooccurence, model_xception).to(device)

    learning_rate = 0.01
    # optimiser = SGD(model_cooccurence.parameters(), lr=learning_rate, momentum=0.9)
    optimiser = SGD(model_xception.parameters(), lr=learning_rate, momentum=0.9)
    # optimiser = SGD(model_ensemble.parameters(), lr=learning_rate, momentum=0.9)

    # # ------------------------------------------------------------------------------------------------------------------
    # # Training
    # # ------------------------------------------------------------------------------------------------------------------

    # Loop epochs
    for epoch in range(n_epochs):
        train(model_xception, device, train_dl, optimiser, epoch)
        validate(model_xception, device, val_dl, epoch)

    torch.save(model_xception.state_dict(), "model_xception_paramters.pt")

    # # ------------------------------------------------------------------------------------------------------------------
    # # Test
    # # ------------------------------------------------------------------------------------------------------------------

    # model_cooccurence.eval()
    # test_loss = 0
    # test_total = 0
    # test_correct = 0

    # with torch.no_grad():
    #     # Loop training iterations
    #     for i, (inputs, targets) in enumerate(test_dl):
            
    #         # Load the input features and labels from the training dataset
    #         targets = targets.to(device)
    #         cooccurence_inputs = inputs[0].to(device)
    #         face_border_inputs = inputs[1].to(device)

    #         output = model_cooccurence(cooccurence_inputs.float())

    #         criterion = CrossEntropyLoss()

    #         loss = criterion(output, targets)

    #         scores, predictions = torch.max(output.data, 1)
    #         test_total += targets.size(0)
    #         test_correct += int(sum(predictions == targets))

    #     acc = round((test_correct / test_total) * 100, 2)
    #     print('Test Loss: {}, Test Accuracy: {}'.format(loss.item(), acc))