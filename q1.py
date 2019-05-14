#!/usr/bin/env python3

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib as mp
mp.use('Agg') 

model = None
criterion = nn.NLLLoss()
optimizer = None
device = torch.device('cpu')
batch_size = 100
batch_set = 40000 / batch_size
log_limit = batch_size / 10
epochs = 50

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 100)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        sig = nn.Sigmoid()
        x = x.view(-1, 3*32*32)
        x = sig(self.fc1(x))
        x = self.fc1_drop(x)
        return fn.log_softmax(self.fc2(x), dim=1)


def main(lr):
    global model, optimizer

    # Load in batch data
    train_data = datasets.CIFAR10('.', 
                                train=True, 
                                download=True,  
                                transform=transforms.ToTensor())

    testing_data = datasets.CIFAR10('.',
                                train=False,
                                download=True,
                                transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                shuffle=True)
                                
    testing_loader = torch.utils.data.DataLoader(dataset=testing_data,
                                batch_size=batch_size,
                                shuffle=False)

    # Seperate training and validation batches
    train_batches = []
    validate_batches = []
    for i, val in enumerate(train_loader):
        if i < batch_set:
            train_batches.append(val)
        else:
            validate_batches.append(val)
    
    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)

    # Run training
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(1, epochs + 1):
        # Training data
        train(epoch, train_batches, log_interval=log_limit)
    
        # Training and validation set validation
        validate(train_acc, train_loss, train_batches)
        validate(valid_acc, valid_loss, validate_batches)

    # Testing set
    test_validate(test_acc, test_loss, testing_loader)
    print(f'Testing accuracy, loss: {test_acc[0]}, {test_loss[0]}')

    # Plot
    epoch_label = [i for i in range(1, epochs + 1)]
    plt.title('Training and Validation Loss vs Epoch')
    plt.xlabel('Current Epoch')
    plt.ylabel('Loss')
    plt.plot(epoch_label, train_loss, marker='o', color='red')
    plt.plot(epoch_label, valid_loss, marker='o', color='blue')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.savefig(f'q1_loss_{lr}.png')
    plt.clf()

    plt.title('Training and Validation Accuracy vs Epoch')
    plt.xlabel('Current Epoch')
    plt.ylabel('Acc')
    plt.plot(epoch_label, train_acc, marker='o', color='red')
    plt.plot(epoch_label, valid_acc, marker='o', color='blue')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.savefig(f'q1_acc_{lr}.png')
    plt.clf()

def train(epoch, train_batch, log_interval=50):
    # Set model to training mode
    model.train()
    
    # Loop over each batch from the training set
    loss = None
    for batch_idx, (data, target) in enumerate(train_batch):

        # Zero gradient buffers
        optimizer.zero_grad() 
        
        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_batch) * batch_size,
                100. * batch_idx / len(train_batch), loss.data.item()))


def validate(accuracy_vector, loss_vector, validation_batches):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_batches:
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_batches)
    loss_vector.append(val_loss)

    accuracy = correct.to(torch.float32) / len(validation_batches * batch_size)
    accuracy_vector.append(accuracy)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_batches * batch_size), 100. * accuracy))


def test_validate(accuracy_vector, loss_vector, validation_loader):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), 100. * accuracy))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Invalid arguments")
        exit(1)
    else:
        try:
            lr = float(sys.argv[1])
        except:
            print("lr must be a number")

        main(lr)
