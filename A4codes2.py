import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

def learn(X, y):
    _, train_dataloader = prepareData(X, y)
    
    print(device)
    model = Net().to(device)
    
    # Optimizer: Stochastic gradient descent
    optimizer = optim.SGD(model.parameters(), lr=0.01)
        
    n_epochs = 30
    start_time = time.time()
    for epoch in range(n_epochs):
        print(epoch)
        train(model, optimizer, train_dataloader, device)
        test(model, testdataloader, device)
    print(f"Time: {time.time() - start_time:.3f}")

    return model 

# Prepare dataset
def prepareData(X, y):
    # Convert to PyTorch Tensors
    tensor_X = torch.Tensor(X)
    tensor_y = torch.LongTensor(y)

    dataset = TensorDataset(tensor_X, tensor_y) # create PyTorch TensorDataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True) # create PyTorch DataLoader

    return dataset, dataloader

# Define neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2352, 784)  # fully connected layers
        self.fc2 = nn.Linear(784, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)  # PyTorch: drop probability

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # ReLU activation
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        # softmax for multi-class classification
        # using log of softmax here due to how the loss is defined later
        output = F.log_softmax(x, dim=1)
        return output

# Training loop
def train(model, optimizer, train_dataloader, device):
    model.train()  # entering training mode (dropout behaves differently)
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)  # move data to the sPame device
        optimizer.zero_grad()  # clear existing gradients
        output = model(data)  # forward pass
        loss = F.nll_loss(output, target)  # compute the loss
        loss.backward()  # backward pass: calculate the gradients
        optimizer.step()  # take a gradient step


# Test loop
def test(model, test_dataloader, device):
    model.eval()  # entering evaluation mode (dropout behaves differently)
    m = len(test_dataloader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():  # do not compute gradient
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= m
    accuracy = correct / m
    print(f'Test average loss: {test_loss:.4f}, accuracy: {accuracy:.3f}')
    return test_loss, accuracy



# Training

def classify(Xtest, model):
    model.eval()  # entering evaluation mode (dropout behaves differently)
    yhat = []
    with torch.no_grad():  # do not compute gradient
        for data in Xtest:
            data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            yhat.append(pred)
    return yhat


train_data = np.loadtxt('A4data/A4train.csv', delimiter=',')
train_y = train_data[:, 0]
train_X = train_data[:, 1:] / 255.
_, traindataloader = prepareData(train_X, train_y)

test_data = np.loadtxt('A4data/A4val.csv', delimiter=',')
test_y = test_data[:, 0]
test_X = test_data[:, 1:] / 255.
_, testdataloader = prepareData(test_X, test_y)
    
def main():
    # train_data = np.loadtxt('A4data/A4train.csv', delimiter=',')
    # train_y = train_data[:, 0]
    # train_X = train_data[:, 1:] / 255.

    # test_data = np.loadtxt('A4data/A4val.csv', delimiter=',')
    # test_y = test_data[:, 0]
    # test_X = test_data[:, 1:] / 255.
    # _, testdataloader = prepareData(test_X, test_y)

    model = learn(train_X, train_y)
    
    yhat = classify(test_X, model)
    # use proper device
    
    test(model, testdataloader, device)
    
main()