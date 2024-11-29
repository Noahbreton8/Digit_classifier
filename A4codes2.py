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
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
        
    n_epochs = 30
    start_time = time.time()
    for epoch in range(n_epochs):
        print(epoch)
        train(model, optimizer, train_dataloader, device)
        #test(model, testdataloader, device)
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 1 input channel (grayscale), 32 output channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 32 input channels, 64 output channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling with a 2x2 window
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 14 * 7, 512)  # Adjusted for the flattened size after pooling
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)  # Dropout for regularization
        
    def forward(self, x):
        # Reshape input to (batch_size, 1, 28, 28)
        x = x.view(-1, 1, 56, 28)
        
        # Convolutional and pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)  # ReLU activation
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Softmax for multi-class classification
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
def test(model, model2, test_dataloader, testdataloader2, device):
    model.eval()  # entering evaluation mode (dropout behaves differently)
    model2.eval()
    m = len(test_dataloader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():  # do not compute gradient
        for (data, target),(data2, _) in zip(test_dataloader,testdataloader2):
            data, target = data.to(device), target.to(device)
            data2 = data.to(device)
            output = model(data)
            output_max = output.amax(dim=1, keepdim=True)
            output2 = model2(data2)
            output2_max = output2.amax(dim=1, keepdim=True)
            
            filter = output_max > output2_max
            
            output = torch.where(filter, output, output2)

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

# train_data = np.loadtxt('A4data/A4train.csv', delimiter=',')
# train_y = train_data[:, 0]
# train_X = train_data[:, 1:] / 255.
# _, traindataloader = prepareData(train_X, train_y)

train_data = np.loadtxt('A4data/A4train.csv', delimiter=',')
train_y = train_data[:, 0]
train_X = train_data[:, 1: 28*28 *2 +1] / 255.
train2_X = np.concatenate([train_data[:, 1: 28*28 *1+1] / 255., train_data[:, 28*28 *2 + 1:] / 255.], axis=1)
_, traindataloader = prepareData(train_X, train_y)
_, train2dataloader = prepareData(train2_X, train_y)

test_data = np.loadtxt('A4data/A4val.csv', delimiter=',')
test_y = test_data[:, 0]
test_X = test_data[:, 1: 28*28 *2 +1] / 255.
test2_X = np.concatenate([test_data[:, 1: 28*28 *1+1] / 255., test_data[:, 28*28 *2 + 1:] / 255.], axis=1)
_, testdataloader = prepareData(test_X, test_y)
_, testdataloader2 = prepareData(test2_X, test_y)
    
def main():
    # train_data = np.loadtxt('A4data/A4train.csv', delimiter=',')
    # train_y = train_data[:, 0]
    # train_X = train_data[:, 1:] / 255.

    # test_data = np.loadtxt('A4data/A4val.csv', delimiter=',')
    # test_y = test_data[:, 0]
    # test_X = test_data[:, 1:] / 255.
    # _, testdataloader = prepareData(test_X, test_y)
    print(train2_X.shape)
    print(train_X.shape)
    model = learn(train_X, train_y)
    model2 = learn(train2_X, train_y)
    
    # test(model, model2, )
    
    #yhat = classify(test_X, model)
    # use proper device
    
    test(model, model2, testdataloader, testdataloader2, device)
    
main()