import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

class Parity(nn.Module):
    def __init__(self):
        super(Parity, self).__init__()

        #convolutions and pooling
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        #connected layers and batch normalizations
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)  
        
        #convolutions and pooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        #connected layers and batch normalizations
        x = x.view(-1, 64 * 7 * 7) 
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x)) # turns output into a number from 0-1 inclusive
        return x

class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()

        #convolutions and pooling
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        #connected layers and batch normalizations
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)

        #convolutions and pooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        #connected layers and batch normalizations
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1) #output is a vector of 10 numbers each with a probability
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.parity_classifier = Parity()
        self.digit_classifier = DigitClassifier()

    def forward(self, x):
        #getting the 3 different images
        top_img = x[:, :28 * 28]
        middle_img = x[:, 28 * 28:2 * 28 * 28]
        bottom_img = x[:, 2 * 28 * 28:]

        #reshaping images for models
        top_img = top_img.view(-1, 1, 28, 28)
        middle_img = middle_img.view(-1, 1, 28, 28)
        bottom_img = bottom_img.view(-1, 1, 28, 28)

        #gets the probability the top image is odd
        parity = self.parity_classifier(top_img)

        #gets prediction of middle and bottom image
        middle_prediction = self.digit_classifier(middle_img)
        bottom_prediction = self.digit_classifier(bottom_img)

        #weighted prediction from parity and vectors from middle and bottom
        combined_prediction = parity * middle_prediction + (1 - parity) * bottom_prediction

        return combined_prediction

def prepareData(X, y):
    tensor_X = torch.Tensor(X)
    tensor_y = torch.LongTensor(y)
    dataset = TensorDataset(tensor_X, tensor_y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataset, dataloader

#taken from lecture 13 demo (train + prepare)
def learn(X, y):
    y = y.astype(int)   #make sure the labels are integers
    X = X / 255.0       #normalize the data

    _, train_dataloader = prepareData(X, y)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    n_epochs = 20
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}")
        for data, target in train_dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

    return model

def classify(Xtest, model):
    model.eval()

    Xtest = Xtest / 255.

    tensor_X = torch.Tensor(Xtest)
    dataset = TensorDataset(tensor_X)
    dataloader = DataLoader(dataset)

    model.eval()

    m = Xtest.shape[0]
    y_hat = np.zeros((m, 1))

    for i, data in enumerate(dataloader):
        data = data[0].to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        y_hat[i] = pred.cpu().numpy()

    return y_hat

#main function that loads train and test data and runs and verifies model
def main():
    path_to_train = "A4data/A4train.csv"
    path_to_test = "A4data/A4val.csv"

    train_data = np.loadtxt(path_to_train, delimiter=",")
    train_y = train_data[:, 0]
    train_X = train_data[:, 1:]

    model = learn(train_X, train_y)

    test_data = np.loadtxt(path_to_test, delimiter=",")
    test_y = test_data[:, 0]
    test_X = test_data[:, 1:]

    yhat = classify(test_X, model)
    test_y = test_y.reshape(-1, 1)

    correct = np.sum(yhat == test_y)
    total = test_y.shape[0]
    print(correct/total)