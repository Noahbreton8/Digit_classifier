import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to visualize an image
def plotImg(x):
    img = x.reshape((84, 28))
    plt.imshow(img, cmap="gray")
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Convolutional layers for path A (top + middle images)
        self.conv1_a = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_a = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool_a = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional layers for path B (top + bottom images)
        self.conv1_b = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_b = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool_b = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers for path A (top + middle)
        self.fc1_a = nn.Linear(64 * 7 * 14, 1024)  # Adjusted after pooling
        self.bn1_a = nn.BatchNorm1d(1024)
        self.fc2_a = nn.Linear(1024, 256)
        self.bn2_a = nn.BatchNorm1d(256)
        self.fc3_a = nn.Linear(256, 10)

        # Fully connected layers for path B (top + bottom)
        self.fc1_b = nn.Linear(64 * 7 * 14, 1024)  # Adjusted after pooling
        self.bn1_b = nn.BatchNorm1d(1024)
        self.fc2_b = nn.Linear(1024, 256)
        self.bn2_b = nn.BatchNorm1d(256)
        self.fc3_b = nn.Linear(256, 10)

        # Dropout to regularize
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        batch_size = x.size(0)

        # Path A (top + middle images) - First process top and middle
        x1 = x[:, :2 * 28 * 28].view(batch_size, 1, 28, 56)  # Reshaping for 2D conv

        # Convolutional layers
        x1 = self.pool_a(F.relu(self.conv1_a(x1)))
        x1 = self.pool_a(F.relu(self.conv2_a(x1)))
        
        # Flatten before passing to fully connected layers
        x1 = x1.view(batch_size, -1)
        
        # Fully connected layers for path A
        x1 = self.bn1_a(self.fc1_a(x1))
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.bn2_a(self.fc2_a(x1))
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        out1 = F.log_softmax(self.fc3_a(x1), dim=1)

        # Path B (top + bottom images) - First process top and bottom
        x2 = torch.cat((x[:, :28 * 28], x[:, 28 * 28 * 2:]), dim=1).view(batch_size, 1, 28, 56)  # Reshaping for 2D conv
        
        # Convolutional layers
        x2 = self.pool_b(F.relu(self.conv1_b(x2)))
        x2 = self.pool_b(F.relu(self.conv2_b(x2)))
        
        # Flatten before passing to fully connected layers
        x2 = x2.view(batch_size, -1)
        
        # Fully connected layers for path B
        x2 = self.bn1_b(self.fc1_b(x2))
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.bn2_b(self.fc2_b(x2))
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        out2 = F.log_softmax(self.fc3_b(x2), dim=1)

        # For each element in the batch, get which output is larger
        output_max1 = out1.amax(dim=1, keepdim=True)
        output_max2 = out2.amax(dim=1, keepdim=True)
        filter = output_max1 > output_max2
        output = torch.where(filter, out1, out2)

        return output

# Prepare data
def prepareData(X, y):
    tensor_X = torch.Tensor(X)
    tensor_y = torch.LongTensor(y)
    dataset = TensorDataset(tensor_X, tensor_y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataset, dataloader

# Training loop
def train(model, optimizer, train_dataloader, device):
    model.train()
    for data, target in train_dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

# Testing loop
def test(model, test_dataloader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_dataloader.dataset)
    accuracy = correct / len(test_dataloader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.3f}")
    return test_loss, accuracy

# Learn function
def learn(X, y):
    _, train_dataloader = prepareData(X, y)
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    n_epochs = 30
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}")
        train(model, optimizer, train_dataloader, device)
        test(model, test_dataloader, device)
    return model

# Classify function
def classify(Xtest, model):
    model.eval()
    yhat = []
    with torch.no_grad():
        for data in Xtest:
            data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            yhat.append(pred.item())
    return np.array(yhat)

test_data = np.loadtxt("A4data/A4val.csv", delimiter=",")
test_y = test_data[:, 0]
test_X = test_data[:, 1:] / 255.0
_, test_dataloader = prepareData(test_X, test_y)

# Main function
def main():
    train_data = np.loadtxt("A4data/A4train.csv", delimiter=",")
    train_y = train_data[:, 0]
    train_X = train_data[:, 1:] / 255.0
    model = learn(train_X, train_y)
    
    # test(model, test_dataloader, device)

main()