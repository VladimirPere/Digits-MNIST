import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

data = pd.read_csv('/home/vladimir/Downloads/Data/pytorchtrain.csv')

target_name = 'label'
target = data[target_name]
target = torch.tensor(target.values, dtype=torch.int64)
training_data = data.drop(columns=[target_name])
training_data = torch.tensor(training_data.values, dtype=torch.float32)

train_data, test_data, train_target, test_target = train_test_split(training_data, target, random_state=4, test_size=.3)

# variables for training data
learning_rate = 5e-4
dr = .2
epochs = 30
batch_size = 128

train_dataset = TensorDataset(train_data, train_target)
test_dataset = TensorDataset(test_data, test_target)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

class NeuralNetwork(nn.Module):
    """Neural Network of 784->512->512->10"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

def train_loop(dataloader, model, loss_fn, optimizer):
    """Loop that trains model using Neural Network"""
    # set the model to training mode
    model.train()
    for X, y in dataloader:
        # compute prediction and loss
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()

def test_loop(dataloader, model, loss_fn):
    """Loop that evalutes test data and outputs accuracy, loss, and list of accuracy scores"""
    # set the model to evaluation mode
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # evaluating the model with torch.no_grad()
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # finding accuracy
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    accuracies.append(correct)


# type of loss function used
loss_fn = nn.CrossEntropyLoss()

# optimizer: does gradient descent at learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# scheduler: reduces learning rate by {factor} if accuracy does not increase for {patience} epochs
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 
                                           mode='max', 
                                           patience=2, 
                                           factor=.5)

accuracies = []

#Epoch Loop
for t in range(epochs):
    print(f"Epoch {t+1}")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
    scheduler.step(accuracies[-1])

# plot
plt.figure(figsize=(10, 6))
x = np.arange(1, epochs + 1, step=1)
plt.plot(x, accuracies, c='red', marker='o', label='Test Accuracy')

# line of best fit
m, b = np.polyfit(x, accuracies, deg=1)
x_line = np.linspace(1, epochs, 100)
plt.plot(x_line, m * x_line + b, color='blue', linestyle='--', linewidth=2, label='Trend')

# labels
plt.xlabel('Epoch')
plt.xticks(np.arange(0, epochs + 1, step=1))
plt.ylabel('Accuracy')
plt.title('MNIST Digits Classification - Test Accuracy per Epoch')
plt.legend()
plt.tight_layout()
plt.show()

# add graph of each epoch

# OUTPUTS
# Epoch 1
# -------------------------------
# Test Error: 
#  Accuracy: 94.8%, Avg loss: 0.181922 

# Epoch 2
# -------------------------------
# Test Error: 
#  Accuracy: 95.5%, Avg loss: 0.155023 

# Epoch 3
# -------------------------------
# Test Error: 
#  Accuracy: 96.4%, Avg loss: 0.127107 

# Epoch 4
# -------------------------------
# Test Error: 
#  Accuracy: 96.5%, Avg loss: 0.125730 

# Epoch 5
# -------------------------------
# Test Error: 
#  Accuracy: 96.8%, Avg loss: 0.119539 

# Epoch 6
# -------------------------------
# Test Error: 
#  Accuracy: 96.9%, Avg loss: 0.119042 

# Epoch 7
# -------------------------------
# Test Error: 
#  Accuracy: 97.1%, Avg loss: 0.108375 

# Epoch 8
# -------------------------------
# Test Error: 
#  Accuracy: 97.0%, Avg loss: 0.118242 

# Epoch 9
# -------------------------------
# Test Error: 
#  Accuracy: 96.9%, Avg loss: 0.137353 

# Epoch 10
# -------------------------------
# Test Error: 
#  Accuracy: 97.0%, Avg loss: 0.121264 

# Done!
