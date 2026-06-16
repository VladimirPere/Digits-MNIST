import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn

data = pd.read_csv('/home/vladimir/Downloads/Data/pytorchtrain.csv')

target_name = 'label'
target = data[target_name]
target = torch.tensor(target.values, dtype=torch.int64)
training_data = data.drop(columns=[target_name])
training_data = torch.tensor(training_data.values, dtype=torch.float32)

train_data, test_data, train_target, test_target = train_test_split(training_data, target, random_state=4, test_size=.3)

# Variables for training data
learning_rate = 1e-3
epochs = 10
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
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

def train_loop(dataloader, model, loss_fn, optimizer):
    # Set the model to training mode
    model.train()
    for X, y in dataloader:
        # Compute prediction and loss
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad()
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Finding Accuracy
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



#Type of loss function used
loss_fn = nn.CrossEntropyLoss()

#Optimizer: Does gradient descent at learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Epoch Loop
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")


        # OUTPUTS
# Epoch 1
# -------------------------------
# Test Error: 
#  Accuracy: 95.3%, Avg loss: 0.158502 

# Epoch 2
# -------------------------------
# Test Error: 
#  Accuracy: 95.1%, Avg loss: 0.172692 

# Epoch 3
# -------------------------------
# Test Error: 
#  Accuracy: 95.7%, Avg loss: 0.161926 

# Epoch 4
# -------------------------------
# Test Error: 
#  Accuracy: 96.0%, Avg loss: 0.149921 

# Epoch 5
# -------------------------------
# Test Error: 
#  Accuracy: 96.2%, Avg loss: 0.176076 

# Epoch 6
# -------------------------------
# Test Error: 
#  Accuracy: 96.0%, Avg loss: 0.182651 

# Epoch 7
# -------------------------------
# Test Error: 
#  Accuracy: 96.2%, Avg loss: 0.172156 

# Epoch 8
# -------------------------------
# Test Error: 
#  Accuracy: 96.7%, Avg loss: 0.155451 

# Epoch 9
# -------------------------------
# Test Error: 
#  Accuracy: 95.9%, Avg loss: 0.193775 

# Epoch 10
# -------------------------------
# Test Error: 
#  Accuracy: 96.6%, Avg loss: 0.159814 

# Done!
