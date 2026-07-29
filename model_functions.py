import torch
from torch import nn


class LinearNN(nn.Module):
    """Neural Network of 784->512->512->10"""
    def __init__(self, dr):
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

class Conv2DNN(nn.Module):
    """Neural Network of Conv2d(1,16) -> Conv2d(16,32) -> 32*7*7 -> 256 -> 10"""
    def __init__(self, dr):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),                             # final linear layer
            nn.Linear(32*7*7, 256),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        return self.network(x)


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
    """Loop that evalutes test data and outputs accuracy and loss"""
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
    print(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return correct