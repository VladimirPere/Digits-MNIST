import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from model_functions import LinearNN, Conv2DNN, train_loop, test_loop
import os
from pathlib import Path
import time

# csv folder directory
BASE_DIR = Path(__file__).parent
CSV_DIR = BASE_DIR/'csv'

data = pd.read_csv(CSV_DIR/'pytorchtrain.csv')

target_name = 'label'
target = data[target_name]
target = torch.tensor(target.values, dtype=torch.int64)
training_data = data.drop(columns=[target_name])
training_data = torch.tensor(training_data.values, dtype=torch.float32)

train_data, test_data, train_target, test_target = train_test_split(training_data, target, random_state=4, test_size=.3)

# variables for training data
lr = 5e-4
dr = .2
epochs = 25
batch_size = 128

train_dataset = TensorDataset(train_data, train_target)
test_dataset = TensorDataset(test_data, test_target)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# neural network models
lin_model = LinearNN(dr=dr)
conv_model = Conv2DNN(dr=dr)

# type of loss function used
loss_fn = nn.CrossEntropyLoss()

# optimizers: does gradient descent at learning rate
lin_optimizer = torch.optim.Adam(lin_model.parameters(), lr=lr)
conv_optimizer = torch.optim.Adam(conv_model.parameters(), lr=lr)

# schedulers: reduces learning rate by {1-factor} if accuracy does not increase for {patience} epochs
lin_scheduler = lr_scheduler.ReduceLROnPlateau(lin_optimizer, 
                                           mode='max', 
                                           patience=2, 
                                           factor=.5)
conv_scheduler = lr_scheduler.ReduceLROnPlateau(conv_optimizer, 
                                           mode='max', 
                                           patience=2, 
                                           factor=.3)

# linear neural network epoch loop and time
lin_acc = []

start_time = time.perf_counter()
for t in range(epochs):
    train_loop(train_dataloader, lin_model, loss_fn, lin_optimizer)
    lin_acc.append(test_loop(test_dataloader, lin_model, loss_fn))
    lin_scheduler.step(lin_acc[-1])
end_time = time.perf_counter()
print(f"Time for Linear Neural Network: {(end_time - start_time):.4f} seconds")

# conv2d neural network epoch loop and time
conv_acc = []

start_time = time.perf_counter()
for t in range(epochs):
    train_loop(train_dataloader, conv_model, loss_fn, conv_optimizer)
    conv_acc.append(test_loop(test_dataloader, conv_model, loss_fn))
    conv_scheduler.step(conv_acc[-1])
end_time = time.perf_counter()
print(f"Time for Conv2D Neural Network: {(end_time - start_time):.4f} seconds")

# # list of accuracies per epoch
# rounded = [f"{n:.3f}" for n in lin_acc]
# print(f'Final accuracies for Linear Neural Network:\n{rounded}')
# rounded = [f"{n:.3f}" for n in conv_acc]
# print(f'Final accuracies for Conv2D Neural Network:\n{rounded}')


# ploting test accuracies per epoch
plt.figure(figsize=(10, 6))
x = np.arange(1, epochs + 1, step=1)
plt.plot(x, lin_acc, c='red', marker='o', label='Test Accuracy - Linear Neural Network')
plt.plot(x, conv_acc, c='blue', marker='x', label='Test Accuracy - Conv2D Neural Network')

# plot lines at max accuracy for cleaner understanding
plt.axhline(y=max(lin_acc), 
            color='red', 
            linestyle=':', 
            alpha=0.4, 
            label=f'Linear best: {max(lin_acc):.3f}')
plt.axhline(y=max(conv_acc), 
            color='blue', 
            linestyle=':', 
            alpha=0.4,
            label=f'Conv2D best: {max(conv_acc):.3f}')

# labels
plt.xlabel('Epoch')
plt.xticks(np.arange(0, epochs + 1, step=1))
plt.ylabel('Accuracy')
plt.title('MNIST Digits Classification - Test Accuracy per Epoch')
plt.legend()
plt.tight_layout()
os.makedirs(BASE_DIR / 'images', exist_ok=True)
plt.savefig(BASE_DIR / 'images/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()