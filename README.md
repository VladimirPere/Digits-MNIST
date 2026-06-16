Data From: "MNIST Digit Recognizer" by ANIMATRONBOT
Found Here: https://www.kaggle.com/datasets/animatronbot/mnist-digit-recognizer

Data: 42,000 samples of 784 features. Features are pixels in a 28x28 grid that make a black and white image of a digit (0-9).
Model: Fully connected MLP (784 → 512 → 512 → 10) built in PyTorch

# Process
- Transform data into a pytorch tensor.
- Split data into training and testing data and put into data loader.
- Create a Neural Network that flattens the data and puts it through 2 hidden layers with 512 node. Using ReLU as the activation function. 512 was used for easier processing while still being accurate.
- Training loop created where data is inputted into the neural network, loss is calculated and stored in loss function and then optimized.
- Test loop created. Test loss and accuracy is measured during each epoch. Total loss and accuracy is then printed.
- Cross Entropy Loss used for simplicity and effective multi-class classification.
- Adam optimizer chosen for its adaptive learning rate and strong general performance on neural networks.
- Epoch loop created for training and testing data.

# Final Results
The model increases it's accuracy over the first 5 epochs, after which the accuracy oscillates between 96% and 97% accuracy. More hyperparameter tuning needed for better results.
A learning rate scheduler can further improve results

# Outputs
Epoch 1
-------------------------------
Test Error: 
Accuracy: 95.3%, Avg loss: 0.158502 

<Epochs 2-9 omitted>

Epoch 10
-------------------------------
Test Error: 
Accuracy: 96.6%, Avg loss: 0.159814 
