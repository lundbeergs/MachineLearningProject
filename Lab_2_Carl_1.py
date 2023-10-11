import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load in the dataset
x_test_regression = np.load('Xtest_Classification1.npy')
x_train_regression = np.load('Xtrain_Classification1.npy')
y_train_regression = np.load('ytrain_Classification1.npy')


# Split the dataset into training
X_train, X_temp, y_train, y_temp = train_test_split(x_train_regression, y_train_regression, test_size=0.3, random_state=39)

# Split part of the training set into validation
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=39)

# Create a Model class that inherits nn.Module
class Model(nn.Module):
    """
    Neural Network Model for training and testing

    # Input layers: 2352
        There are 2352 elements in our training dataset
    
    # Hidden layers: 3
        h1, h2, two hidden layers to begin with

    # Output: 2
        The outdata is either 0 or 1
    """
    def __init__(self, in_features=2352, h1=1500, h2=700, h3=4, out_features=2):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_features)

    def forward (self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)

        return x
    
if __name__ == '__main__':

    # Initialize the model
    torch.manual_seed(39)
    model = Model()

    # Convert X features into Float tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    X_valid = torch.FloatTensor(X_valid)

    # Convert the y labels to Long tensors
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    y_valid = torch.LongTensor(y_valid)

    # Set the criterion of model to measure the error, how far off the predictions are from the data
    criterion = nn.CrossEntropyLoss()
    
    # Choose The Gradient Decent Optimizer, lr = learning rate (if error dosen't go down after a bunch of iterations (epox), lower our learning rate) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    # Training the model section

    # One epoch is one run of all the data through our neural network
    epochs = 100
    losses = []
    valid_losses = []

    # Define variables to keep track of early stopping
    best_valid_loss = float('inf')  # Initialize with a very high value
    patience = 5  # Number of epochs to wait for improvement
    wait = 0  # Counter for the number of epochs without improvement

    for i in range(epochs):

        # Training phase
        model.train()
        optimizer.zero_grad()

        # Go forward and get a prediction
        y_pred = model.forward(X_train) # Get predicted results

        # Measure the loss/error, gonna be high at frist
        loss = criterion(y_pred, y_train)

        # Keep Track of our losses
        losses.append(loss.detach().numpy())

        # Do some back propagation: take the error rate forward propagation and feed it
        # through the network to fine tune the weights
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            y_valid_pred = model(X_valid)
            valid_loss = criterion(y_valid_pred, y_valid)
            valid_losses.append(valid_loss)

         # Check for early stopping based on validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss  # Update the best validation loss
            wait = 0  # Reset the counter
        else:
            wait += 1

        if wait >= patience:
            print(f'Early stopping after {i+1} epochs without improvement.')
            break  # Stop training
        
        # Print model status every 10 epoch
        if i % 10 == 0:
            print(f'Epoch: {i} and loss: {loss} and validation loss: {valid_loss}')
    
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test)
        test_loss = criterion(y_test_pred, y_test)

    print(f'Final Test Loss: {test_loss}')

    # Additional code for visualizing performance
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()