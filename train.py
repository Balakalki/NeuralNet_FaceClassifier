import numpy as np
from network import Network  # Import the Network class from the network module

# Define your training data and labels
# Example:
# x_train = ...
# y_train = ...

# Initialize your neural network
nLayers = 6  # Example: Number of layers
obj = Network(nLayers)

# Create the network architecture
obj.Create_Network(4096)

# Define training parameters
epochs = 300  # Example: Number of epochs
learning_rate = 0.005  # Example: Learning rate

# Train the network
obj.fit(x_train, y_train, epochs, learning_rate)
