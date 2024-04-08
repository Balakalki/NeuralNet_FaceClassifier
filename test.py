import numpy as np
from network import Network  # Import the Network class from the network module

# Define your test data and labels
# Example:
# x_test = ...
# y_test = ...

# Initialize your neural network
nLayers = 6  # Example: Number of layers
obj = Network(nLayers)

# Create the network architecture
obj.Create_Network(4096)

# Load trained weights (if applicable)
# Example:
# obj.load_weights('trained_weights')

# Calculate accuracy on test data
accuracy = obj.calculate_accuracy(x_test, y_test)
print("Accuracy:", accuracy)
