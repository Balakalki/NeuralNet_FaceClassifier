import numpy as np

# Layer Class:
# Represents a single layer in a neural network.
class Layer:
  # Constructor:
  # Initializes the Layer object with the provided parameters.
  #
  # Parameters:
  # - nNeurons: The number of neurons in the layer.
  # - weights: The weights matrix representing connections between neurons in the current layer and the previous layer.
  # - bias: The bias vector for the neurons in the current layer.
  # - activation_function: The activation function applied to the output of the layer.
  # - activation_derivative: The derivative of the activation function, used for backpropagation.
  def __init__(self,nNeurons,weights,bias,activation_function,activation_derivative):
    self.nNeurons = nNeurons
    self.weights = weights
    self.bias = bias
    self.activation_function = activation_function
    self.activation_derivative = activation_derivative
    self.delta = None

  # Method: calculate_output
  # Computes the output of the layer given the input data.
  # If input_Layer is True, assumes the layer is the input layer and directly sets the output to the input data.
  # Otherwise, computes the weighted sum of the input data, adds the bias, applies the activation function, and stores the output.
  #
  # Parameters:
  # - data: The input data to the layer.
  # - input_Layer: Boolean flag indicating whether the layer is the input layer.
  #
  # Returns:
  # - output: The output of the layer.
  def calculate_output(self,data,input_Layer=False):
    if(input_Layer):
      self.output = data
    else:
      self.weighted_sum = np.dot(data,self.weights)+self.bias
      self.output = np.array(self.activation_function(self.weighted_sum))
    return self.output



# Network Class:
# Represents a neural network composed of multiple layers.
class Network:
  # Constructor:
  # Initializes the Network object with the number of layers.
  #
  # Parameters:
  # - nLayers: The number of layers in the network.
  def __init__(self,nLayers):
    self.nLayers=nLayers
    self.Layers=[]
    self.pred_output=[]
  # Activation Functions:
  # Define various activation functions used in the network.

  # Linear activation function.
  def linear(self,x):
    return x
  # ReLU (Rectified Linear Unit) activation function.
  def relu(self,x):
    return np.maximum(0,x)
  # Derivative of the ReLU activation function.
  def relu_derivative(self,x):
    return np.where(x < 0, 0, 1)
  # Softmax activation function.
  def softmax(self,x):
    exp_values=np.exp(x)
    expSum=np.sum(exp_values)
    return exp_values/expSum
  # Derivative of the softmax activation function.
  def softmax_derivative(self,x):
    return self.softmax(x) * (1 - self.softmax(x))


  # Method: Create_Network
  # Creates the network architecture by adding layers with specified parameters.
  #
  # Parameters:
  # - nInputs: The number of input neurons to the network.
  def Create_Network(self,nInputs):
    for i in range(self.nLayers-1):
      if i==0:
        weights=np.ones(nInputs)
        bias=np.ones((nInputs))
        self.Layers.append(Layer(nInputs,weights,bias,self.linear, self.linear))
      else:
        nNeurons=int(input(f"enter number of neurons of Layer {i+1} "))
        weights=0.1*np.random.randn(nInputs,nNeurons)
        bias=0.1*np.random.randn((nNeurons))
        self.Layers.append(Layer(nNeurons,weights,bias,self.relu, self.relu_derivative))
        nInputs=nNeurons
    nNeurons=int(input(f"enter number of neurons in output layer "))
    weights=0.1*np.random.randn(nInputs,nNeurons)
    bias=np.zeros((nNeurons))
    self.Layers.append(Layer(nNeurons,weights,bias,self.softmax, self.softmax_derivative))

  # Method: forwardPass
  # Performs a forward pass through the network.
  #
  # Parameters:
  # - Input_data: The input data to be passed through the network.
  #
  # Returns:
  # - Output_data: The output data produced by the network.
  def forwardPass(self,Input_data):
    for i in range(self.nLayers):
      if(i==0):
        Input_data=self.Layers[i].calculate_output(Input_data,True)
      else:
        Input_data=self.Layers[i].calculate_output(Input_data)
    return Input_data

  # Method: calculate_deltas
  # Calculates delta values for each layer in the network during backpropagation.
  #
  # Parameters:
  # - targets: The target values for the current training batch.
  def calculate_deltas(self, targets):
    for i in range(self.nLayers - 1, 0, -1):
      if i == len(self.Layers)-1:
        self.Layers[i].delta = (self.Layers[i].output - targets)
      else:
        self.Layers[i].delta = np.dot(self.Layers[i+1].delta,self.Layers[i+1].weights.T) * self.Layers[i].activation_derivative(self.Layers[i].weighted_sum)

  # Method: Update_Weights
  # Updates the weights and biases of each layer in the network using gradient descent.
  #
  # Parameters:
  # - lr: The learning rate for gradient descent.
  def Update_Weights(self,lr):
    for i in range(self.nLayers - 1, 0, -1):
      self.Layers[i].bias -= np.dot(lr, self.Layers[i].delta)
      self.Layers[i].weights -= self.Layers[i].delta[np.newaxis,:] * (np.dot(lr,self.Layers[i-1].output)[:, np.newaxis]  * self.Layers[i].weights)

  # Method: calculate_error
  # Calculates the error between predicted outputs and target values.
  #
  # Parameters:
  # - targets: The target values for the current training batch.
  # - outputs: The predicted outputs produced by the network.
  #
  # Returns:
  # - loss: The error between predicted outputs and target values.
  def calculate_error(self, targets, outputs):
      epsilon = 1e-15
      outputs = np.clip(outputs, epsilon, 1 - epsilon)
      loss = - np.sum(targets * np.log(outputs))
      return loss

  # Method: backwardPass
  # Performs a backward pass through the network (backpropagation) to update weights and biases.
  #
  # Parameters:
  # - targets: The target values for the current training batch.
  # - lr: The learning rate for gradient descent.
  def backwardPass(self,targets, lr):
    self.calculate_deltas(targets)
    self.Update_Weights(lr)

  # Method: One_hot
  # Encodes the target labels using one-hot encoding.
  #
  # Parameters:
  # - x: The target label to be encoded.
  #
  # Returns:
  # - ans: The one-hot encoded target label.
  def One_hot(self,x):
    ans = []
    for i in range(4):
      if(i==x):
        ans.append(1)
      else:
        ans.append(0)
    return np.array(ans)


  # Method: fit
  # Trains the network on the provided input data and target labels for the specified number of epochs.
  #
  # Parameters:
  # - input_datas: The input data for training.
  # - target_labels: The target labels for training.
  # - epochs: The number of training epochs.
  # - learning_rate: The learning rate for gradient descent.
  def fit(self, input_datas, target_labels, epochs, learning_rate):
      for epoch in range(epochs):
        total_error = 0
        for input_data,target in zip(input_datas, target_labels):
          outputs = self.forwardPass(input_data)
          targets = self.One_hot(target)
          error = self.calculate_error(targets, outputs)
          total_error+=error
          self.backwardPass(targets, learning_rate)
        print(f"Epoch {epoch + 1} / {epochs}, Error: {total_error/len(input_datas)}")

  # Calculate the accuracy of the neural network on the test dataset.
  # Parameters:
  # - xtest (array-like): Input test data.
  # - ytest (array-like): True labels for the test data.
  # Returns:
  # - accuracy (float): Accuracy of the neural network on the test dataset.
  def calculate_accuracy(self, xtest, ytest):
    correct_predictions = 0
    total_samples = len(ytest)
    for input_data, output_data in zip(xtest, ytest):
      output = self.forwardPass(input_data)
      predicted_class = np.argmax(output)
      if predicted_class == output_data and output[predicted_class] >= 0.9:
        correct_predictions += 1
    accuracy = correct_predictions / total_samples
    return accuracy

nLayers=int(input("enter number of layers"))

obj=Network(nLayers)  #object for Network class
obj.Create_Network(4096)  #creates a network by creating layers and appending to Layers list in Network class (this takes input length as argument)
