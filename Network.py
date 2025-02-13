import numpy as np
'''Layer class which stores
  Layer weights matrix,
  Layer bias ,
  activation function of layer and
  output of Layer
  (Output should be stored when we call calculate_output function)
'''

class Layer:
  def __init__(self,nNeurons,weights,bias,activation_function,activation_derivative):
    self.nNeurons = nNeurons
    self.weights = weights
    self.bias = bias
    self.activation_function = activation_function
    self.activation_derivative = activation_derivative
    self.delta = None
  def calculate_output(self,data,input_Layer=False):#returns output which should be used as input for the next layer
    if(input_Layer):#for input layer the weight matrix should be transposed because input layer only takes one feature of input  not entire input
      self.output = np.array(data)
    else:
      self.weighted_sum = np.dot(data,self.weights)+self.bias
      self.output = np.array(self.activation_function(self.weighted_sum))#stores the outputs of Layer which contains all neurons outputs of that particular layer
    return self.output



'''
Network class which contains
number of layers as nLayers and
list containing all layer objects

has a create_Network functino which used to create weights and bias for entire network

forwardPass function is used to pass the input to the network through input layers and travels to the output layer through hidden layers
calculate_error function takes the target output as argument and calculates the error between target output and network output
'''

class Network:
  def __init__(self,nLayers):
    self.nLayers=nLayers
    self.Layers=[]
    self.pred_output=[]


  def sigmoid(self,x):
    return 1/(1+np.exp(x))
  def sigmoid_derivative(self,x):
    return self.sigmoid(x)*(1-self.sigmoid(x))
  def input_act(self,x):
    return x
  def relu(self,x):
    return np.maximum(0,x)
  def relu_derivative(self,x):
    return np.where(x < 0, 0, 1)
  def softmax(self,x):
    exp_values=np.exp(x)
    expSum=np.sum(exp_values)
    return exp_values/expSum
  def softmax_derivative(self,x):
    # return self.softmax(x) * (1 - self.softmax(x))
    s = self.softmax(x).reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)



  def Create_Network(self,nInputs):
    for i in range(self.nLayers-1):
      if i==0:#for input layer each neuron has single weight as it takes single input
        weights=np.ones(nInputs)
        bias=np.ones((nInputs))
        self.Layers.append(Layer(nInputs,weights,bias,self.input_act, self.input_act))
      else:
        nNeurons=int(input(f"enter number of neurons of Layer {i+1} "))
        weights=0.1*np.random.randn(nInputs,nNeurons)
        bias=0.1*np.random.randn((nNeurons))
        self.Layers.append(Layer(nNeurons,weights,bias,self.relu, self.relu_derivative))
        nInputs=nNeurons
    #output layer which takes different activation function
    nNeurons=int(input(f"enter number of neurons in output layer "))
    weights=0.1*np.random.randn(nInputs,nNeurons)
    bias=np.zeros((nNeurons))
    self.Layers.append(Layer(nNeurons,weights,bias,self.softmax, self.softmax_derivative))




  def forwardPass(self,Input_data):
    for i in range(self.nLayers):
      if(i==0):
        Input_data=self.Layers[i].calculate_output(Input_data,True)
      else:
        Input_data=self.Layers[i].calculate_output(Input_data)
    return Input_data



  def One_hot(self,x):
    ans = []
    for i in range(4):
      if(i==x):
        ans.append(1)
      else:
        ans.append(0)
    return np.array(ans)


  def calculate_deltas(self, targets):
    for i in range(self.nLayers - 1, 0, -1):
      if i == len(self.Layers)-1:
        self.Layers[i].delta = (self.Layers[i].output - targets)
      else:
        self.Layers[i].delta = np.dot(self.Layers[i+1].delta, self.Layers[i+1].weights.T) * self.Layers[i].activation_derivative(self.Layers[i].weighted_sum)


  def Update_Weights(self,lr):
    for i in range(self.nLayers - 1, 0, -1):
      # self.Layers[i].bias -= np.dot(lr, self.Layers[i].delta)
      # self.Layers[i].weights -= self.Layers[i].delta[np.newaxis,:] * (np.dot(lr,self.Layers[i-1].output)[:, np.newaxis] * self.Layers[i].weights)
      self.Layers[i].weights -= lr * np.dot(self.Layers[i-1].output.T[:, np.newaxis], self.Layers[i].delta[np.newaxis, :])
      self.Layers[i].bias -= lr * self.Layers[i].delta





  def calculate_error(self, targets, outputs):
    return -np.sum(targets * np.log(outputs))

      # epsilon = 1e-15
      # outputs = np.clip(outputs, epsilon, 1 - epsilon)
      # loss = -np.sum(targets * np.log(outputs))
      # return loss



  def backwardPass(self,targets, lr):
    self.calculate_deltas(targets)
    self.Update_Weights(lr)



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


nLayers=int(input("enter number of layers"))
obj=Network(nLayers)  #object for Network class
obj.Create_Network(4096)  #creates a network by creating layers and appending to Layers list in Network class (this takes input length as argument)


# obj.fit(flatten_inputs, ytrain, 1000, 0.1)
# print(obj.Layers[-1].output)
