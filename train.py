# network creation
nLayers=int(input("enter number of layers"))
obj=Network(nLayers)  #object for Network class
obj.Create_Network(4096)  #creates a network by creating layers and appending to Layers list in Network class (this takes input length as argument)


# obj.fit(flatten_inputs, ytrain, 1000, 0.1)
# print(obj.Layers[-1].output)

obj.fit(xtrain, ytrain, 30, 0.003) # store the training inputs in xtrain and their corresponding labels in ytrain
