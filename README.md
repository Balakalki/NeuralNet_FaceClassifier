# NeuralNet_FaceClassifier
 An ML project using a fully customizable Artificial Neural Network (ANN) built from scratch in Python (with NumPy and Pandas). Designed to handle various classification tasks, including facial image recognition and handwritten digit classification. Initially build for Face Classification


# ANN-Based Classifier
# Overview
 This project implements a neural network-based classifier from scratch using Python. It includes two main classes: Layer and Network, which handle the construction and training of the neural network. The Layer class represents individual layers in the network, while the Network class manages the overall network architecture and training process. The model is adaptable and can be used for multiple classification tasks by modifying input attributes.


# Dataset

 The model has been tested on multiple datasets, demonstrating its ability to generalize across different domains

# Face Classification Dataset

Consists of 1872 images of 20 individuals with different facial expressions and poses.
Images are preprocessed to a size of 64x64 pixels.
Stored in CSV format with each row representing a flattened array of 4096 pixels (64x64) along with a corresponding label.
Labels indicate face orientation:
0: Left-facing
1: Right-facing
2: Upward-facing
3: Straight-facing

**Additional Information:**
 The dataset is provided in CSV format, with each row representing an image and its corresponding label. Each image is represented as a flattened array of 4096 pixels (64x64), stored in the columns of the CSV file. Additionally, there is an extra column to store the labels of each image, where:

- 0 represents the face is facing to the left

- 1 represents the face is facing to the right
 
- 2 represents the face is facing upwards
 
- 3 represents the face is facing straight.


The dataset was created by processing grayscale images from [this source](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-8/faceimages/faces), which provides 20 folders with corresponding person names. The original images in the URL are in PGM format, and they were resized to 64x64 pixels before being processed into the CSV file.

# MNIST Handwritten Digits Dataset
Consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels.
Preprocessed into a flattened format for compatibility with the neural network.
The model achieved 90%+ accuracy on the MNIST dataset, showcasing its adaptability for digit classification tasks.

# Model Architecture
The neural network consists of three layers, including one hidden layers, one input layer, and one output layer:

Input Layer: Accepts input features (e.g., 64x64 for face classification, 28x28 for MNIST).
Hidden Layers: Utilize ReLU activation function to introduce non-linearity.                    
Output Layer: Uses Softmax activation function for multi-class classification.

The number of neurons in each layer can be customized depending on the dataset and classification task.

# Results

Successfully classifies facial orientations with high accuracy.
Demonstrates strong generalization capabilities across different datasets.
Can be adapted for any classification problem requiring an ANN-based approach.

# Usage
**1. Navigate to google colab ANN_(Mnist and Face Classifier)**

**2. Create the Neural Network:**
   
- Define the network architecture by setting the number of layers and neurons per layer in the Network class constructor. For example, if you want to create a 3-layer network with the following architecture:

  - Layer 1: 64x64 input neurons
  - Layer 3: 512 neurons
  - Output Layer: 4 neurons

  You can initialize the network as follows:
  
 python
 
 *Create the network object*
 
 neural_network = Network(nLayers=6)
 
 *Define the architecture and train the network*
 
 neural_network.Create_Network(nInputs=64*64) # Input Layer
 
 neural_network.Create_Network(512)   # Layer 2
 
 neural_network.Create_Network(4)     # Output Layer
  
**3. next run the shell in colab which is calling fit method**

**4. Evaluate Accuracy:**

 After training the model, you can evaluate its accuracy on the test dataset using the calculate_accuracy method. Pass the test data and corresponding labels to this method to obtain the accuracy.

 *Example:*

 #Calculate accuracy

 run the accuracy shell in colab after training


**5. Make Predictions:**

 Once the network is trained, you can use it to make predictions on new facial images. Pass the input data to the forwardPass method of the Network object to obtain the predicted outputs.

 *Example:*

 #Make predictions

 prediction = neural_network.forwardPass(input_data)

 print("Prediction:", prediction)


**6. Explore and Customize:**

 Modify the network architecture, activation functions, and training parameters to adapt to different classification tasks.


# Contributors
Aluri Bala Kalki

# License
This project is licensed under the MIT License.

