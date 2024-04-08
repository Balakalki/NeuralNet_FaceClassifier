# NeuralNet_FaceClassifier
 An AI project using neural networks built from scratch in Python (with NumPy and Pandas) for facial image classification. Dive into deep learning and explore facial recognition techniques.


# Face Classifier Neural Network
# Overview
This project implements a neural network-based face classifier from scratch using Python. It includes two main classes: Layer and Network, which handle the construction and training of the neural network. The Layer class represents individual layers in the network, while the Network class manages the overall network architecture and training process.


# Dataset

The dataset consists of 1872 images of 20 individuals with different facial expressions and poses. Each image is preprocessed to a size of 64x64 pixels before being fed into the neural network.

**Additional Information:**
The dataset is provided in CSV format, with each row representing an image and its corresponding label. Each image is represented as a flattened array of 4096 pixels (64x64), stored in the columns of the CSV file. Additionally, there is an extra column to store the labels of each image, where:

- 0 represents the face is facing to the left

- 1 represents the face is facing to the right
 
- 2 represents the face is facing upwards
 
- 3 represents the face is facing straight.


The dataset was created by processing grayscale images from [this source](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-8/faceimages/faces), which provides 20 folders with corresponding person names. The original images in the URL are in PGM format, and they were resized to 64x64 pixels before being processed into the CSV file.


# Model Architecture
The neural network architecture comprises 6 layers, including 4 hidden layers, one input layer, and one output layer. The hidden layers utilize ReLU activation function to introduce non-linearity, while the output layer employs softmax activation function for multi-class classification.

**Output Layer:**
The output layer consists of four neurons, each representing one direction in which the face is facing: left, right, up, and straight. The labels in the dataset correspond to these directions, with 0 representing left, 1 representing right, 2 representing up, and 3 representing straight.



# Training and Testing
The dataset is split into training and testing sets using the train_test_split function. The model is trained on the training set and evaluated on the testing set to assess its performance.

# Results

After training the model, it achieved the best results with the 6-layer architecture. The trained model demonstrates the ability to classify facial images accurately, distinguishing between different individuals and facial expressions. The network achieved an impressive accuracy of 99% on the test dataset, showcasing its effectiveness in face classification tasks.


# Usage
**1. Clone the Repository:**
   
git clone https://github.com/Balakalki/NeuralNet_FaceClassifier.git

**2. Navigate to the Project Directory:**
   
cd NeuralNet_FaceClassifier

**3. Install Dependencies:**
   
Ensure you have Python installed on your system. Install the required libraries using pip:

pip install numpy pandas

**4. Create and Train the Neural Network:**
   
-> Define the network architecture by setting the number of layers and neurons per layer in the Network class constructor.

-> Initialize the network using the Create_Network method, specifying the number of input neurons.

-> Train the network by calling the fit method with the input data and target labels, along with the desired number of epochs and learning rate.

*Example:*

#Create the network object

neural_network = Network(nLayers=3)

#Define the architecture and train the network

neural_network.Create_Network(nInputs=64*64)

neural_network.fit(input_datas, target_labels, epochs=100, learning_rate=0.01)


**5. Evaluate Accuracy:**

After training the model, you can evaluate its accuracy on the test dataset using the calculate_accuracy method. Pass the test data and corresponding labels to this method to obtain the accuracy.

*Example:*

#Calculate accuracy

accuracy = neural_network.calculate_accuracy(xtest, ytest)

print("Accuracy:", accuracy)


**6. Make Predictions:**

Once the network is trained, you can use it to make predictions on new facial images. Pass the input data to the forwardPass method of the Network object to obtain the predicted outputs.

*Example:*

#Make predictions

prediction = neural_network.forwardPass(input_data)

print("Prediction:", prediction)


**7. Explore and Customize:**

Feel free to explore and customize the network architecture, activation functions, and training parameters according to your requirements.


# Contributors
Aluri Bala Kalki

# License
This project is licensed under the MIT License.

