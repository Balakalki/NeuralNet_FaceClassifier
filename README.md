# NeuralNet_FaceClassifier
 An AI project using neural networks built from scratch in Python (with NumPy and Pandas) for facial image classification. Dive into deep learning and explore facial recognition techniques.


# Face Classifier Neural Network
# Overview
This project implements a deep learning model for facial image classification using artificial neural networks (ANN). The model is built from scratch in Python, using NumPy and Pandas libraries. It leverages ReLU activation function in the hidden layers, softmax activation function in the output layer, and categorical cross-entropy loss function for training.

# Dataset
The dataset consists of 1872 images of 20 individuals with different facial expressions and poses. Each image is preprocessed to a size of 64x64 pixels before being fed into the neural network.

# Model Architecture
The neural network architecture comprises 6 layers, including 4 hidden layers, one input layer, and one output layer. The hidden layers utilize ReLU activation function to introduce non-linearity, while the output layer employs softmax activation function for multi-class classification.

# Training and Testing
The dataset is split into training and testing sets using the train_test_split function. The model is trained on the training set and evaluated on the testing set to assess its performance.

# Results
After training the model, it achieved the best results with the 6-layer architecture. The trained model demonstrates the ability to classify facial images accurately, distinguishing between different individuals and facial expressions.

# Usage
Clone the repository: git clone https://github.com/Balakalki/NeuralNet_FaceClassifier.git

Install the required dependencies: pip install numpy pandas

Run the train.py script to train the neural network on the provided dataset.

Use the trained model for facial image classification by running the predict.py script.

# Contributors
Aluri Bala Kalki

# License
This project is licensed under the MIT License.

Feel free to customize the content according to your project specifics and preferences!
