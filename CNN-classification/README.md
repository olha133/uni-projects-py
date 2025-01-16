# Fashion MNIST Classification using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. The dataset consists of grayscale images representing 10 different fashion items. The images are in the form of (28, 28) pixel arrays, and the task is to train a CNN to accurately classify these images.

## Data
The Fashion MNIST dataset contains:
- 60,000 images in the **training set**
- 10,000 images in the **test set**

### Preprocessing
The images were transformed from the shape `(28, 28)` to `(1, 28, 28)` to adapt to the input requirements of the neural network. The pixel values were normalized by dividing them by 255, bringing the range of values to \([0, 1]\). This normalization helps the optimization algorithm converge faster during training.

## Creating the Neural Network
The model is defined by creating a class called `CNN`, which extends the `nn.Module` class. 

### Architecture
The `init()` method of the CNN class defines the layers of the model. The network consists of three layers, each responsible for different aspects of feature extraction and processing. 

The `forward()` method performs a forward pass over the input data, applying the defined layers to produce predictions.

## Training the Model
The model is trained using a function called `TrainModel`. During each epoch, the following steps are performed:
1. Calculate the start and end indices of the current batch of input data.
2. Perform a forward pass through the CNN to generate predictions.
3. Calculate the loss value.
4. Reset the gradients stored in the optimizer object.
5. Call the `backward()` method on the loss to compute the gradients.
6. Call the `step()` method on the optimizer to update the model's weights based on the computed gradients.

## Making Predictions
A function called `MakePredictions` is used to make predictions on the test set. It processes all batches and combines the individual predictions to create the final predictions for the entire dataset.
