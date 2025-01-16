# Iris Dataset Regression with PyTorch

This project implements a regression model using PyTorch to predict the petal length and petal width of Iris flowers based on their sepal length and sepal width.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Functions

### `prepare_data()`
Loads and preprocesses the Iris dataset, performs scaling, and splits the data into training and validation sets.

### `create_model(input_dim, output_dim, learning_rate)`
Creates a neural network model with 4 layers using PyTorch's `torch.nn.Sequential`.

### `train_model(X_train, Y_train, X_val, Y_val, model, optimizer, loss_fn, nite)`
Trains the model on the training data and evaluates it on the validation data for a specified number of iterations.

### `evaluate_model(X, Y, model)`
Evaluates the model on the provided data and returns the predicted values.

## Example Output

The output includes the classification report and loss curves.
