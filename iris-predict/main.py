import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
import torch
from torch_lr_finder import LRFinder

def prepare_data():
    sns.set()
    # Load the dataset
    iris = sns.load_dataset("iris") 

    # Filter Setosa species
    df = iris[iris.species != "setosa"]

    # Display pair relationships between variables
    g = sns.pairplot(df, hue="species") 
    #plt.show()

    # Create a new column and map species names to numeric values
    df['species_n'] = iris.species.map({'versicolor':1, 'virginica':2})

    # Create a NumPy array, select 'sepal_length' and 'sepal_width'; 'petal_length' and 'petal_width'
    X_iris = np.asarray(df.loc[:, ['sepal_length', 'sepal_width']], dtype=np.float32) 
    Y_iris = np.asarray(df.loc[:, ['petal_length', 'petal_width']], dtype=np.float32)

    # Create a NumPy array containing the labels for the dataset
    label_iris = np.asarray(df.species_n, dtype=int)

    # StandardScaler is a class for scaling numerical data
    # fit_transform assigns a scale to the data and then scales the data
    # the result is a new numpy array, transformed to have a mean of 0 and a standard deviation of 1
    # this ensures that each feature is equally important during the learning process
    X_iris = StandardScaler().fit_transform(X_iris)
    Y_iris = StandardScaler().fit_transform(Y_iris)

    # Split data into training and testing sets
    # The argument train_size specifies that the training set should be 50% of the entire dataset
    # The argument stratify ensures that the split is made based on the label_iris vector
    X_train, X_val, Y_train, Y_val, label_train, label_val = \
        sklearn.model_selection.train_test_split(X_iris, Y_iris, label_iris, train_size=0.5, stratify=label_iris)
    
    return X_train, X_val, Y_train, Y_val, label_train, label_val

def create_model(input_dim, output_dim, learning_rate):
    # Sequential object is a container for a series of PyTorch modules,
    # where the output of one module is used as input to the next module
    model = torch.nn.Sequential(
        # Linear layer takes 'input_dim' input elements and creates 100 output elements
        torch.nn.Linear(input_dim, 128),
        # If input is positive, output is same as input, otherwise, output = 0
        torch.nn.ReLU(),
        # Takes 100 output elements from ReLU and creates 'output_dim' output elements
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, output_dim),
    )
    # Optimizer takes model parameters (i.e., weights of the linear layers) and updates them during training
    # to minimize the loss function. It also adjusts the learning rate during training
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Mean squared error loss
    # The loss is the sum of squared errors across all samples
    loss_fn = torch.nn.MSELoss(reduction='sum')
    
    return model, optimizer, loss_fn

def train_model(X_train, Y_train, X_val, Y_val, model, optimizer, loss_fn, nite):
    losses_train, losses_val = [], []

    # Converted from NumPy arrays to PyTorch tensors
    X_train = torch.from_numpy(X_train)
    Y_train = torch.from_numpy(Y_train)
    X_val = torch.from_numpy(X_val)
    Y_val = torch.from_numpy(Y_val)

    for t in range(nite):

        # Predictions based on training input data
        y_pred = model(X_train)

        # Training loss is calculated using loss_fn, predicted output y_pred, and target output Y_train
        loss = loss_fn(y_pred, Y_train)
        # Gradients are set to zero to prevent accumulation of previous gradients
        optimizer.zero_grad()
        # Computes gradients for all model parameters
        loss.backward()
        # Optimizer updates model parameters using computed gradients
        optimizer.step()

        # with torch.no_grad() disables gradient computations to speed up calculations
        with torch.no_grad():
            # Validation loss is calculated using
            # the given loss function loss_fn, predicted output y_pred, and target output Y_val
            y_pred = model(X_val)
            loss_val = loss_fn(y_pred, Y_val)
            
        if t % 20 == 0:
             print(t, loss.item(), loss_val.item())
        losses_train.append(loss.item())
        losses_val.append(loss_val.item())

    return model, losses_train, losses_val

def evaluate_model(X, Y, model):
    with torch.no_grad():
        Y_pred = model(torch.from_numpy(X))
        loss = torch.nn.functional.mse_loss(Y_pred, torch.from_numpy(Y))
        return Y_pred.numpy()

def main():
    # Read and prepare data
    X_train, X_val, Y_train, Y_val, label_train, label_val = prepare_data()

    # Create model
    model, optimizer, loss_fn = create_model(X_train.shape[1], Y_train.shape[1], learning_rate = 1e-4)

    # Train model
    model, losses_train, losses_val = train_model(X_train, Y_train, X_val, Y_val, model, optimizer, loss_fn, 300)

    # Evaluate model
    Y_pred = evaluate_model(X_val, Y_val, model)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    
    print('Classification report:\n', sklearn.metrics.classification_report(label_val, Y_pred_classes))
    # Plot losses
    plt.plot(losses_train, label='Training loss')
    plt.plot(losses_val, label='Validation loss')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()
