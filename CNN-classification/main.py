# importing libraries
import torch
from torch import nn
from torch.optim import RMSprop, Adam
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# convert to PyTorch tensor
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

#reshape images
x_train = x_train.reshape(-1,1,28,28)
x_test = x_test.reshape(-1,1,28,28)

# bring the numbers in the range [0,1]
x_train = x_train/255.0
x_test =  x_test/255.0

training =  y_train.unique()

# create Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.seq = nn.Sequential(
            # first convolution layer
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding="same"),
            # Rectified Linear Unit - activation function
            nn.ReLU(),

            # second convolution layer
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,3), padding="same"),
            nn.ReLU(),

            # third convolution layer
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3,3), padding="same"),
            nn.ReLU(),

            # flattening the output of the third layer
            nn.Flatten(),

            # use a linear layer
            nn.Linear(8*28*28, len(training)),      
        )

    def forward(self, x_batch):
        # forward pass
        predic = self.seq(x_batch)
        return predic
cnn = CNN()

predic = cnn(x_train[:5])

# training CNN
def TrainModel(model, loss_f, optimizer, x, y, batch_s=32, epochs=5):
    for i in range(epochs):
        # calculate start and end indexes of input data
        batches = torch.arange((x.shape[0] // batch_s) + 1)

        losses = [] ## Record loss of each batch
        for batch in batches:
            if batch != batches[-1]:
                start, end = int(batch*batch_s), int(batch * batch_s + batch_s)
            else:
                start, end = int(batch*batch_s), None

            x_batch, y_batch = x[start:end], y[start:end]

            # make predictions by forward pass through network
            predic = model(x_batch) 

            # calculate loss
            loss = loss_f(predic, y_batch)
            losses.append(loss)

            # zeros previous gradients 
            optimizer.zero_grad()

            # calculate gradients
            loss.backward()

            # update
            optimizer.step()

        print("Categorical Cross Entropy : {:.3f}".format(torch.tensor(losses).mean()))

# use cross entropy loss function
loss = nn.CrossEntropyLoss()

torch.manual_seed(42)

epochs = 15
learning_r = torch.tensor(1/1e3) # 0.001
batch_s=128

cnn = CNN()
cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = Adam(params = cnn.parameters(), lr=learning_r)

TrainModel(cnn, cross_entropy_loss, optimizer, x_train, y_train, batch_s = batch_s, epochs = epochs)

# do function that performs prediction on input data
def MakePredictions(model, input_data, batch_s=32):
    batches = torch.arange((input_data.shape[0] // batch_s)+1) ### Batch Indices

    # disables automatic gradients calculations
    with torch.no_grad(): 
        preds = []
        for batch in batches:
            if batch != batches[-1]:
                start, end = int(batch * batch_s), int(batch*batch_s + batch_s)
            else:
                start, end = int(batch * batch_s), None

            X_batch = input_data[start:end]

            preds.append(model(X_batch))

    return preds

test_predic = MakePredictions(cnn, x_test, batch_s=128) ## Make Predictions on test dataset

test_predic = torch.cat(test_predic) ## Combine predictions of all batches

test_predic = test_predic.argmax(dim=1)

train_predic = MakePredictions(cnn, x_train, batch_s=128) ## Make Predictions on train dataset

train_predic = torch.cat(train_predic)

train_predic = train_predic.argmax(dim=1)

print("Train Accuracy : {:.3f}".format(accuracy_score(y_train, train_predic)))
print("Test  Accuracy : {:.3f}".format(accuracy_score(y_test, test_predic)))

convert1 = nn.Conv2d(1,16, (3,3), padding="same")
convert2 = nn.Conv2d(16,32, (3,3), padding="same")

predic1 = convert1(torch.rand(50,1,28,28))
predic2 = convert2(predic1)

print("Layer 1 Weights Shape : {}".format(list(convert1.parameters())[0].shape))
print("Layer 2 Weights Shape : {}".format(list(convert2.parameters())[0].shape))

print("\nInput Shape         : {}".format((50,1,28,28)))
print("Layer 1 Output Shape  : {}".format(predic1.shape))
print("Layer 2 Output Shape  : {}".format(predic2.shape))