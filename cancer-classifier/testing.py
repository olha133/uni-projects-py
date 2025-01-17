import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch

def prepare_data():
    df = pd.read_csv("Cancer_Data.csv")
    df['diagnosis'].replace(['B', 'M'],
                        [0, 1], inplace=True)

    plt.figure(figsize=(10,6))
    sns.set_style("whitegrid")
    plt.pie(df['diagnosis'].value_counts(),autopct='%1.2f%%', startangle=90)
    plt.axis('equal')
    plt.title("B x M")
    #plt.show()

    M = df[df.diagnosis == 1] #Diagnosis transfers all values of M to M data
    B = df[df.diagnosis == 0] #Diagnosis transfers all values of B to B data

    plt.scatter(M.radius_mean,M.texture_mean, label = "Malignant", alpha = 0.3)
    plt.scatter(B.radius_mean,B.texture_mean,label = "Benign", alpha = 0.3)

    plt.xlabel("radius_mean")
    plt.ylabel("texture_mean")

    plt.legend()
    #plt.show()

    df = df.drop(labels="Unnamed: 32", axis=1)
    df = df.drop(labels="id", axis=1)

    X, y = df.drop('diagnosis', axis=1), df[['diagnosis']]
    scaler = MinMaxScaler()
    le = LabelEncoder()

    X_scaled = scaler.fit_transform(X)
    y = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=43)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    return X_train, X_val, y_train, y_val

def create_model(input_dim, output_dim, learning_rate):
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, output_dim)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    return model, optimizer, loss_fn

def train_model(X_train, Y_train, X_val, Y_val, model, optimizer, loss_fn, epochs):
    accuracy = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, Y_train[:, None])
        
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
          normalized_preds = torch.round(torch.sigmoid(y_pred)).squeeze()
          train_acc = (Y_train == normalized_preds).sum().item() / len(Y_train)

          test_preds = model(X_val)
          normalized_preds = torch.round(torch.sigmoid(test_preds)).squeeze()
          test_acc = (Y_val == normalized_preds).sum().item() / len(Y_val)

          print(f"Loss: {loss.item()}, Epoch: {epoch}, Train Acc: {train_acc}, Test Acc: {test_acc}")
          if (test_acc > accuracy):
            accuracy = test_acc
    return accuracy

def main():
    # Read and prepare data
    X_train, X_val, Y_train, Y_val = prepare_data()
    # Create model
    model, optimizer, loss_fn = create_model(X_train.shape[-1], 1, 0.01)
    # Train model
    accuracy = train_model(X_train, Y_train, X_val, Y_val, model, optimizer, loss_fn, 1000)
    print(accuracy)

    
if __name__ == "__main__":
    main()