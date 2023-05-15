import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def train_and_evaluate_nn(x_train, x_val, x_test, y_train, y_val, y_test, num_epochs=20, batch_size=32, lr=0.001, plot_loss=False, seed=4206942):
    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Convert numpy arrays to PyTorch tensors
    x_train_tensor = torch.from_numpy(x_train).float()
    x_val_tensor = torch.from_numpy(x_val).float()
    x_test_tensor = torch.from_numpy(x_test).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    y_val_tensor = torch.from_numpy(y_val).long()
    y_test_tensor = torch.from_numpy(y_test).long()

    # Create TensorDatasets for training and validation
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # Define the neural network model
    class SimpleNN(nn.Module):
        def __init__(self, input_size, num_classes):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, num_classes)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.relu(x)
            x = self.fc4(x)
            x = self.softmax(x)
            return x

        # Initialize the model, loss function and optimizer
    input_size = x_train.shape[1]
    num_classes = len(np.unique(y_train))
    model = SimpleNN(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_data, val_target in val_loader:
                val_output = model(val_data)
                v_loss = criterion(val_output, val_target)
                val_loss += v_loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss}, Val Loss: {val_loss}')

    if plot_loss:
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()
        
    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy}%')

    

    return model, accuracy
