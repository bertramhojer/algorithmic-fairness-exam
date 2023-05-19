import time
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from PCA import fair_PCA


# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.prepare_input(x)
        x = self.fc_layers(x)
        return x

    def prepare_input(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif isinstance(x, list):
            x = torch.tensor(x).float()
        elif not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a numpy array, list or PyTorch tensor")
        return x
    
def train_and_evaluate_nn(x_train, x_val, x_test, y_train, y_val, y_test, train_groups, pca, num_epochs=20, batch_size=128, lr=0.001, plot_loss=False, seed=4206942, f1_freq_=5):
    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    if pca:
        print(f"'Number of components:' , {x_train.shape[1]}")
        X_fair_PCA, U, explained_variance = fair_PCA(x_train, n_components=x_train.shape[1], groups=train_groups)
        x_train = X_fair_PCA
        x_val = x_val @ U
        x_test = x_test @ U

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

    # Initialize the model, loss function and optimizer
    input_size = x_train.shape[1]
    num_classes = len(np.unique(y_train))
    model = SimpleNN(input_size, num_classes)
    # compute class weights
    class_weights = torch.tensor(torch.tensor([(1 / (y_train == 0).sum()), 1 / (y_train == 1).sum()])).float()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    train_losses = []
    train_f1_scores = []
    val_losses = []
    val_f1_scores = []
    # You can control the frequency of F1 score calculation using this parameter
    f1_freq = f1_freq_ # change this to set your desired frequency

    # Inside your training loop:
    for epoch in tqdm(range(num_epochs)):
        start_time = time.perf_counter()
        model.train()

        train_loss = 0
        train_preds = []
        train_targets = []
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Compute F1 score for training set every "f1_calculation_frequency" epochs
            if epoch % f1_freq == 0:
                _, pred_classes = torch.max(output, 1)
                train_preds.extend(pred_classes.cpu().numpy())
                train_targets.extend(target.cpu().numpy())

        if epoch % f1_freq == 0:
            train_f1 = f1_score(train_targets, train_preds, average='weighted')
            train_f1_scores.append(train_f1)

        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for val_data, val_target in val_loader:
                val_output = model(val_data)
                v_loss = criterion(val_output, val_target)
                val_loss += v_loss.item()

                # Compute F1 score for validation set every "f1_calculation_frequency" epochs
                if epoch % f1_freq == 0:
                    _, pred_classes = torch.max(val_output, 1)
                    val_preds.extend(pred_classes.cpu().numpy())
                    val_targets.extend(val_target.cpu().numpy())

        if epoch % f1_freq == 0:
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

        if epoch % f1_freq == 0:
            val_f1 = f1_score(val_targets, val_preds, average='weighted')
            val_f1_scores.append(val_f1)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')
            end_time = time.perf_counter()
            print(f'Epoch {epoch} took {end_time - start_time:.2f} seconds')

    if plot_loss:
        fig, axs = plt.subplots(2)
        axs[0].plot(train_losses, label='Train Loss')
        axs[0].plot(val_losses, label='Val Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        axs[1].plot(train_f1_scores, label='Train F1 Score')
        axs[1].plot(val_f1_scores, label='Val F1 Score')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('F1 Score')
        axs[1].legend()
        plt.savefig(f'../plots/NN_pca:{pca}_E:{num_epochs}_lr:{lr}_bs:{batch_size}.png')
        
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

    # Save the model
    model_path = f'../models/NN_pca:{pca}_E:{num_epochs}_lr:{lr}_bs:{batch_size}.pt'
    torch.save(model.state_dict(), model_path)

    return model, input_size, model_path


from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model_path, x_train, x_test, y_test, input_size, num_classes, pca=False, train_groups=None):
    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    if pca:
        print(f"'Number of components:' , {x_train.shape[1]}")
        print(f"X_train shape: {x_train.shape}")
        X_fair_PCA, U, explained_variance = fair_PCA(x_train, n_components=x_train.shape[1], groups=train_groups)
        x_train = X_fair_PCA
        x_test = x_test @ U
        print(f"X_train_pca shape: {x_train.shape}")

    # Initialize the model
    model = SimpleNN(x_train.shape[1], num_classes).to(device)
    
    # Load the model
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    
    # Convert x_test and y_test to tensors
    x_test_tensor = torch.from_numpy(x_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).long().to(device)

    with torch.no_grad():
        # Get the model predictions
        outputs = model(x_test_tensor)
        
        # Convert the predictions to probabilities using softmax
        probabilities, predicted = torch.max(outputs.data, 1)

        # Compute accuracy
        accuracy = accuracy_score(y_test_tensor.cpu().numpy(), predicted.cpu().numpy())
        
        # Compute F1 score
        f1 = f1_score(y_test_tensor.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
        print(classification_report(y_test_tensor.cpu().numpy(), predicted.cpu().numpy()))

    print(f'Accuracy: {accuracy}')
    print(f'F1 Score: {f1}')

    return accuracy, f1