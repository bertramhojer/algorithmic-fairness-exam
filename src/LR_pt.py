from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import time
from sklearn.metrics import classification_report
from tqdm import tqdm

class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.prepare_input(x)
        output = torch.sigmoid(self.linear(x))
        return output
    
    def prepare_input(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif isinstance(x, list):
            x = torch.tensor(x).float()
        elif not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a numpy array, list or PyTorch tensor")
        return x

def logistic_loss(y_true, y_pred, eps=1e-9):
    return -(y_true * torch.log(y_pred + eps) + (1 - y_true) * torch.log(1 - y_pred + eps)).mean()

def l2_loss(beta, gamma):
    return gamma * torch.sum(beta[1:] ** 2)

def fair_loss(y, y_pred, groups):
    y = y.squeeze()
    y_pred = y_pred.squeeze()
    groups = groups.squeeze()

    unique_groups = torch.unique(groups)
    assert len(unique_groups) == 2, "fair_loss function assumes exactly two groups"

    group1_mask = (groups == unique_groups[0])
    group2_mask = (groups == unique_groups[1])

    n1 = torch.sum(group1_mask)
    n2 = torch.sum(group2_mask)

    y_pred_diff = y_pred[group1_mask].view(-1, 1) - y_pred[group2_mask].view(1, -1)
    y_dist_matrix = (y[group1_mask].view(-1, 1) == y[group2_mask].view(1, -1)).float()

    cost = torch.sum(y_dist_matrix * y_pred_diff)

    return (cost / (n1 * n2)) ** 2

def fair_loss_sample(y, y_pred, groups, sample_size=5_000):
    y = y.squeeze()
    y_pred = y_pred.squeeze()
    groups = groups.squeeze()

    unique_groups = torch.unique(groups)
    assert len(unique_groups) == 2, "fair_loss function assumes exactly two groups"

    group1_mask = (groups == unique_groups[0])
    group2_mask = (groups == unique_groups[1])

    n1 = torch.sum(group1_mask)
    n2 = torch.sum(group2_mask)

    # Select a subset of samples from each group
    group1_samples = torch.randperm(n1)[:sample_size]
    group2_samples = torch.randperm(n2)[:sample_size]

    y_pred_diff = y_pred[group1_mask][group1_samples].view(-1, 1) - y_pred[group2_mask][group2_samples].view(1, -1)
    y_dist_matrix = (y[group1_mask][group1_samples].view(-1, 1) == y[group2_mask][group2_samples].view(1, -1)).float()

    cost = torch.sum(y_dist_matrix * y_pred_diff)

    return (cost / (sample_size * sample_size)) ** 2

def compute_cost(model, X, y, groups, _lambda, _gamma, fair_loss_=False):
    logits = model(X)
    y_pred = torch.sigmoid(logits)
    beta = list(model.parameters())[0]

    if fair_loss_ == True:
        logistic_loss_value = logistic_loss(y, y_pred)
        l2_loss_value = l2_loss(beta, _gamma)
        fair_loss_value = sum(_lambda * fair_loss_sample(y, logits, groups[:, i]) for i in range(groups.shape[1]))
        total_loss = logistic_loss_value + l2_loss_value + fair_loss_value

        # log the values
        # print("logistic_loss:", logistic_loss_value.item(), "(", (logistic_loss_value / total_loss).item() * 100, "%)")
        # print("l2_loss:", l2_loss_value.item(), "(", (l2_loss_value / total_loss).item() * 100, "%)")
        # print("fair_loss:", fair_loss_value.item(), "(", (fair_loss_value / total_loss).item() * 100, "%)")

        return total_loss
    
    elif not fair_loss_:
        return logistic_loss(y, y_pred) + l2_loss(beta, _gamma)
    elif fair_loss_ == 'NO l2':
        return logistic_loss(y, y_pred)

def train_lr(X_train, y_train, X_val, y_val, groups, val_groups, num_epochs=100, fair_loss_=False, plot_loss=True, num_samples= 1_000_000, val_check = True, _lambda=1, _gamma=0.1, learning_rate=0.01):
    # Check if MPS is available
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).long().view(-1, 1).to(device)
    groups_tensor = torch.from_numpy(groups).long().to(device)

    # Check if validation data is provided
    if val_check:
        val_groups_tensor = torch.from_numpy(val_groups).long().to(device)
        X_val_tensor = torch.from_numpy(X_val).float().to(device)
        y_val_tensor = torch.from_numpy(y_val).long().view(-1, 1).to(device)

    # Create the logistic regression model
    model = LogisticRegression(input_size=X_train.shape[1]).to(device)

    # Create the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Create lists to store the losses and F1 scores
    train_losses = []
    val_losses = []
    train_f1_scores = []
    val_f1_scores = []

    # early stopping
    patience = 3  # or any other number of your choosing
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    # Train the model
    for epoch in tqdm(range(num_epochs)):
        start_time = time.perf_counter()
        model.train()

        # Calculate the cost
        train_cost = compute_cost(model, X_train_tensor, y_train_tensor, groups_tensor, _lambda, _gamma, fair_loss_=fair_loss_)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        train_cost.backward()
        optimizer.step()

        if val_check:
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_cost = compute_cost(model, X_val_tensor, y_val_tensor, val_groups_tensor, _lambda, _gamma, fair_loss_=fair_loss_)

            val_losses.append(val_cost.item())
            # Calculate F1 score for validation data
            val_pred = model(X_val_tensor) > 0.5
            val_f1 = f1_score(y_val_tensor.cpu().numpy(), val_pred.cpu().numpy())
            val_f1_scores.append(val_f1)

        train_losses.append(train_cost.item())
        # Calculate F1 score for training data
        train_pred = model(X_train_tensor) > 0.5
        train_f1 = f1_score(y_train_tensor.cpu().numpy(), train_pred.cpu().numpy())
        train_f1_scores.append(train_f1)


        if (epoch + 1) % 10 == 0:
            if val_check:
                print(f'Epoch {epoch + 1}/{num_epochs}, Train Cost: {train_cost.item()}, Val Cost: {val_cost.item()}')
                end_time = time.perf_counter()
                print(f'Epoch {epoch} took {end_time - start_time:.2f} seconds')
            else:  
                print(f'Epoch {epoch + 1}/{num_epochs}, Train Cost: {train_cost.item()}')
        
       

        if val_check:
                if val_cost.item() < best_val_loss:
                    best_val_loss = val_cost.item()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement == patience:
                        print("Stopping early!")
                        break
        
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
        plt.savefig(f'plots/LRmodel_S:{num_samples}_E:{num_epochs}_F:{fair_loss_}_L:{_lambda}_G:{_gamma}.png')
    
    model.eval()
    if val_check:
        with torch.no_grad():
            y_val_pred = model(X_val_tensor) > 0.5
            print(classification_report(y_val_tensor.numpy(), y_val_pred.numpy()))
    else:
        with torch.no_grad():
            y_train_pred = model(X_train_tensor) > 0.5
        return y_train_pred, model

    torch.save(model.state_dict(), f'models/LRmodel_S:{num_samples}_E:{num_epochs}_F:{fair_loss_}_L:{_lambda}_G:{_gamma}.pt')
    
    return model, fig, axs