import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.optimize as opt
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, classification_report, precision_score, recall_score
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio
)
import torch
from LR_pt import compute_cost, LogisticRegression, train_lr

def sigmoid(x):
    """
    This is logistic regression
    f = 1/(1+exp(-beta^T * x))
    This function assumes as input that you have already multiplied beta and X together
    """
    return 1/(1+np.exp(-x))
 
 
# def logistic_loss(y_true, y_pred, eps = 1e-9):
#     """
#     Loss for the logistic regression, y_preds are probabilities
#     eps: epsilon for stability
#     """
#     # print parameters
#     return -np.mean(y_true * np.log(y_pred + eps) + (1-y_true) * np.log(1 - y_pred + eps))

def l2_loss(beta):
    """
    L2-Regularisation
    """
    return np.sum(beta[1:]**2)
 
# def fair_loss_gpt(y, y_pred, groups):
#     """
#     Group fairness Loss
#     GPT 4 PROPOSED THIS VERSION
#     """
#     y = np.array(y)
#     y_pred = np.array(y_pred)
#     groups = np.array(groups)

#     unique_groups = np.unique(groups)
#     assert len(unique_groups) == 2, "fair_loss function assumes exactly two groups"

#     # Create masks for the two groups
#     group1_mask = (groups == unique_groups[0])
#     group2_mask = (groups == unique_groups[1])

#     # Calculate the number of elements in each group
#     n1 = np.sum(group1_mask)
#     n2 = np.sum(group2_mask)

#     # Calculate the pairwise differences between y_pred for the two groups
#     y_pred_diff = y_pred[group1_mask].reshape(-1, 1) - y_pred[group2_mask].reshape(1, -1)

#     # Create a pairwise distance matrix for y
#     y_dist_matrix = (y[group1_mask].reshape(-1, 1) == y[group2_mask].reshape(1, -1)).astype(int)

#     # Compute the cost
#     cost = np.sum(y_dist_matrix * y_pred_diff)

#     return (cost / (n1 * n2)) ** 2
 
# def compute_gradient(beta, X, y, groups, _lambda,_gamma):
#     """Calculate the gradient - used for finding the best beta values. 
#        You do not need to use groups and lambda (fmin_tnc expects same input as in func, that's why they are included here)"""
#     grad = np.zeros(beta.shape)
#     ## grad is a vector that is [#num_features, 1]
#     # WE HAVE RECHECK THIS PART OF THE CODE
#     grad = np.mean( (sigmoid(np.dot(X,beta)) - y)[:, np.newaxis] * X ) + 2 * _gamma * beta
#     return grad
 
# def compute_cost(beta ,X, y, _lambda, _gamma, fair_loss_ = False, groups = None):
#     """Computes cost function with constraints"""
#     logits = np.dot(X, beta)
#     y_pred = sigmoid(logits)
    
#     # CHECK IF WE SHOULD USE THE LOGITS OR THE Y_PRED IN FAIR LOSS?
#     # AND SHOULD WE TUNE LAMBDA ALSO?
#     # AND SHOULD WE USE THE SAME LAMBDA FOR BOTH GROUPS? YES 
    
#     if fair_loss_:
#         return (
#                 logistic_loss(y, y_pred)
#                 + _gamma * l2_loss(beta)
#                 + sum(_lambda * fair_loss_gpt(y, logits, groups[:, i]) for i in range(groups.shape[1]))
#                 )
#     elif not fair_loss_:
#         return logistic_loss(y, y_pred) + _gamma * l2_loss(beta)
#     elif fair_loss_ == 'NO l2':
#         return logistic_loss(y, y_pred)
    
def standardize_tensor(tensor):
    mean = torch.mean(tensor, dim=0)
    std = torch.std(tensor, dim=0)
    standardized_tensor = (tensor - mean) / std
    return standardized_tensor


def validation_mask(X, arr, _fold_size, i):
    start = i * _fold_size
    end = (i + 1) * _fold_size
    if len(arr) - end < _fold_size:
        end = len(arr) - 1
    indices = arr[start:end]
        
    # create validation set
    mask = np.ones(len(X), dtype=bool)
    mask[indices] = False
    return mask
def validation_mask_torch(X, arr, _fold_size, i):
    start = i * _fold_size
    end = (i + 1) * _fold_size
    if len(arr) - end < _fold_size:
        end = len(arr) - 1
    indices = arr[start:end]
    
    # Create validation set mask
    mask = torch.zeros(len(X), dtype=torch.bool)
    mask[torch.tensor(indices)] = True
    return mask
def calculate_fair_accuracy(y_, y_pred, verbose=False):
    tp = np.sum((y_ == 1) & (y_pred == 1))
    tn = np.sum((y_ == 0) & (y_pred == 0))
    fp = np.sum((y_ == 0) & (y_pred == 1))
    fn = np.sum((y_ == 1) & (y_pred == 0))
 
    # Calculate TPR and TNR
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    
    if verbose:
        print("TPR:", tpr, "TNR:", tnr, "Accuracy:", (tpr + tnr)/2)
    return (tpr + tnr)/2 


def grid_search(gammas, X_train_cv, y_train_cv, train_groups, num_folds: int = 5, verbose = False, _lambda = 0):
    hyp_scores = []
    #create folds
    arr = np.arange(X_train_cv.shape[0])
    np.random.shuffle(arr)
    _fold_size = len(arr) // num_folds

    # not use protected features in training
    betas = np.random.rand(X_train_cv.shape[1])

    for _gamma in gammas:
        print("Gamma: ", _gamma) 
        fair_accuracy, accuracy, f1_scores, balanced_accuracy_scores = cross_val_random(y_train_cv, num_folds, verbose, arr, _fold_size, X_train_cv, train_groups)
 
        average_accuracy = np.mean(balanced_accuracy_scores)
        if verbose:
            print("Average balanced_accuracy_scores: ", average_accuracy)
        hyp_scores.append((average_accuracy, _gamma))
    return hyp_scores

def cross_val_random(y_train_cv, iter, verbose, arr, _fold_size, X_train_cv_dropped , train_groups):
    fair_accuracy = []
    accuracy = []
    f1_scores = []
    balanced_accuracy_scores = []
    for i in range(iter - 1):
        mask = validation_mask(X_train_cv_dropped, arr, _fold_size, i)
        # standardize data for each fold
        #X_train_scaled = standardize_tensor(X_train_cv_dropped[mask])
        scaler = StandardScaler()
        scaler.fit(X_train_cv_dropped[mask])
        X_train_scaled = scaler.transform(X_train_cv_dropped[mask])
        y_pred, model = train_lr(X_train_scaled, y_train_cv[mask], 'X_val', 'y_val', train_groups, 'val_groups', num_epochs=1000, fair_loss_=False, plot_loss=False, num_samples= 1000, val_check = False)
        # transform y_pred from tensor to numpy array
        y_pred = y_pred.detach().numpy()
        #fair_accuracy.append(calculate_fair_accuracy(y_train_cv[mask], y_pred))
        balanced_accuracy_scores.append(balanced_accuracy_score(y_train_cv[mask], y_pred)) # want to check if fair_accuracy is the same as balanced_accuracy
        f1_scores.append(f1_score(y_train_cv[mask], y_pred))
        accuracy.append(accuracy_score(y_train_cv[mask], y_pred))
        if verbose:
            print("Fold: ", i)
    return "fair_accuracy", accuracy, f1_scores, balanced_accuracy_scores


def get_tuned_gamma(gammas, X_train, y_train, train_groups, num_folds=5, verbose=False):
    hyp_scores = grid_search(gammas, X_train, y_train, train_groups, num_folds=num_folds, verbose=verbose)
    best_gamma = max(hyp_scores, key=lambda item:item[0])[1]
    print("Best gamma: ", best_gamma)
    return best_gamma

def evaluate(X_test, y_test, result):
    # Compute the predictions using the logistic regression weights
    predictions = sigmoid(np.dot(X_test, result[0]))
    binary_predictions = (predictions > 0.5).astype(int)
    # Calculate the accuracy of the logistic regression model
    #print(f'{X_test.shape, y_test.shape, binary_predictions.shape=}')
    accuracy = np.mean(binary_predictions == y_test)
    print(f"Logistic regression accuracy: {accuracy * 100:.2f}%")
    return binary_predictions

def train(X_train, y_train, X_test_, y_test_, groups, fair_loss_, best_gamma, lambda_val = 1):
    betas = np.random.rand(X_train.shape[1])
    result = opt.fmin_tnc(func=compute_cost, x0=betas, maxfun = 1000, args = (X_train, y_train, lambda_val, best_gamma, fair_loss_, groups), xtol=1e-4, ftol=1e-4, approx_grad=True, messages=0)
    preds = evaluate(X_test_, y_test_, result)
    print(classification_report(y_test_, preds))
    return preds


def tune_lambda(x_train, y_train, test_groups, groups, x_test, y_test, fair_loss_, best_gamma, one_hot_cols):
    def get_preds(result, X_test):
        predictions = sigmoid(np.dot(X_test, result[0]))
        return (predictions > 0.5).astype(int)

    performance_metrics = {'F1 Score': []}
    
    # Add F1 Score, Demographic Parity, and Equalized Odds metrics for each column
    for col in one_hot_cols:
        performance_metrics[f'F1 Score for {col}'] = []
        performance_metrics[f'{col} Demographic Parity Difference'] = []
        performance_metrics[f'{col} Demographic Parity Ratio'] = []
        performance_metrics[f'{col} Equalized Odds Difference'] = []
        performance_metrics[f'{col} Equalized Odds Ratio'] = []

    lambda_vals = [0.001, 0.005, 0.01, 0.05, 0.1, 1]

    device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    X_train_tensor = torch.from_numpy(x_test).float().to(device)
    y_train_tensor = torch.from_numpy(y_test).long().view(-1, 1).to(device)
    for lambda_val in lambda_vals:
        betas = np.random.rand(x_train.shape[1])

        #result = opt.fmin_tnc(func=compute_cost, x0=betas, maxfun = 1000, args = (x_train, y_train, lambda_val, best_gamma, fair_loss_, groups), xtol=1e-4, ftol=1e-4, approx_grad=True, messages=0)
        y_train_pred, model = train_lr(x_train, y_train, 'X_test', 'y_test', groups, 'test_groups', num_epochs=1000, fair_loss_=fair_loss_, plot_loss=False, num_samples= 1000, val_check = False, lambda_val = lambda_val)
        #test_preds = get_preds(result, x_test)

        # Compute predictions for test set with a pytorch model
        y_test_pred = model(x_test) > 0.5
        y_train_pred = y_train_pred.detach().numpy()
        test_preds = y_test_pred.detach().numpy()

        # Generate masks dynamically for each column in one_hot_cols
        masks = {col: test_groups[:, i] == 1 for i, col in enumerate(one_hot_cols)}

        print("Lambda: ", lambda_val)
        performance_metrics['F1 Score'].append(f1_score(y_test, test_preds))
        
        # Compute metrics for each column
        for col in one_hot_cols:
            mask = masks[col]
            # Check if test_preds[mask] is empty
            if test_preds[mask].size == 0:
                print(f'No {col} predictions for this lambda value {lambda_val}')
                performance_metrics[f'F1 Score for {col}'].append(0)
            else: 
                performance_metrics[f'F1 Score for {col}'].append(f1_score(y_test[mask], test_preds[mask]))
            performance_metrics[f'{col} Demographic Parity Difference'].append(demographic_parity_difference(y_test, test_preds, sensitive_features=test_groups[:, one_hot_cols.index(col)]))
            performance_metrics[f'{col} Demographic Parity Ratio'].append(demographic_parity_ratio(y_test, test_preds, sensitive_features=test_groups[:, one_hot_cols.index(col)]))
            performance_metrics[f'{col} Equalized Odds Difference'].append(equalized_odds_difference(y_test, test_preds, sensitive_features=test_groups[:, one_hot_cols.index(col)]))
            performance_metrics[f'{col} Equalized Odds Ratio'].append(equalized_odds_ratio(y_test, test_preds, sensitive_features=test_groups[:, one_hot_cols.index(col)]))

    return performance_metrics

def plot_lambda_tuning(performance_metrics, lambda_vals, one_hot_cols):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))  # changed to axs and added 2, 1 for two subplots
    plt.style.use('bmh')
    lambda_str = [str(l) for l in lambda_vals]

    col_name_mapping = {
        'Race_American_Indian_Alaska_Native': 'AI/AN',
        'Race_Asian': 'Asian',
        'Race_Black_African_American': 'Black',
        'Race_Native_Hawaiian_Pacific_Islander': 'NH/PI',
        'Race_White': 'White',
        'Race_White_Latino': 'White Latino',
    }

    for col in one_hot_cols:
        short_name = col_name_mapping[col]
        axs[0].plot(lambda_str, performance_metrics[f'F1 Score for {col}'], label=short_name)  # changed to axs[0]
        axs[0].scatter(lambda_str, performance_metrics[f'F1 Score for {col}'])
        axs[1].plot(lambda_str, performance_metrics[f'{col} Equalized Odds Difference'], label=short_name)  # new subplot for Equalized Odds Difference
        axs[1].scatter(lambda_str, performance_metrics[f'{col} Equalized Odds Difference'])

    axs[0].legend(loc='upper right')
    axs[0].set_title('F1-score for different lambda values')
    axs[0].set_xlabel('Lambda')
    axs[0].set_ylabel('F1-score')

    axs[1].legend(loc='upper right')  # added legend, title, x and y labels for the new subplot
    axs[1].set_title('Equalized Odds Difference for different lambda values')
    axs[1].set_xlabel('Lambda')
    axs[1].set_ylabel('Equalized Odds Difference')

    plt.tight_layout()
    plt.show()