import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.optimize as opt
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, classification_report
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio
)
import torch
from LR_pt import compute_cost, train_lr
from tqdm import tqdm
import time
from sklearn.model_selection import KFold
from collections import defaultdict

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print("Execution time: {:.4f} seconds".format(execution_time))
        return result
    return wrapper

def sigmoid(x):
    """
    This is logistic regression
    f = 1/(1+exp(-beta^T * x))
    This function assumes as input that you have already multiplied beta and X together
    """
    return 1/(1+np.exp(-x))

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

def grid_search(gammas, X_train_cv, y_train_cv, train_groups, fair_loss_, num_folds: int = 5, verbose = False, _lambda = 0):
    hyp_scores = []
    #create folds
    arr = np.arange(X_train_cv.shape[0])
    np.random.shuffle(arr)
    _fold_size = len(arr) // num_folds

    for _gamma in tqdm(gammas):
        print("Gamma: ", _gamma) 
        fair_accuracy, accuracy, f1_scores, balanced_accuracy_scores = cross_val_random(y_train_cv, num_folds, verbose, arr, _fold_size, X_train_cv, train_groups, fair_loss_)
 
        average_accuracy = np.mean(balanced_accuracy_scores)
        if verbose:
            print("Average balanced_accuracy_scores: ", average_accuracy)
        hyp_scores.append((average_accuracy, _gamma))
    return hyp_scores

def cross_val_random(y_train_cv, iter, verbose, arr, _fold_size, X_train_cv_dropped , train_groups, fair_loss_, num_samples = 1000):
    accuracy = []
    f1_scores = []
    balanced_accuracy_scores = []
    for i in range(iter - 1):
        mask = validation_mask(X_train_cv_dropped, arr, _fold_size, i)
        # standardize data for each fold
        scaler = StandardScaler()
        scaler.fit(X_train_cv_dropped[mask])
        X_train_scaled = scaler.transform(X_train_cv_dropped[mask])
        
        y_pred, model = train_lr(X_train_scaled, y_train_cv[mask], 'X_val', 'y_val', train_groups, 'val_groups', num_epochs=2000, 
                                 fair_loss_= fair_loss_, plot_loss=True, num_samples= num_samples, val_check = False)
        y_pred = y_pred.detach().numpy()

        balanced_accuracy_scores.append(balanced_accuracy_score(y_train_cv[mask], y_pred)) 
        f1_scores.append(f1_score(y_train_cv[mask], y_pred))
        accuracy.append(accuracy_score(y_train_cv[mask], y_pred))
        if verbose:
            print("Fold: ", i)
    return "fair_accuracy", accuracy, f1_scores, balanced_accuracy_scores


def get_tuned_gamma(gammas, X_train, y_train, train_groups, num_folds=3, verbose=False, fair_loss_=False):
    hyp_scores = grid_search(gammas, X_train, y_train, train_groups, fair_loss_ = fair_loss_, num_folds=num_folds, verbose=verbose)
    best_gamma = max(hyp_scores, key=lambda item:item[0])[1]
    print("Best gamma: ", best_gamma)
    return best_gamma

def evaluate(X_test, y_test, result):
    # Compute the predictions using the logistic regression weights
    predictions = sigmoid(np.dot(X_test, result[0]))
    binary_predictions = (predictions > 0.5).astype(int)
    
    # Calculate the accuracy of the logistic regression model
    accuracy = np.mean(binary_predictions == y_test)
    print(f"Logistic regression accuracy: {accuracy * 100:.2f}%")
    return binary_predictions

def train(X_train, y_train, X_test_, y_test_, groups, fair_loss_, best_gamma, lambda_val = 1):
    betas = np.random.rand(X_train.shape[1])
    result = opt.fmin_tnc(func=compute_cost, x0=betas, maxfun = 1000, args = (X_train, y_train, lambda_val, best_gamma, fair_loss_, groups), xtol=1e-4, ftol=1e-4, approx_grad=True, messages=0)
    preds = evaluate(X_test_, y_test_, result)
    print(classification_report(y_test_, preds))
    return preds


def tune_lambda_old(x_train, y_train, test_groups, groups, x_test, y_test, best_gamma, one_hot_cols, lambda_vals):
    performance_metrics = {'F1 Score': []}
    
    # Add F1 Score, Demographic Parity, and Equalized Odds metrics for each column
    for col in one_hot_cols:
        performance_metrics[f'F1 Score for {col}'] = []
        performance_metrics[f'{col} Demographic Parity Difference'] = []
        performance_metrics[f'{col} Demographic Parity Ratio'] = []
        performance_metrics[f'{col} Equalized Odds Difference'] = []
        performance_metrics[f'{col} Equalized Odds Ratio'] = []

    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    #convert x_test and y_test to tensors
    x_test_tensor = torch.from_numpy(x_test).float().to(device)
    for lambda_val in tqdm(lambda_vals):
        # Train model with a pytorch model
        y_train_pred, model = train_lr(x_train, y_train, 'X_test', 'y_test', groups, 'test_groups', num_epochs=500, fair_loss_=True, plot_loss=True, 
                                       num_samples= 2000, val_check = False, _lambda=lambda_val, _gamma=best_gamma)

        # Compute predictions for test set with a pytorch model
        y_test_pred = model(x_test_tensor) > 0.5
        y_train_pred = y_train_pred.detach().numpy()
        test_preds = y_test_pred.detach().numpy()

        print(f'{y_train_pred.shape=}, {test_preds.shape=}')

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
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
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
    # save the plot
    plt.savefig('../plots/lambda_tuning.png')



def tune_lambda_cv(x_train, y_train, test_groups, groups, x_test, y_test, best_gamma, one_hot_cols, lambda_vals, num_folds=3):
    # Initialize KFold cross-validator
    kf = KFold(n_splits=num_folds)
    
    performance_metrics = {'F1 Score': []}
    for col in one_hot_cols:
        performance_metrics[f'F1 Score for {col}'] = []
        performance_metrics[f'{col} Demographic Parity Difference'] = []
        performance_metrics[f'{col} Demographic Parity Ratio'] = []
        performance_metrics[f'{col} Equalized Odds Difference'] = []
        performance_metrics[f'{col} Equalized Odds Ratio'] = []

    for lambda_val in tqdm(lambda_vals):
        f1_scores = []
        for train_index, val_index in kf.split(x_train):
            x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            groups_fold = groups[train_index, :]
            val_groups = groups[val_index, :] 
            print(f'{x_train_fold.shape=}, {x_val_fold.shape=}, {y_train_fold.shape=}, {y_val_fold.shape=}, {val_groups.shape=}, {groups.shape=}')
            y_train_pred, model = train_lr(x_train_fold, y_train_fold, 'X_val', 'y_val', groups_fold, 'val_groups', num_epochs=100, fair_loss_=True, plot_loss=True, 
                                           num_samples=1000, val_check = False, _lambda=lambda_val, _gamma=best_gamma)

            y_val_pred = model(x_val_fold) > 0.5
            y_train_pred = y_train_pred.detach().numpy()
            val_preds = y_val_pred.detach().numpy()

            f1_scores.append(f1_score(y_val_fold, val_preds))

            # Initialize a dictionary to store the sum of the metrics for each fold
            metrics_sums = defaultdict(lambda: defaultdict(float))

            for idx, col in enumerate(one_hot_cols):
                mask = val_groups[:, idx] == 1

                if val_preds[mask].size == 0:
                    print(f'No {col} predictions for this lambda value {lambda_val}')
                    metrics_sums[col]['F1 Score'] += 0
                else:
                    metrics_sums[col]['F1 Score'] += f1_score(y_val_fold[mask], val_preds[mask])

                metrics_sums[col]['Demographic Parity Difference'] += demographic_parity_difference(y_val_fold, val_preds, sensitive_features=val_groups[:, one_hot_cols.index(col)])
                metrics_sums[col]['Demographic Parity Ratio'] += demographic_parity_ratio(y_val_fold, val_preds, sensitive_features=val_groups[:, one_hot_cols.index(col)])
                metrics_sums[col]['Equalized Odds Difference'] += equalized_odds_difference(y_val_fold, val_preds, sensitive_features=val_groups[:, one_hot_cols.index(col)])
                metrics_sums[col]['Equalized Odds Ratio'] += equalized_odds_ratio(y_val_fold, val_preds, sensitive_features=val_groups[:, one_hot_cols.index(col)])
        
        # Calculate average F1 score and add to performance_metrics
        performance_metrics['F1 Score'].append(np.mean(f1_scores))

        # Calculate average of other metrics and add to performance_metrics
        for col in one_hot_cols:
            performance_metrics[f'F1 Score for {col}'].append(metrics_sums[col]['F1 Score'] / num_folds)
            performance_metrics[f'{col} Demographic Parity Difference'].append(metrics_sums[col]['Demographic Parity Difference'] / num_folds)
            performance_metrics[f'{col} Demographic Parity Ratio'].append(metrics_sums[col]['Demographic Parity Ratio'] / num_folds)
            performance_metrics[f'{col} Equalized Odds Difference'].append(metrics_sums[col]['Equalized Odds Difference'] / num_folds)
            performance_metrics[f'{col} Equalized Odds Ratio'].append(metrics_sums[col]['Equalized Odds Ratio'] / num_folds)

    return performance_metrics
