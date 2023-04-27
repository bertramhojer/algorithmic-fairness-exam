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


def sigmoid(x):
    """
    This is logistic regression
    f = 1/(1+exp(-beta^T * x))
    This function assumes as input that you have already multiplied beta and X together
    """
    return 1/(1+np.exp(-x))
 
 
def logistic_loss(y_true, y_pred, eps = 1e-9):
    """
    Loss for the logistic regression, y_preds are probabilities
    eps: epsilon for stability
    """
    # print parameters
    return -np.mean(y_true * np.log(y_pred + eps) + (1-y_true) * np.log(1 - y_pred + eps))

def l2_loss(beta):
    """
    L2-Regularisation
    """
    return np.sum(beta[1:]**2)
 
def fair_loss_gpt(y, y_pred, groups):
    """
    Group fairness Loss
    GPT 4 PROPOSED THIS VERSION
    """
    y = np.array(y)
    y_pred = np.array(y_pred)
    groups = np.array(groups)

    unique_groups = np.unique(groups)
    assert len(unique_groups) == 2, "fair_loss function assumes exactly two groups"

    # Create masks for the two groups
    group1_mask = (groups == unique_groups[0])
    group2_mask = (groups == unique_groups[1])

    # Calculate the number of elements in each group
    n1 = np.sum(group1_mask)
    n2 = np.sum(group2_mask)

    # Calculate the pairwise differences between y_pred for the two groups
    y_pred_diff = y_pred[group1_mask].reshape(-1, 1) - y_pred[group2_mask].reshape(1, -1)

    # Create a pairwise distance matrix for y
    y_dist_matrix = (y[group1_mask].reshape(-1, 1) == y[group2_mask].reshape(1, -1)).astype(int)

    # Compute the cost
    cost = np.sum(y_dist_matrix * y_pred_diff)

    return (cost / (n1 * n2)) ** 2
 
def compute_gradient(beta, X, y, groups, _lambda,_gamma):
    """Calculate the gradient - used for finding the best beta values. 
       You do not need to use groups and lambda (fmin_tnc expects same input as in func, that's why they are included here)"""
    grad = np.zeros(beta.shape)
    ## grad is a vector that is [#num_features, 1]
    # WE HAVE RECHECK THIS PART OF THE CODE
    grad = np.mean( (sigmoid(np.dot(X,beta)) - y)[:, np.newaxis] * X ) + 2 * _gamma * beta
    return grad
 
def compute_cost(beta ,X, y, _lambda, _gamma, fair_loss_ = False, groups = None):
    """Computes cost function with constraints"""
    logits = np.dot(X, beta)
    y_pred = sigmoid(logits)
    
    # CHECK IF WE SHOULD USE THE LOGITS OR THE Y_PRED IN FAIR LOSS?
    # AND SHOULD WE TUNE LAMBDA ALSO?
    # AND SHOULD WE USE THE SAME LAMBDA FOR BOTH GROUPS? YES 
    if fair_loss_:
        return logistic_loss(y, y_pred) + _gamma * l2_loss(beta) + _lambda * fair_loss_gpt(y, logits, groups[:,1]) + _lambda * fair_loss_gpt(y, logits, groups[:,0])
    elif not fair_loss_:
        return logistic_loss(y, y_pred) + _gamma * l2_loss(beta)
    elif fair_loss_ == 'NO l2':
        return logistic_loss(y, y_pred)
    
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

def grid_search(gammas, X_train_cv, y_train_cv, num_folds: int = 5, verbose = False, _lambda = 0):
    hyp_scores = []
    
    #create folds
    arr = np.arange(X_train_cv.shape[0])
    np.random.shuffle(arr)
    _fold_size = len(arr) // num_folds

    # not use protected features in training
    betas = np.random.rand(X_train_cv.shape[1])
    for _gamma in gammas:
        print("\nGamma: ", _gamma) 
        fair_accuracy, accuracy, f1_scores, balanced_accuracy_scores = cross_val_random(y_train_cv, num_folds, verbose, _lambda, arr, _fold_size, X_train_cv, betas, _gamma)
 
        average_accuracy = np.mean(fair_accuracy)
        if verbose:
            print("Average accuracy: ", average_accuracy)
        hyp_scores.append((average_accuracy, _gamma))
    return hyp_scores

def cross_val_random(y_train_cv, iter, verbose, _lambda, arr, _fold_size, X_train_cv_dropped, betas, _gamma):
    print('Cross validation')
    fair_accuracy = []
    accuracy = []
    f1_scores = []
    balanced_accuracy_scores = []
    for i in range(iter - 1):
        mask = validation_mask(X_train_cv_dropped, arr, _fold_size, i)
        # standardize data for each fold
            
        scaler = StandardScaler()
        scaler.fit(X_train_cv_dropped[mask])
        X_train_scaled = scaler.transform(X_train_cv_dropped)
            
        result = opt.fmin_tnc(func=compute_cost, x0=betas, maxfun = 1000, args = (X_train_scaled[mask], y_train_cv[mask], _lambda, _gamma), xtol=1e-4, ftol=1e-4, approx_grad=True, disp=0)
            
        # preds on left out fold 
        pred = sigmoid(np.dot(X_train_scaled[mask], result[0][:,np.newaxis]))
        y_pred = ((pred > 0.5)+0).ravel()
 
        fair_accuracy.append(calculate_fair_accuracy(y_train_cv[mask], y_pred))
        balanced_accuracy_scores.append(balanced_accuracy_score(y_train_cv[mask], y_pred)) # want to check if fair_accuracy is the same as balanced_accuracy
        f1_scores.append(f1_score(y_train_cv[mask], y_pred))
        accuracy.append(accuracy_score(y_train_cv[mask], y_pred))

        if verbose:
            print("Fold: ", i)
    return fair_accuracy, accuracy, f1_scores, balanced_accuracy_scores


def get_tuned_gamma(gammas, X_train, y_train, num_folds=5, verbose=False):
    hyp_scores = grid_search(gammas, X_train, y_train, num_folds=num_folds, verbose=verbose)
    # get best gamma
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


def tune_lambda(x_train, y_train, test_groups, groups, x_test, y_test, fair_loss_, best_gamma):
    def get_preds(result, X_test):
        predictions = sigmoid(np.dot(X_test, result[0]))
        return (predictions > 0.5).astype(int)

    #best_gamma = 0.1 #best computed gamma without fairness regulization #get_tuned_gamma(gammas, X_train_scaled, y_train, groups, num_folds=5, verbose=True)
    lambda_vals = [0.001, 0.005, 0.01, 0.05, 0.1, 1]
    fair_loss_ = True

    performance_metrics = {
    'F1 Score': [],
    'F1 Score for Men': [],
    'F1 Score for Women': [],
    'F1 Score for White people': [],
    'F1 Score for African-american people': [],
    'Gender Demographic Parity Difference': [],
    'Gender Demographic Parity Ratio': [],
    'Gender Equalized Odds Difference': [],
    'Gender Equalized Odds Ratio': [],
    'Race Demographic Parity Difference': [],
    'Race Demographic Parity Ratio': [],
    'Race Equalized Odds Difference': [],
    'Race Equalized Odds Ratio': []
}

    for lambda_val in lambda_vals:
        betas = np.random.rand(x_train.shape[1])
        result = opt.fmin_tnc(func=compute_cost, x0=betas, maxfun = 1000, args = (x_train, y_train, lambda_val, best_gamma, fair_loss_, groups), xtol=1e-4, ftol=1e-4, approx_grad=True, messages=0)

        test_preds = get_preds(result, x_test)
        mask_men = test_groups[:, 0] == 1
        mask_women = test_groups[:, 0] == 0
        test_preds_men = test_preds[mask_men]
        test_preds_women = test_preds[mask_women]
    
    # Also split for race
        mask_whites = test_groups[:, 1] == 1
        mask_african = test_groups[:, 1] == 0
        test_preds_white = test_preds[mask_whites]
        test_preds_african_american = test_preds[mask_african]

        print("Lambda: ", lambda_val)
    
        performance_metrics['F1 Score'].append(f1_score(y_test, test_preds))
        performance_metrics['F1 Score for Men'].append(f1_score(y_test[mask_men], test_preds_men))
        performance_metrics['F1 Score for Women'].append(f1_score(y_test[mask_women], test_preds_women))
        performance_metrics['F1 Score for White people'].append(f1_score(y_test[mask_whites], test_preds_white))
        performance_metrics['F1 Score for African-american people'].append(f1_score(y_test[mask_african], test_preds_african_american))

    # Compute fairness metrics for each protected attribute
        performance_metrics['Gender Demographic Parity Difference'].append(demographic_parity_difference(y_test, test_preds, sensitive_features=test_groups[:, 0]))
        performance_metrics['Gender Demographic Parity Ratio'].append(demographic_parity_ratio(y_test, test_preds, sensitive_features=test_groups[:, 0]))
        performance_metrics['Gender Equalized Odds Difference'].append(equalized_odds_difference(y_test, test_preds, sensitive_features=test_groups[:, 0]))
        performance_metrics['Gender Equalized Odds Ratio'].append(equalized_odds_ratio(y_test, test_preds, sensitive_features=test_groups[:, 0]))

        performance_metrics['Race Demographic Parity Difference'].append(demographic_parity_difference(y_test, test_preds, sensitive_features=test_groups[:, 1]))
        performance_metrics['Race Demographic Parity Ratio'].append(demographic_parity_ratio(y_test, test_preds, sensitive_features=test_groups[:, 1]))
        performance_metrics['Race Equalized Odds Difference'].append(equalized_odds_difference(y_test, test_preds, sensitive_features=test_groups[:, 1]))
        performance_metrics['Race Equalized Odds Ratio'].append(equalized_odds_ratio(y_test, test_preds, sensitive_features=test_groups[:, 1]))

    return performance_metrics

def plot_lambda_tuning(performance_metrics, lambda_vals):
    fig, axes = plt.subplots(1, 1, figsize=(10, 4))
    plt.style.use('bmh')
    lambda_str = [str(l) for l in lambda_vals]
    axes.plot(lambda_str, performance_metrics['F1 Score for Men'], label='Men', color='red')
    axes.scatter(lambda_str, performance_metrics['F1 Score for Men'], color='red')
    axes.plot(lambda_str, performance_metrics['F1 Score for Women'], label='Women', color='blue')
    axes.scatter(lambda_str, performance_metrics['F1 Score for Women'], color='blue')
    axes.plot(lambda_str, performance_metrics['F1 Score for White people'], label='Caucasian', color='green')
    axes.scatter(lambda_str, performance_metrics['F1 Score for White people'], color='green')
    axes.plot(lambda_str, performance_metrics['F1 Score for African-american people'], label='African-american', color='orange')
    axes.scatter(lambda_str, performance_metrics['F1 Score for African-american people'], color='orange')
    axes.legend(loc='upper right')

    fig.suptitle('F1-score for different lambda values')
    axes.set_xlabel('Lambda')
    axes.set_ylabel('F1-score')