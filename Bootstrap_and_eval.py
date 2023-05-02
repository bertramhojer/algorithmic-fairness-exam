from model_helper import compute_cost, evaluate
from PCA import PCA, fair_PCA
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score

from scipy import optimize as opt
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio
)
# get pca components from the pca() function
def eval(X_train, X_test, y_test, y_train_enc, y_test_enc, groups):
    X_train_scaled_PCA, explained_variance, sorted_eig_vals, sorted_eig_vectors, X_mean = PCA(X_train, X_train.shape[1], get_eigen=True)
    X_test_scaled_PCA = X_test @ sorted_eig_vectors

    # get debiased data 
    X_train_scaled_fair_PCA, U, explained_variance = fair_PCA(X_train, X_train.shape[1], groups)
    X_test_scaled_fair_PCA = X_train @ U

    def train(X_train, y_train, X_test_, y_test_, groups):
        best_gamma = 0.1 # Not relevant in this func # Best computed gamma without fairness regulization #get_tuned_gamma(gammas, X_train_scaled, y_train, groups, num_folds=5, verbose=True)
        lambda_val = 0.1 # Not relevant in this func # Needs to be the tuned lambda value
        fair_loss_ = 'NO l2' # do not use l2 regularization
        betas = np.random.rand(X_train.shape[1])
        result = opt.fmin_tnc(func=compute_cost, x0=betas, maxfun = 1000, args = (X_train, y_train, lambda_val, best_gamma, fair_loss_, groups), xtol=1e-4, ftol=1e-4, approx_grad=True, messages=0)
        preds = evaluate(X_test_, y_test_, result)
        print(classification_report(y_test, preds))
        return preds

    normal_preds = train(X_train, y_train_enc, X_test, y_test_enc, groups)
    pca_preds = train(X_train_scaled_PCA, y_train_enc, X_test_scaled_PCA, y_test_enc, groups)
    fair_preds = train(X_train_scaled_fair_PCA, y_train_enc, X_test_scaled_fair_PCA, y_test_enc, groups)

    return normal_preds, pca_preds, fair_preds

def bootstrap_eval(normal_preds, pca_preds, fair_preds, y_test, n_bootstrap, sample_size):
    # Merge the 4 arrays into a single array with 4 columns and n rows
    merged_array = np.column_stack((normal_preds, pca_preds, fair_preds, y_test))

    # Perform bootstrapping and calculate accuracy and F1 scores
    accuracy_scores = []
    f1_scores = []

    for _ in range(n_bootstrap):
        # Bootstrap with replacement
        bootstrap_indices = np.random.choice(np.arange(len(merged_array)), size=sample_size, replace=True)
        bootstrap_sample = merged_array[bootstrap_indices]

        # Split the bootstrapped sample into predictions and true labels
        normal_preds_sample = bootstrap_sample[:, 0]
        pca_preds_sample = bootstrap_sample[:, 1]
        fair_preds_sample = bootstrap_sample[:, 2]
        y_test_sample = bootstrap_sample[:, 3]

        # Calculate accuracy and F1 scores for all three types of predictions
        accuracy_normal = accuracy_score(y_test_sample, normal_preds_sample)
        accuracy_pca = accuracy_score(y_test_sample, pca_preds_sample)
        accuracy_fair = accuracy_score(y_test_sample, fair_preds_sample)
        f1_normal = f1_score(y_test_sample, normal_preds_sample)
        f1_pca = f1_score(y_test_sample, pca_preds_sample)
        f1_fair = f1_score(y_test_sample, fair_preds_sample)

        # Store the results
        accuracy_scores.append((accuracy_normal, accuracy_pca, accuracy_fair))
        f1_scores.append((f1_normal, f1_pca, f1_fair))

    return accuracy_scores, f1_scores