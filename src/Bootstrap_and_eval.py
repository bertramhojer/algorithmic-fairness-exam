from matplotlib import pyplot as plt
from model_helper import compute_cost, evaluate
from PCA import PCA, fair_PCA
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score
import seaborn as sns
from scipy import optimize as opt, stats
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
    X_test_scaled_fair_PCA = X_test @ U

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
    #print(f'{X_test_scaled_fair_PCA.shape, y_test_enc.shape=}')
    fair_preds = train(X_train_scaled_fair_PCA, y_train_enc, X_test_scaled_fair_PCA, y_test_enc, groups)

    return normal_preds, pca_preds, fair_preds

def bootstrap_eval_one(unfair_preds, fair_preds, y_test, n_bootstrap, sample_size):
    # Merge the arrays into a single array with columns for unfair_preds, fair_preds (if available), and y_test
    if fair_preds is None:
        merged_array = np.column_stack((unfair_preds, y_test))
    else:
        merged_array = np.column_stack((unfair_preds, fair_preds, y_test))

    # Perform bootstrapping and calculate accuracy, F1 scores, precision, and recall
    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for _ in range(n_bootstrap):
        # Bootstrap with replacement
        bootstrap_indices = np.random.choice(np.arange(len(merged_array)), size=sample_size, replace=True)
        bootstrap_sample = merged_array[bootstrap_indices]

        # Split the bootstrapped sample into predictions and true labels
        unfair_preds_sample = bootstrap_sample[:, 0]
        y_test_sample = bootstrap_sample[:, -1]

        # Calculate accuracy, F1 scores, precision, and recall for unfair predictions
        accuracy_unfair = accuracy_score(y_test_sample, unfair_preds_sample)
        f1_unfair = f1_score(y_test_sample, unfair_preds_sample, average='macro')
        precision_unfair = precision_score(y_test_sample, unfair_preds_sample, average='weighted')
        recall_unfair = recall_score(y_test_sample, unfair_preds_sample, average='weighted')

        # If fair_preds is available, calculate accuracy, F1 scores, precision, and recall for fair predictions
        if fair_preds is not None:
            fair_preds_sample = bootstrap_sample[:, 1]
            accuracy_fair = accuracy_score(y_test_sample, fair_preds_sample)
            f1_fair = f1_score(y_test_sample, fair_preds_sample, average='macro')
            precision_fair = precision_score(y_test_sample, fair_preds_sample, average='weighted')
            recall_fair = recall_score(y_test_sample, fair_preds_sample, average='weighted')
            
            accuracy_scores.append((accuracy_unfair, accuracy_fair))
            f1_scores.append((f1_unfair, f1_fair))
            precision_scores.append((precision_unfair, precision_fair))
            recall_scores.append((recall_unfair, recall_fair))
        else:
            accuracy_scores.append(accuracy_unfair)
            f1_scores.append(f1_unfair)
            precision_scores.append(precision_unfair)
            recall_scores.append(recall_unfair)

    return accuracy_scores, f1_scores, precision_scores, recall_scores


def plot_violin_metrics_with_ci_single(accuracy_scores, f1_scores, precision_scores, recall_scores, confidence_level=0.95):
    # Convert lists to NumPy arrays for easier manipulation
    accuracy_scores = np.array(accuracy_scores)
    f1_scores = np.array(f1_scores)
    precision_scores = np.array(precision_scores)
    recall_scores = np.array(recall_scores)
 
    # Calculate confidence intervals
    def confidence_interval(data, confidence_level):
        mean = np.mean(data)
        stderr = stats.sem(data)
        margin = stderr * stats.t.ppf((1 + confidence_level) / 2, len(data) - 1)
        return mean - margin, mean + margin
 
    accuracy_ci = confidence_interval(accuracy_scores, confidence_level)
    f1_ci = confidence_interval(f1_scores, confidence_level)
    precision_ci = confidence_interval(precision_scores, confidence_level)
    recall_ci = confidence_interval(recall_scores, confidence_level)
 
    # Combine accuracy and F1 scores in a single DataFrame for easier plotting
    data = np.column_stack((accuracy_scores, f1_scores, recall_scores, precision_scores))
    df = pd.DataFrame(data, columns=['Accuracy', 'F1', 'Recall', 'Precision'])
 
    # Plot violin plot for accuracy and F1 scores
    fig, ax = plt.subplots(figsize=(8, 6))
 
    sns.violinplot(data=df, palette=['skyblue', 'lightgreen', 'blue', 'darkblue'], ax=ax)
    ax.set_title('Accuracy, F1 Scores, Recall, Precision')
    ax.set_ylabel('Score')
 
 
    plt.show()
 
    # Print confidence intervals
    print(f'Accuracy CI: {accuracy_ci}')
    print(f'F1 CI: {f1_ci}')
    print(f'Precision CI: {precision_ci}')
    print(f'Recall CI: {recall_ci}')