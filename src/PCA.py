import numpy as np
import pandas as pd
from scipy.linalg import null_space

import matplotlib.pyplot as plt
import seaborn as sns

def PCA(X_train, n_components, get_eigen = False):
    
    # define important numbers
    N = X_train.shape[0]
    m = X_train.shape[1]
    #print(f'Number of samples: {N}, Number of features: {m}')
    
    # compute mean matrix and covariance matrix
    col_means = list(X_train.mean()) # compute mean for each column
    X_mean = np.array([col_means for _ in range(N)]) # create m x m mean matrix

    S = np.array(1/N * (X_train - X_mean).T @ (X_train - X_mean))
    # eigen values and vectors
    eig_vals, eig_vectors = np.linalg.eig(S) # compute eigen values and vectors of covariance matrix
    # sort eigen values and vectors based on size of eigen value in descending order
    sorted_idxs = eig_vals.argsort()[::-1] # use [::-1] because argsort sorts in ascending order
    sorted_eig_vals = eig_vals[sorted_idxs]
    sorted_eig_vectors = eig_vectors[:,sorted_idxs]
    
    # eigen decomposition
    Lambda = np.diag(sorted_eig_vals) # diagonal matrix of eigen values in descending order
    decomposed_S = sorted_eig_vectors @ Lambda @ np.linalg.inv(sorted_eig_vectors) - S # compute eigen-decomposition of covariance matrix S
    assert decomposed_S.sum() < 10**-10, "Eigen-decomposition gave wrong results - check eigen values and vectors" # check if decomposition works - should be 0
    
    X_PCA = (np.array(X_train - X_mean) @ sorted_eig_vectors[:,:n_components])

    explained_variance = [val / sum(sorted_eig_vals) for val in sorted_eig_vals]
    
    if get_eigen:
        return X_PCA, explained_variance, sorted_eig_vals, sorted_eig_vectors, X_mean
    else:
        return X_PCA, explained_variance

def fair_PCA(X_train, n_components, groups):
    # X_train: N x m matrix

    # Compute the nullspace of z^T X and build matrix R
    z = groups
    X = X_train
    z = z - np.mean(z, axis=0)
    R = null_space(np.dot(z.T, X))
    #print(f'{R.shape=}')

    R_TXXR = R.T @ X.T @ X @ R
    #print(f'{R_TXXR.shape=}')

    eig_vals, eig_vectors = np.linalg.eig(R_TXXR)
    #print(f"{eig_vectors.shape=}")
    sorted_idxs = eig_vals.argsort()[::-1]
    sorted_eig_vals = eig_vals[sorted_idxs]
    sorted_eig_vectors = eig_vectors[:, sorted_idxs]
    largest_k_eig_vectors = sorted_eig_vectors[:, :n_components]

    # Build matrix Λ comprising the eigenvectors as columns
    Λ = largest_k_eig_vectors
    
    # Return U = RΛ
    U = R @ Λ
    #print(f'{R.shape=}, {Λ.shape=}')
    #print(f'{X_train.shape=}, {U.shape=}')
    X_fair_PCA = X_train @ U
    #print(f'{X_fair_PCA.shape=}')

    explained_variance = [val / sum(sorted_eig_vals) for val in sorted_eig_vals]
    
    return X_fair_PCA, U, explained_variance

def plot_cumulative_explained_variance(ax, explained_variance):
    cumulative_explained_variance = np.cumsum(explained_variance)
    n_components = len(explained_variance)
    ax.plot(range(1, n_components + 1), cumulative_explained_variance, marker='o')
    ax.set_xlabel('Number of PCA Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Cumulative Explained Variance vs PCA Components')
    ax.set_xticks(range(1, n_components + 1))
    ax.grid()

def project_and_plot_PCA(X_train, size = (25, 5), n_components=2, corr_metric='pearson'):
    fig, axes = plt.subplots(1, 2, figsize=size)
    
    X_PCA, explained_variance = PCA(X_train, n_components)

    # Scatter plot of the first 2 PCA components
    axes[0].scatter(X_PCA[:, 0], X_PCA[:, 1])
    axes[0].set_xlabel(f"PCA 1 - [{explained_variance[0]:.2%}]")
    axes[0].set_ylabel(f"PCA 2 - [{explained_variance[1]:.2%}]")
    axes[0].set_title("PCA Components Scatter Plot")
    axes[0].grid()

    # Cumulative explained variance plot
    plot_cumulative_explained_variance(axes[1], explained_variance)

    plt.tight_layout()
    plt.show()



def reconstruction_loss(X_train, X_test, n_components, groups, fair=False):
    loss = []
    if fair:
        for i in range(1, n_components+1):
            train_PCA, U, _ = fair_PCA(X_train, i, groups)
            test_PCA = X_test @ U
            test_recon = test_PCA @ U.T
            mse = ((X_test.values - test_recon) ** 2).mean(axis=1).mean()
            loss.append(mse)
        
    else:
        train_PCA, explained_variance, sorted_eig_vals, sorted_eig_vectors, X_mean  = PCA(X_train, n_components, get_eigen=True)
        for i in range(1, n_components+1):
            test_PCA = X_test @ sorted_eig_vectors[:,:i]
            test_recon = test_PCA @ sorted_eig_vectors[:,:i].T
            mse = ((X_test.values - test_recon) ** 2).mean(axis=1).mean()
            loss.append(mse)
    return loss

def plot_reconstruction_loss(X_train, X_test, n_components, groups, fair=False):
    recon_loss = reconstruction_loss(X_train, X_test, n_components, groups, fair=False)
    recon_loss_fair = reconstruction_loss(X_train, X_test, n_components, groups, fair=True)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(list(range(1, len(recon_loss)+1)), recon_loss, color='red', label='Standard PCA')
    ax.plot(list(range(1, len(recon_loss)+1)), recon_loss_fair, color='blue', label='Fair PCA')
    ax.set_xticks(list(range(1, len(recon_loss)+1)))
    ax.legend()
    ax.set_xlabel('Number of PCA Components')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Reconstruction Loss vs PCA Components')
    plt.show()


def plot_cumulative_explained_variance(ax, explained_variance):
    cumulative_explained_variance = np.cumsum(explained_variance)
    n_components = len(explained_variance)
    ax.plot(range(1, n_components + 1), cumulative_explained_variance, marker='o')
    ax.set_xlabel('Number of PCA Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Cumulative Explained Variance vs PCA Components')
    ax.set_xticks(range(1, n_components + 1))
    ax.grid()

def project_and_plot_PCA(X_train, n_components=2, size=(25,5), corr_metric='pearson'):
    fig, axes = plt.subplots(1, 2, figsize=size)
    
    X_PCA, explained_variance = PCA(X_train, n_components)

    # Scatter plot of the first 2 PCA components
    axes[0].scatter(X_PCA[:, 0], X_PCA[:, 1])
    axes[0].set_xlabel(f"PCA 1 - [{explained_variance[0]:.2%}]")
    axes[0].set_ylabel(f"PCA 2 - [{explained_variance[1]:.2%}]")
    axes[0].set_title("PCA Components Scatter Plot")
    axes[0].grid()

    # Cumulative explained variance plot
    plot_cumulative_explained_variance(axes[1], explained_variance)

    plt.tight_layout()
    plt.show()

def corr_plot(X_train, corr_metric, groups, n_components=2, fair=False):
    protected_features = groups 
    fig, axes = plt.subplots(figsize=(25, 5))
    # Correlation matrix heatmap
    if fair:
        X_PCA_n, _, _ = fair_PCA(X_train, n_components, groups)
    else:
        X_PCA_n, _ = PCA(X_train, n_components, get_eigen=False)
    
    if corr_metric == 'pearson':
        # Compute correlation between PCA components and protected features
        # Stack the two matrices together
        stacked_matrix = np.column_stack((protected_features, X_PCA_n))

        # Compute the correlation matrix
        corr_matrix = np.corrcoef(stacked_matrix.T)

        # Extract the relevant parts of the correlation matrix
        # (correlation between protected_features and X_PCA_n)
        corr_matrix = corr_matrix[:6, 6:]   
    elif corr_metric == 'spearman':
        corr_matrix = np.corrcoef(np.argsort(np.column_stack((protected_features, X_PCA_n)), axis=0).T)
    else:
        raise ValueError("Invalid correlation metric. Choose either 'pearson' or 'spearman'.")

    # print(corr_matrix.shape) #(6, 15)
    # print(X_PCA_n.shape) #(445, 15)
    # print(X_train.shape) #(445, 15)
    # print(protected_features.shape) #(445, 6)
    # print(np.column_stack((protected_features, X_PCA_n)).T.shape) #(21, 445)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, vmin=-.1, vmax=.1, square=True, ax=axes, fmt=".2f")
    shortened_one_hot_cols = [
    'AI/AN',
    'Asian',
    'Black',
    'NH/PI',
    'White',
    'White_Latino'
    ]
    if fair:
        print(X_PCA_n.shape)
        axes.set_xticklabels(list(range(1, X_PCA_n.shape[1]+1)))
        axes.set_title(f"Corr between Fair PCA and Prot. features")
    else:
        print(X_PCA_n.shape)
        axes.set_xticklabels(list(range(1, X_PCA_n.shape[1]+1)))
        axes.set_title(f"Corr between PCA and Prot. features")
    
    axes.set_yticklabels(shortened_one_hot_cols, va='center')
    axes.set_xlabel("Component")
    axes.set_ylabel("Protected Feature")

    plt.tight_layout()
    plt.show()