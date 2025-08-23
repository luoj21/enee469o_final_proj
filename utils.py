import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans



def plot_confusion_matrix(y_true, y_pred, normalize=False, title='Confusion Matrix'):
    """
    Plots a confusion matrix using seaborn heatmap.

    Inputs:
    - y_true: list or array of true labels
    - y_pred: list or array of predicted labels
    - labels: list of label names (optional)
    - normalize: if True, show percentages instead of raw counts
    - title: title of the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def compute_sparsity(X):
    """ Computes sparsity of a matrix by column average thresholding"""

    m, n = X.shape
    X_sparse = np.zeros((m, n))

    for i in range(n):
        col = X[:, i]
        col_avg = np.mean(col)
        threshold = 0.001 * col_avg
        
        X_sparse[:, i] = [0 if x < threshold else x for x in col]

    return X_sparse, np.count_nonzero(X_sparse) / (m*n)


def initialize_kmeans(X, r, random_state):
    """Initializing W and G matricies for Convex NMF
    
    Inputs:
    - X: m x n matrix
    - r: number of basis elements
    
    Outputs:
    - W: n x r feature matrix
    - G: n x r weight matrix """

    m, n = X.shape

    X_T = X.T
    kmeans = KMeans(n_clusters=r, random_state=random_state)
    labels = kmeans.fit_predict(X_T)

    H = np.zeros((n, r))
    for i in range(n):
        H[i, labels[i]] = 1
    E = np.ones((n, r))
    G = H + (0.2 * E)

    n_k = np.sum(H, axis=0)
    Dn_inv = np.diag(1.0 / (n_k + 1e-10)) 
    W = (H + (0.2 * E)) @ Dn_inv

    return W, G


def separate_matrix(X):
    """Splits matrix in to X+ and X-
    
    Input:
    - X: m x n matrix X
    
    Output:
    - X_pos, X_neg: n x n symmetric matricies"""

    X_pos = 0.5 * (np.abs(X) + X)
    X_neg = 0.5 * (np.abs(X) - X)

    return X_pos, X_neg