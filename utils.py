import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import confusion_matrix



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
