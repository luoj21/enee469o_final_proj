import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from munkres import Munkres



def plot_confusion_matrix(y_true, y_pred, plot = True, normalize=False, title='Confusion Matrix'):
    """
    Plots a confusion matrix using seaborn heatmap.

    Inputs:
    - y_true: list or array of true labels
    - y_pred: list or array of predicted labels
    - labels: list of label names (optional)
    - normalize: if True, show percentages instead of raw counts
    - title: title of the plot

    Outputs:
    - cm: The resulting confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    if plot:
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    return cm


def compute_sparsity(X):
    """ Computes sparsity of a matrix by column average thresholding
    
    Inputs: 
    - X: A m x n matrix

    Outputs:
    - X_sparse: X with small values zereoed out based on column-average thresholding
    - num_non_zero: Number of non-zero elements remaining
    """

    m, n = X.shape
    X_sparse = np.zeros((m, n))

    for i in range(n):
        col = X[:, i]
        col_avg = np.mean(col)
        threshold = 0.001 * col_avg
        X_sparse[:, i] = [0 if x < threshold else x for x in col]

    num_non_zero = np.count_nonzero(X_sparse) / (m*n)
    return X_sparse, num_non_zero


def initialize_kmeans(X, r, random_state):
    """Initializing W and G matricies for Convex NMF
    
    Inputs:
    - X: m x n matrix
    - r: number of basis elements
    
    Outputs:
    - W: n x r feature matrix
    - G: n x r weight matrix """

    _, n = X.shape

    X_T = X.T   # Paper does clustering such that each column is a sample
    kmeans = KMeans(n_clusters=r, random_state=random_state)
    labels = kmeans.fit_predict(X_T)

    H = np.zeros((n, r)) # indicator matrix, where each column denotes if sample i belongs to cluster k
    H[np.arange(n), labels] = 1
 
    E = np.ones((n, r))
    G = H + (0.2 * E)

    n_k = np.sum(H, axis=0)
    Dn_inv = np.diag(1.0 / (n_k + 1e-10))  # number of samples per cluster
    W = (H + (0.2 * E)) @ Dn_inv # Smoothen W
    W = W / np.sum(W,axis = 0, keepdims=True) # Convex combination restriction

    centroids = X @ H @ Dn_inv

    return W, G, centroids


def separate_matrix(X):
    """Splits matrix in to X+ and X-
    
    Input:
    - X: m x n matrix X
    
    Output:
    - X_pos, X_neg: n x n symmetric matricies"""

    X_pos = 0.5 * (np.abs(X) + X)
    X_neg = 0.5 * (np.abs(X) - X)

    return X_pos, X_neg


def calc_snr(x: np.ndarray, x_hat: np.ndarray):
    """Calculates signal to noise ratio of a given signal based off:
    
    - J. L. Roux, S. Wisdom, H. Erdogan and J. R. Hershey, "SDR – Half-baked or Well Done?," ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 
    Brighton, UK, 2019, pp. 626-630, doi: 10.1109/ICASSP.2019.8683855. keywords: {speech enhancement;source separation;signal-to-noise-ratio;objective measure}
    
    Input:
    - x: original signal
    - x_hat: estimated, source separated signal
    
    Outputs:
    - signal to noise ratio 10 * log10(x / x - x_hat)"""


    eps = np.finfo(float).eps
    signal_power = (np.linalg.norm(x) ** 2) 
    noise_power = (np.linalg.norm(x - x_hat) ** 2)

    return 10 * np.log10(signal_power / (noise_power + eps))


def compute_permuted_accuracy(cm):
    """Computes optimized permutation accuracy based off of confusion matrix
    
    Inputs:
    - cm:  N x N confusion matrix
    
    Outputs:
    - permuted_accuracy: The sum of the diagonal of the confusion matrix where the diagonal is maximized
    after permutation"""

    max_val = np.max(cm)
    cost_mat = max_val - cm

    munkres = Munkres()
    idxs = munkres.compute(cost_mat)

    sum = 0
    for i, _ in enumerate(idxs):
        sum += cm[idxs[i]]
    
    permuted_accuracy = sum / np.sum(cm)

    return permuted_accuracy