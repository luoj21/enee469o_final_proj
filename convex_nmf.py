import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

def separate_matrix(X):
    """Splits matrix in to X+ and X-
    
    Input:
    - X: m x n matrix X
    
    Output:
    - X_pos, X_neg: n x n symmetric matricies"""

    X_pos = 0.5 * (np.abs(X) + X)
    X_neg = 0.5 * (np.abs(X) - X)

    return X_pos, X_neg


def convex_nmf(X, r, tol, max_iter, random_state):

    """Convex NMF, where the objective is || X - XWG^T ||_{F}^2 where
    X: m x n
    W: n x r
    G^T: r x n 
    
    Input:
    - X: m x n matrix
    - r: number of basis elements
    - tol: stopping condition for convergence
    - max_iter: maximum number of iterations for updating rule
    
    Output:
    - F: X @ W which is m x r, or the constrained convex combination matrix
    - W: n x r feature matrix
    - G_T: r x n weight matrix
    - residual_vector: list of length max_iter that has all residuals for each iteration"""

    m, n = X.shape
    residual_vector = np.zeros(max_iter)
    eps = 1e-16
    XTX = X.T @ X
    XTX_pos, XTX_neg = separate_matrix(XTX)

    # Initialize W (n x r), G (n x r)
    W, G = initialize_kmeans(X, r, random_state)
    G_T = G.T # (r x n)

    # W = np.random.rand(n,r)
    # G = np.random.rand(n,r)
    # G_T = G.T

    W /= np.sum(W, axis=0, keepdims=True)
    F = X @ W

    for i in tqdm(range(0, max_iter)):
        '''Update encoding matrix'''
        G_numerator = (XTX_pos @ W) + (G @ W.T @ XTX_neg @ W)
        G_denominator = (XTX_neg @ W) + (G @ W.T @ XTX_pos @ W)

        assert np.shape(G_numerator) == np.shape(G_denominator)
        G = G * np.sqrt((G_numerator + eps) / (G_denominator + eps))
        G_T = G.T

        '''Update Convex Combination Matrix'''
        W_numerator = (XTX_pos @ G) + (XTX_neg @ W @ G_T @ G)
        W_denominator = (XTX_neg @ G) + (XTX_pos @ W @ G_T @ G)

        assert np.shape(W_numerator) == np.shape(W_denominator)
        W = W * np.sqrt((W_numerator + eps) / (W_denominator + eps))
        W = W / np.sum(W, axis=0, keepdims=True)

        F = X @ W
        residual = 0.5 * np.linalg.norm(X - (F@G_T), 'fro') ** 2
        residual_vector[i] = residual 

        if i > 1 and i % 10 == 0:
            print(f'Relative error at iteration {i}: {np.abs(residual_vector[i] - residual_vector[i-1]) / np.abs(residual_vector[i-1])}')


        if i > 1 and np.abs(residual_vector[i] - residual_vector[i-1]) / np.abs(residual_vector[i-1]) < tol:
            #print(check_kkt(X, G, W))
            residual_vector = residual_vector[0:i]
            print(f'Convergence achieved at iteration {i}...')
            break

        if i == max_iter - 1:
            print(f'{max_iter} Iterations completed...')

    return F, W, G_T, residual_vector



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


def check_kkt(X, G, W):
    """ Computes KKT complimentary condition as in Proposition 7"""
    ans = ((-X.T @ X @ G) + (X.T @ X @ W @ G.T @ G)) * (W ** 2)
    return np.linalg.norm(ans, 'fro')