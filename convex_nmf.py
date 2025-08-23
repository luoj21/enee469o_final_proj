""" Author: Jason Luo"""
import numpy as np
from tqdm import tqdm

from utils import initialize_kmeans, separate_matrix

class ConvexNMF():
    def __init__(self, X, r, tol, max_iter, random_state):
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
        
        XTX = X.T @ X
        XTX_pos, XTX_neg = separate_matrix(XTX)

        # Initialize W (n x r), G (n x r)
        W, G = initialize_kmeans(X, r, random_state)
        G_T = G.T # (r x n)
        W /= np.sum(W, axis=0, keepdims=True)
        
        self.X = X
        self.XTX_pos = XTX_pos
        self.XTX_neg = XTX_neg
        self.W = W
        self.G_T = G_T

        self.r = r
        self.tol = tol
        self.max_iter = max_iter
    
    
    def factorize(self):
        """Factorize matrix X"""
        residual_vector = np.zeros(self.max_iter)

        # Make copies to avoid overriting initialized W and G^T
        W = self.W.copy()
        G_T = self.G_T.copy()
        G = G_T.T.copy()

        for i in tqdm(range(0, self.max_iter)):
            '''Update encoding matrix'''
            G_numerator = (self.XTX_pos @ W) + (G @ W.T @ self.XTX_neg @ W)
            G_denominator = (self.XTX_neg @ W) + (G @ W.T @ self.XTX_pos @ W)

            assert np.shape(G_numerator) == np.shape(G_denominator)
            G = G * np.sqrt((G_numerator + np.finfo(float).eps) / (G_denominator + np.finfo(float).eps))
            G = np.maximum(G, np.finfo(float).eps)
            G_T = G.T

            '''Update Convex Combination Matrix'''
            W_numerator = (self.XTX_pos @ G) + (self.XTX_neg @ W @ G_T @ G)
            W_denominator = (self.XTX_neg @ G) + (self.XTX_pos @ W @ G_T @ G)

            assert np.shape(W_numerator) == np.shape(W_denominator)
            W = W * np.sqrt((W_numerator + np.finfo(float).eps) / (W_denominator + np.finfo(float).eps))
            W = W / np.sum(W, axis=0, keepdims=True)
            W = np.maximum(W, np.finfo(float).eps)

            F = self.X @ W
            residual = 0.5 * np.linalg.norm(self.X - (F@G_T), 'fro') ** 2
            residual_vector[i] = residual 

            if i > 1 and i % 10 == 0:
                #print(f'Relative error at iteration {i}: {np.abs(residual_vector[i] - residual_vector[i-1]) / np.abs(residual_vector[i-1])}')
                pass


            if i > 1 and residual_vector[i-1] > residual_vector[i] and np.abs(residual_vector[i] - residual_vector[i-1]) / np.abs(residual_vector[i-1] + np.finfo(float).eps) < self.tol:
                residual_vector = residual_vector[0:i]
                print(f'Convergence achieved at iteration {i}...')
                break

            if i == self.max_iter - 1:
                print(f'{self.max_iter} Iterations completed...')

        return F, W, G_T, residual_vector