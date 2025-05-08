""" Author: Denis Fon"""

import numpy as np

from tqdm import tqdm

class StandardNMF:
    def __init__(self, V, num_bases, n_iter=100, tol = 1e-4, random_state = 100):
        self.V = V
        self.tol = tol
        self.num_bases = num_bases
        self.n_iter = n_iter

        # Rnadomly Initialize W and H
        np.random.seed(random_state)
        self.W = np.abs(np.random.rand(V.shape[0], num_bases))
        self.H = np.abs(np.random.rand(num_bases, V.shape[1]))

    def factorize(self):
        residual_vector = np.zeros(self.n_iter)
        for i in tqdm(range(self.n_iter)):
            # Update H
            numerator = self.W.T @ self.V
            denominator = (self.W.T @ self.W @ self.H) + 1e-9
            self.H *= numerator / denominator

            # Update W
            numerator = self.V @ self.H.T
            denominator = (self.W @ self.H @ self.H.T) + 1e-9
            self.W *= numerator / denominator

            residual = 0.5 * np.linalg.norm(self.V - (self.W @ self.H), 'fro') ** 2
            residual_vector[i] = residual

            if i > 1 and residual_vector[i-1] > residual_vector[i] and np.abs(residual_vector[i] - residual_vector[i-1]) / np.abs(residual_vector[i-1] + np.finfo(float).eps) < self.tol:
                residual_vector = residual_vector[0:i]
                print(f'Convergence achieved at iteration {i}...')
                break

            if i == self.n_iter - 1:
                print(f'{self.n_iter} Iterations completed...')
            


        return self.W, self.H, residual_vector
