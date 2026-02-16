import unittest
import numpy as np

from utils import *
from sklearn.cluster import KMeans

class testUtils(unittest.TestCase):
    
    def test_initialize_kmeans(self):
        X = np.array([
        [1.3, 1.8, 4.8, 7.1, 5.0, 5.2, 8.0],
        [1.5, 6.9, 3.9, -5.5, -8.5, -3.9, -5.5],
        [6.5, 1.6, 8.2, -7.2, -8.7, -7.9, -5.2],
        [3.8, 8.3, 4.7, 6.4, 7.5, 3.2, 7.4],
        [-7.3, -1.8, -2.1, 2.7, 6.8, 4.8, 6.2]
        ])

        km = KMeans(n_clusters=2, random_state=123)
        _ = km.fit(X.T)
        _, _, centroids = initialize_kmeans(X, 2, 123)
        np.testing.assert_allclose(km.cluster_centers_, centroids.T, rtol=1e-8)

    
    def test_separate_matrix(self):
        X = np.array([[1, -2, 3],
                      [-4, 5, -6],
                      [7, -8, 9]])
        
        X_pos, X_neg = separate_matrix(X)
        self.assertTrue(np.all(X_pos >= 0), "X_pos has negative values")
        self.assertTrue(np.all(X_neg >= 0), "X_neg has negative values")
        X_reconstr = X_pos - X_neg
        np.testing.assert_array_equal(X, X_reconstr)

    
    def test_compute_permuted_accuracy1(self):
        cm = np.array([[30, 10], [40, 100]])
        acc = compute_permuted_accuracy(cm)
        self.assertEqual(acc, 130 / np.sum(cm))


    def test_compute_permuted_accuracy2(self):
        cm = np.array([[2, 100, 32], [4, 67, 90], [10, 1, 23]])
        acc = compute_permuted_accuracy(cm)
        self.assertEqual(acc, (100 + 90 + 10) / np.sum(cm))



if __name__ == "__main__":
    unittest.main()