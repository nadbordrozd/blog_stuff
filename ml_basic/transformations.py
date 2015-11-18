import numpy as np

class PCA(object):
    def __init__(self, k):
        self.k = k
        self.means = None
        self.w = None

    def fit(self, X, y=None):
        self.means = X.mean(axis=0)
        cov = np.cov(X.T)
        eig_val_cov, eig_vec_cov = np.linalg.eig(cov)
        eig_pairs = list(zip(eig_val_cov, eig_vec_cov))
        top_pairs = sorted(eig_pairs, reverse=True)[:self.k]
        self.w = np.vstack([pair[1] for pair in top_pairs])
        return self

    def transform(self, X):
        return self.w.dot(X.T).T