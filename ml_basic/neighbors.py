import numpy as np
from collections import Counter

class KNNRegressor(object):
    def __init__(self, k):
        self.X = None
        self.y = None
        self.k = k


    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        preds = []
        for x in X:
            dists = np.linalg.norm(self.X - x, axis=1)
            preds.append(self.y[np.argsort(dists)][:self.k].mean())
        return np.array(preds)


class KNNClassifier(object):
    def __init__(self, k):
        self.X = None
        self.y = None
        self.k = k


    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        preds = []
        for x in X:
            dists = np.linalg.norm(self.X - x, axis=1)
            counts = Counter(self.y[np.argsort(dists)][:self.k])
            preds.append(max(counts.items(), key=lambda x: x[0])[1])
        return preds