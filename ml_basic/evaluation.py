from __future__ import division
import numpy as np
from collections import Counter


def train_test_split(X, y, ratio=0.8):
    n = len(y)
    inds = range(n)
    np.random.shuffle(inds)
    train_inds = inds[:int(ratio * n)]
    test_inds = inds[int(ratio * n):]
    return X[train_inds], y[train_inds], X[test_inds], y[test_inds]


def k_fold(X, y, k):
    n = len(y)
    inds = range(n)
    np.random.shuffle(inds)
    thresholds = [(n + 1) * i // k for i in range(k + 1)]
    for i in range(k):
        left, right = thresholds[i], thresholds[i+1]
        test_inds = inds[left: right]
        train_inds = inds[:left] + inds[right:]
        X_train = X[train_inds]
        y_train = y[train_inds]
        X_test = X[test_inds]
        y_test = y[test_inds]
        yield X_train, y_train, X_test, y_test


def mse(pred, actual):
    return np.linalg.norm(pred - actual)/np.sqrt(len(pred))

def mae(pred, actual):
    return np.linalg.norm(pred - actual, 1)/np.sqrt(len(pred))

def test_regression(model, X, y, k=6):
    cost = 0
    for X_train, y_train, X_test, y_test in k_fold(X, y, k):
        model.fit(X_train, y_train)
        cost += mse(model.predict(X_test), y_test)
    return cost/k

def accuracy(preds, y):
    return (y == preds).sum()/len(y)


def roc_curve(probs, y):
    n = len(y)
    pos = y.sum()
    order = np.argsort(-probs)
    y_ = y[order]

    points = [[0, 0]]
    tp, fp = 0, 0
    i = 0
    for actual in y_:
        if actual:
            tp += 1
        else:
            fp += 1
        points.append([fp/pos, tp/pos])
        i += 1
    points.append([1, 1])
    return np.array(points)

def true_roc(probs, y):
    n = len(y)
    pos = y.sum()
    order = np.argsort(-probs)
    y_ = y[order]
    probs = probs[order]

    points = [[0, 0]]
    tp, fp = 0, 0
    i = 0
    while i < n:
        curr = probs[i]
        while probs[i] == curr:
            if y_[i]:
                tp += 1
            else:
                fp += 1
            i += 1
            if i == n:
                break
        points.append([fp/pos, tp/pos])
    points.append([1, 1])
    return np.array(points)


def auc(points):
    """points must be ordered by x"""
    area = 0
    prev_x, prev_y = points[0]
    for x, y in points[1:]:
        area += (x - prev_x) * (y + prev_y) / 2
        prev_x, prev_y = x, y
    return area



def benchmark(model, X, y, k=5):
    import pylab
    aucs = []
    for X_train, y_train, X_test, y_test in k_fold(X, y, k):
        model.fit(X_train, y_train)
        roc = roc_curve(model.predict(X_test), y_test)
        pylab.plot(roc[:, 0], roc[:, 1])
        aucs.append(auc(roc))
    pylab.legend(["auc = %s" % a for a in aucs])
    print np.mean(aucs)

def multinomial_benchmark(model, X, y, k=5):
    accs = []
    for X_train, y_train, X_test, y_test in k_fold(X, y, k):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accs.append(accuracy(preds, y_test))
    print np.mean(accs)


class DummyClassifier(object):
    def __init__(self):
        self.prediction = None

    def predict(self, X):
        return np.array([self.prediction] * len(X))

    def fit(self, X, y):
        counts = Counter(y)
        self.prediction = max(counts.items(), key=lambda x: x[1])[0]
        return self

class DummyRegressor(object):
    def __init__(self):
        self.prediction = None

    def predict(self, X):
        return np.array([self.prediction] * len(X))

    def fit(self, X, y):
        self.prediction = y.mean()
        return self