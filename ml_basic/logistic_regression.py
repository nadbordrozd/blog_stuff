import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression(object):
    def __init__(self, alpha=0.1, iters=100, lambda_=0):
        self.theta = None
        self.alpha = alpha
        self.iters = iters
        self.lambda_ = lambda_

        self.cost_history = []
        self.cv_cost_history = []


    def predict(self, X):
        return sigmoid(X.dot(self.theta))


    def cost(self, X, y):
        preds = self.predict(X)
        return (y * np.log(preds) + np.log(1 - preds) * (1 - y)).mean()

    def fit(self, X, y, test_X=None, test_y=None):
        n, dim = X.shape
        self.theta = np.zeros(dim)

        for _ in xrange(self.iters):
            self.theta -= self.alpha * (
                (self.predict(X) - y).dot(X) / n + self.lambda_ * self.theta)
            self.cost_history.append(self.cost(X, y))
            if test_X is not None:
                self.cv_cost_history.append(self.cost(test_X, test_y))

        return self


class MultinomialLogisticRegression(object):
    def __init__(self, alpha=0.1, iters=100, lambda_=0):
        self.alpha = alpha
        self.iters = iters
        self.lambda_ = lambda_
        self.regressors = None
        self.labels = None

    def predict(self, X):
        n = len(X)
        predictions = np.hstack([r.predict(X).reshape(n, 1) for r in self.regressors])
        inds = np.argmax(predictions, axis=1)
        return [self.labels[ind] for ind in inds]

    def fit(self, X, y):
        self.labels = list(set(y))
        self.regressors = [
            LogisticRegression(self.alpha, self.iters, self.lambda_) for _ in
            self.labels]
        for label, regressor in zip(self.labels, self.regressors):
            y_ = y == label
            regressor.fit(X, y_)

        return self



