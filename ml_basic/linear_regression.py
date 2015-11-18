import numpy as np


class LinearRegression(object):
    """linear regression that calculates and remembers cost at each step of
    training
    """
    def __init__(self, iters=100, alpha=0.1, lambda_=0.):
        self.iters = iters
        self.alpha = alpha
        self.lambda_ = lambda_
        self.theta = None
        self.cost_history = []
        self.cv_cost_history = []

    def cost(self, X, y):
        return np.sqrt(((self.predict(X) - y) ** 2).mean())

    def predict(self, X):
        return X.dot(self.theta)

    def fit(self, X, y, cv_X=None, cv_y=None):
        self.cost_history = []
        self.cv_cost_history = []
        n, dim = X.shape
        self.theta = np.zeros(dim)

        for _ in range(self.iters):
            self.theta -= self.alpha * (
                (self.predict(X) - y).dot(X) / n + self.lambda_ * self.theta)

            self.cost_history.append(self.cost(X, y))
            if cv_X is not None:
                self.cv_cost_history.append(self.cost(cv_X, cv_y))


class LinReg(object):
    def __init__(self, alpha, iters, lambda_):
        self.alpha = alpha
        self.iters = iters
        self.lambda_ = lambda_
        self.theta = None

    def cost(self, X, y):
        np.linalg.norm(self.predict(X) - y)

    def predict(self, X):
        return X.dot(self.theta)

    def fit(self, X, y):
        n, dim = X.shape
        self.theta = np.zeros(dim)
        for _ in range(self.iters):
            self.theta -= self.alpha * ((self.predict(X) - y).dot(X)/n
                                        + self.lambda_ * self.theta)
