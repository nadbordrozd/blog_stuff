from __future__ import division
import numpy as np
from collections import Counter, defaultdict

class MMGKNB(object):
    def __init__(self, w2v, alpha=1, sigma=1):
        self.alpha = alpha
        self.sigma = sigma
        self.w2v = w2v
        self.vocab = None
        self.priors = {}
        self.class_word_counts = defaultdict(Counter)
        self.class_totals = defaultdict(lambda: 0)

        self.class2vecs = {}
        self.cache = {}

    def vec_loglhood(self, class_, w):
        if w not in self.w2v:
            return 0
        key = (class_, w)
        if key in self.cache:
            return self.cache[key]
        x = self.w2v[w]
        prob = np.exp(-((self.class2vecs[class_] - x)**2/(2 * self.sigma**2)).sum(axis=1)).sum()
        ret = np.log(prob)
        self.cache[key] = ret
        return ret


    def predict_one(self, x):
        scores = {}
        for class_, prior in self.priors.items():
            log_prob = np.log(prior)
            for w in x:
                log_prob += self.vec_loglhood(class_, w)
            scores[class_] = log_prob
        return max(scores.items(), key=lambda z: z[1])[0]

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def fit(self, X, y):
        self.priors = dict((class_, count/len(y)) for class_, count in Counter(y).items())

        class2vecs = defaultdict(list)
        for x, class_ in zip(X, y):
            self.class_word_counts[class_].update(x)
            self.class_totals[class_] += len(x)

            vectors = [self.w2v[w] for w in x if w in self.w2v]
            class2vecs[class_].extend(vectors)

        self.class2vecs = class2vecs
        for class_ in class2vecs:
            class2vecs[class_] = np.array(class2vecs[class_])

        self.vocab = set(t for x in X for t in x)

from scipy.stats import gaussian_kde
class Better(object):
    def __init__(self, w2v, alpha=1, sigma=1):
        self.alpha = alpha
        self.sigma = sigma
        self.w2v = w2v
        self.vocab = None
        self.priors = {}
        self.cache = {}

        self.class2kde = {}


    def predict_one(self, x):
        scores = {}
        for class_, prior in self.priors.items():
            log_prob = np.log(prior)
            vectors = [self.w2v[w] for w in x if w in self.w2v]
            log_prob += np.sum(np.log(self.class2kde[class_].evaluate(vectors)))
            scores[class_] = log_prob
        return max(scores.items(), key=lambda z: z[1])[0]

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def fit(self, X, y):
        self.priors = dict((class_, count/len(y)) for class_, count in Counter(y).items())

        class2vecs = defaultdict(list)
        for x, class_ in zip(X, y):
            vectors = [self.w2v[w] for w in x if w in self.w2v]
            class2vecs[class_].extend(vectors)

        for class_ in class2vecs:
            class2vecs[class_] = np.array(class2vecs[class_])
            self.class2kde[class_] = gaussian_kde(class2vecs[class_])


        self.vocab = set(t for x in X for t in x)
