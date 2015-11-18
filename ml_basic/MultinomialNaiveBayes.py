from __future__ import division
import numpy as np
from collections import defaultdict, Counter

class MultinomialNB(object):
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.priors = {}
        self.class_word_counts = defaultdict(Counter)
        self.class_totals = defaultdict(lambda: 0)

    def cond_prob(self, class_, word):
        return ((self.class_word_counts[class_].get(word, 0)
                 + self.alpha) /
                (self.class_totals[class_] +
                len(self.vocab) * self.alpha))
    #             len(self.class_word_counts[class_]) * self.alpha))

    def predict_one(self, x):
        scores = {}
        for class_, prior in self.priors.items():
            log_prob = np.log(prior)
            for w in x:
                if w in self.vocab:
                    log_prob += np.log(self.cond_prob(class_, w))
            scores[class_] = log_prob
        print scores
        return max(scores.items(), key=lambda x: x[1])[0]

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def fit(self, X, y):
        self.priors = dict((class_, count/len(y))
                           for class_, count in Counter(y).items())
        for x, class_ in zip(X, y):
            self.class_word_counts[class_].update(x)
            self.class_totals[class_] += len(x)
        self.vocab = set(t for doc in X for t in doc)