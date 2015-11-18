from __future__ import division
import numpy as np
from collections import Counter, defaultdict

class NB(object):
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.priors = {}
        self.class_word_counts = {}
        self.class_counts = {}
        self.vocab = set()
        
    def cond_prob(self, class_, word):
        return (self.class_word_counts[class_].get(word, 0) 
                + self.alpha)/\
               (self.class_counts[class_] + 
                self.alpha * len(self.class_counts))
        
    def predict_one(self, x):
        class_log_probs = {}
        for class_, prior in self.priors.items():
            log_prob = np.log(prior)
            for word in self.vocab:
                lhood = self.cond_prob(class_, word)
                if word in x:
                    log_prob += np.log(lhood)
                else:
                    log_prob += np.log(1 - lhood)
            class_log_probs[class_] = log_prob
        print class_log_probs
        return max(class_log_probs.items(), key=lambda x: x[1])[0]
    
    def predict(self, X):
        return [self.predict_one(x) for x in X]
    
    def fit(self, X, y):
        self.vocab = list(set(word for doc in X for word in doc))
        n = len(y)
        self.class_counts = Counter(y)
        self.priors = dict((class_, c/n) 
                           for class_, c in self.class_counts.items())
        self.class_word_counts = defaultdict(Counter)
        for x, class_ in zip(X, y):
            self.class_word_counts[class_].update(x)



class NB2(object):
    """optimized version of Bernoulli Naive Bayes that doesn't need to iterate
    over all words in the vocabulary (only over the ones in the text)
    """

    def __init__(self, alpha=1):
        self.alpha = alpha
        self.priors = {}
        self.class_word_counts = {}
        self.class_counts = {}
        self.vocab = []

        self.neg_odds = {}

    def cond_prob(self, class_, word):
        if word not in self.vocab:
            return 0.5
        return ((self.class_word_counts[class_].get(word, 0)
                 + self.alpha) /
               (self.class_counts[class_]
                + len(self.class_counts) * self.alpha))


    def predict_one(self, x):
        class_log_probs = {}
        for class_, prior in self.priors.items():
            log_prob = np.log(prior) + self.neg_odds[class_]
            for word in x:
                cond_prob = self.cond_prob(class_, word)
                log_prob += np.log(cond_prob) - np.log(1 - cond_prob)

            class_log_probs[class_] = log_prob
        print class_log_probs
        return max(class_log_probs.items(), key=lambda x: x[1])[0]

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def fit(self, X, y):
        self.vocab = set(word for doc in X for word in doc)
        n = len(y)
        self.class_counts = Counter(y)
        self.priors = dict((class_, c/n)
                           for class_, c in self.class_counts.items())
        self.class_word_counts = defaultdict(Counter)
        for x, class_ in zip(X, y):
            self.class_word_counts[class_].update(x)

        for class_ in self.class_counts:
            self.neg_odds[class_] = sum(
                np.log(1 - self.cond_prob(class_, word))
                for word in self.vocab)