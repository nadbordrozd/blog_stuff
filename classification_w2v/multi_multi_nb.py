from __future__ import division
import numpy as np
from collections import Counter, defaultdict

class MMGNB(object):
    def __init__(self, w2v, alpha=1, beta=1):
        self.w2v = w2v
        self.alpha = alpha
        self.beta = beta
        self.vocab = None
        self.priors = {}
        self.class_word_counts = defaultdict(Counter)
        self.class_totals = defaultdict(lambda: 0)

        self.class2cov = {}
        self.class2mu = {}
        self.class2invcov = {}
        self.class2detcov = {}

        self.dim = len(w2v.itervalues().next())
        self.cache = {}

    def word_loglhood(self, class_, word):
        key = (class_, word)
        if key in self.cache:
            return self.cache[key]

        if word not in self.w2v:
            word_counts = self.class_word_counts[class_]
            if word not in word_counts:
                ret = 0
            else:
                ret = np.log((word_counts[word] + self.alpha) / (self.class_totals[class_] + self.alpha * len(word_counts)))
        else:
            x = self.w2v[word]
            mu = self.class2mu[class_]
            det = self.class2detcov[class_]
            inv = self.class2invcov[class_]
            ret = - 0.5 * np.log(det) - 0.5*((x-mu).T.dot(inv)).dot(x-mu) - 0.5 * self.dim * np.log(2 * np.pi)
        self.cache[key] = ret
        return ret

    # def word_loglhood(self, class_, word):
    #     lhood = (self.class_word_counts[class_].get(word, 0) + self.alpha) /\
    #     (self.class_totals[class_] + len(self.vocab) * self.alpha)
    #     return np.log(lhood)


    def predict_one(self, x):
        scores = {}
        for class_, prior in self.priors.items():
            log_prob = np.log(prior)
            for word in x:
                log_prob += self.word_loglhood(class_, word)
            scores[class_] = log_prob
        return max(scores.items(), key=lambda z: z[1])[0]

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def fit(self, X, y):
        self.priors = dict((class_, count/len(y)) for class_, count in Counter(y).items())

        class2vecs = defaultdict(list)
        for x, class_ in zip(X, y):
            # self.class_word_counts[class_].update(x)
            # self.class_totals[class_] += len(x)

            vectors = [self.w2v[w] for w in x if w in self.w2v]
            class2vecs[class_].extend(vectors)

        for class_, vectors in class2vecs.items():
            stacked = np.vstack((vectors))
            # print class_
            # print len(vectors)
            mu = stacked.mean(axis=0)
            cov = np.cov(stacked.T)
            reg_cov = cov + np.eye(len(mu)) * self.beta
            self.class2mu[class_] = mu
            self.class2cov[class_] = reg_cov

            self.class2invcov[class_] = np.linalg.inv(reg_cov)
            self.class2detcov[class_] = np.linalg.det(reg_cov)
        # self.vocab = set(t for x in X for t in x['words'])




# from __future__ import division
# import numpy as np
# from collections import defaultdict, Counter
#
# class MultinomialNB(object):
#     def __init__(self, alpha=1):
#         self.alpha = alpha
#         self.priors = {}
#         self.class_word_counts = defaultdict(Counter)
#         self.class_totals = defaultdict(lambda: 0)
#
#     def cond_prob(self, class_, word):
#         return ((self.class_word_counts[class_].get(word, 0)
#                  + self.alpha) /
#                 (self.class_totals[class_] +
#                 len(self.vocab) * self.alpha))
#     #             len(self.class_word_counts[class_]) * self.alpha))
#
#     def predict_one(self, x):
#         scores = {}
#         for class_, prior in self.priors.items():
#             log_prob = np.log(prior)
#             for w in x:
#                 if w in self.vocab:
#                     log_prob += np.log(self.cond_prob(class_, w))
#             scores[class_] = log_prob
#         print scores
#         return max(scores.items(), key=lambda x: x[1])[0]
#
#     def predict(self, X):
#         return [self.predict_one(x) for x in X]
#
#     def fit(self, X, y):
#         self.priors = dict((class_, count/len(y))
#                            for class_, count in Counter(y).items())
#         for x, class_ in zip(X, y):
#             self.class_word_counts[class_].update(x)
#             self.class_totals[class_] += len(x)
#         self.vocab = set(t for doc in X for t in doc)
#
#
#
# def pdf_multivariate_gauss(x, mu, cov):
#     '''
#     Caculate the multivariate normal density (pdf)
#
#     Keyword arguments:
#         x = numpy array of a "d x 1" sample vector
#         mu = numpy array of a "d x 1" mean vector
#         cov = "numpy array of a d x d" covariance matrix
# #     '''
# #     assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
# #     assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
# #     assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
# #     assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
# #     assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
#     part1 = np.linalg.det(cov)**(-1/2)
#     part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
#     return float(part1 * np.exp(part2))