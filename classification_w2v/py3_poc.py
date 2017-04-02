# this works with python3
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # this line is different from python2 version - no more itervalues
        self.dim = len(list(word2vec.values())[0])

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

w2v = {
    'Berlin': [1, 1],
    'London': [1.01, 1.01],
    'Madrid': [1.02, 1.02],
    'cow':    [-1, -1],
    'cat':    [-1.01, -1.01],
    'dog':    [-1.02, -1.02],
    'pink':   [1, -1],
    'yellow': [1.01, -1.01],
    'red':    [1.02, -1.02]
}


model = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])

X = [['Berlin', 'London'],
     ['cow', 'cat'],
     ['pink', 'yellow']]
y = ['capitals', 'animals', 'colors']

model.fit(X, y)

# never before seen words!!!
test_X = [['dog'], ['red'], ['Madrid']]

print(model.predict(test_X))
# prints ['animals' 'colors' 'capitals']
