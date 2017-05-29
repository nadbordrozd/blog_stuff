from random import randint
from unidecode import unidecode
import joblib
import json
import numpy as np

sklearn_path = '../data/sklearn.py'
austen_path = '../data/austen.txt'
merged_path = '../data/merged.txt'

train_text_path = '../data/train.txt'
test_text_path = '../data/test.txt'
X_train_path = '../data/X_train'
X_test_path = '../data/X_test'
y_train_path = '../data/y_train'
y_test_path = '../data/y_test'

labels_path = '../data/target.txt'
features_path = '../data/X'
target_path = '../data/y'

n_train = 4000000
n_test = n_train

chars = '\n !"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ{|}~'
charset = set(chars)
n_chars = len(charset)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
char2vec = {}
for c in charset:
    vec = np.zeros(n_chars)
    vec[char_indices[c]] = 1
    char2vec[c] = vec


def sanitize_text(text):
    return ''.join(c for c in unidecode(text.decode('utf-8')).replace('\t', '    ') if c in charset)


def encode_text(text):
    return np.vstack(char2vec[c] for c in text)


if __name__ == '__main__':
    with open(sklearn_path, 'rb') as in_0, open(austen_path, 'rb') as in_1:
        sklearn = sanitize_text(in_0.read())
        austen = sanitize_text(in_1.read())

    target = []
    texts = [sklearn, austen]
    parts = []
    i = [0, 0]
    jump_sizes = [
        [40, 350],
        [40, 250]
    ]
    file_ind = 0
    while i[0] < len(texts[0]) and i[1] < len(texts[1]):
        min_size, max_size = jump_sizes[file_ind]
        size = randint(min_size, max_size)
        start_ind = i[file_ind]
        parts.append(texts[file_ind][start_ind:start_ind + size])
        i[file_ind] += size
        target += [file_ind] * size
        file_ind = 1 - file_ind

    merged_text = "".join(parts)
    target = target[:len(merged_text)]

    train_text = merged_text[:n_train]
    test_text = merged_text[n_train:n_train + n_test]

    y_train = np.array(target[:n_train]).reshape((1, n_train, 1))
    y_test = np.array(target[n_train:n_train+n_test]).reshape((1, n_test, 1))

    X_train = encode_text(train_text).reshape((1, n_train, n_chars))
    X_test = encode_text(test_text).reshape((1, n_test, n_chars))

    with open(train_text_path, "wb") as out:
        out.write(train_text)

    with open(test_text_path, 'wb') as out:
        out.write(test_text)

    joblib.dump(X_train, X_train_path)
    joblib.dump(X_test, X_test_path)

    joblib.dump(y_train, y_train_path)
    joblib.dump(y_test, y_test_path)
