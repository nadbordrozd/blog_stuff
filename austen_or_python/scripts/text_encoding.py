from unidecode import unidecode
import numpy as np

chars = '\n !"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ{|}~'
charset = set(chars)
n_chars = len(charset)
char2ind = dict((c, i) for i, c in enumerate(chars))
ind2char = dict((i, c) for i, c in enumerate(chars))

char2vec = {}
for c in charset:
    vec = np.zeros(n_chars)
    vec[char2ind[c]] = 1
    char2vec[c] = vec


def encode_text(text):
    return np.vstack(char2vec[c] for c in text)


def sanitize_text(text):
    return ''.join(c for c in unidecode(text.decode('utf-8')).replace('\t', '    ') if c in charset)
