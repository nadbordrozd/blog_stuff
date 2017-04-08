from random import randint
from unidecode import unidecode
import joblib
import json
import numpy as np

sklearn_path = '../data/sklearn.py'
austen_path = '../data/austen.txt'
merged_path = '../data/merged.txt'
labels_path = '../data/target.txt'
features_path = '../data/X'
target_path = '../data/y'


chars = '\n !"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ{|}~'
charset = set(chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


def sanitize_text(text):
    return ''.join(c for c in unidecode(text.decode('utf-8')).replace('\t', '    ') if c in charset)


def encode_text(text):
    zeros = [[0] * len(chars) for _ in range(len(text))]
    for i, c in enumerate(text):
        zeros[i][char_indices[c]] = 1
    return zeros

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

    with open(merged_path, "wb") as out:
        out.write(merged_text)

    with open(labels_path, "wb") as out:
        json.dump(target, out)

    encoded = np.array(encode_text(merged_text))
    joblib.dump(encoded, features_path)
    joblib.dump(np.array(target), target_path)
