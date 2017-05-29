from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, TimeDistributed
from keras.callbacks import ModelCheckpoint

from glob import glob
from random import choice
import numpy as np
from text_encoding import char2vec, n_chars


def chars_from_files(list_of_files):
    while True:
        filename = choice(list_of_files)
        with open(filename, 'rb') as f:
            chars = f.read()
            for c in chars:
                yield c


def splice_texts(files_a, jump_size_a, files_b, jump_size_b):
    a_chars = chars_from_files(files_a)
    b_chars = chars_from_files(files_b)
    generators = [a_chars, b_chars]

    a_range = range(jump_size_a[0], jump_size_a[1])
    b_range = range(jump_size_b[0], jump_size_b[1])
    ranges = [a_range, b_range]

    source_ind = choice([0, 1])
    while True:
        jump_size = choice(ranges[source_ind])
        gen = generators[source_ind]
        for _ in range(jump_size):
            yield (gen.next(), source_ind)
        source_ind = 1 - source_ind


def generate_batches(files_a, jump_size_a, files_b, jump_size_b, batch_size, sample_len, return_text=False):
    gens = [splice_texts(files_a, jump_size_a, files_b, jump_size_b) for _ in range(batch_size)]
    while True:
        X = []
        y = []
        texts = []
        for g in gens:
            chars = []
            vecs = []
            labels = []
            for _ in range(sample_len):
                c, l = g.next()
                vecs.append(char2vec[c])
                labels.append([l])
                chars.append(c)
            X.append(vecs)
            y.append(labels)
            if return_text:
                texts.append(''.join(chars))

        if return_text:
            yield (np.array(X), np.array(y), texts)
        else:
            yield (np.array(X), np.array(y))


if __name__ == '__main__':
    model_path = '../models/model_2'
    fa = glob('../data/sklearn_clean/*.py')
    juma = [100, 200]
    fb = glob('../data/austen_clean/part*.txt')
    jumb = [100, 200]
    batch_size = 1024
    seq_len = 100
    rnn_size = 128
    batch_shape = (batch_size, seq_len, n_chars)
    steps_per_epoch = 1000
    epochs = 200

    model = Sequential()
    model.add(LSTM(rnn_size, return_sequences=True, batch_input_shape=batch_shape, stateful=True))
    model.add(LSTM(rnn_size, return_sequences=True, batch_input_shape=batch_shape, stateful=True,
                   go_backwards=True))
    model.add(Dropout(0.2))
    model.add(LSTM(rnn_size, return_sequences=True, batch_input_shape=batch_shape, stateful=True))
    model.add(LSTM(rnn_size, return_sequences=True, batch_input_shape=batch_shape, stateful=True,
                   go_backwards=True))
    model.add(Dropout(0.2))
    model.add(LSTM(rnn_size, return_sequences=True, batch_input_shape=batch_shape, stateful=True))
    model.add(LSTM(rnn_size, return_sequences=True, batch_input_shape=batch_shape, stateful=True,
                   go_backwards=True))
    model.add(Dropout(0.2))

    model.add(TimeDistributed(Dense(units=1, activation='sigmoid')))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'binary_crossentropy'])

    generator = generate_batches(fa, juma, fb, jumb, batch_size, seq_len)
    checkpointer = ModelCheckpoint(model_path)
    model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[checkpointer])
