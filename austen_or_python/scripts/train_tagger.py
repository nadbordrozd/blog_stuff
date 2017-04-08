import numpy as np
import json
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, TimeDistributed
from prepare_data import target_path, features_path, charset
import joblib
import os

model_path = '../models/m1'
if not os.path.exists(model_path):
    os.makedirs(model_path)

n_train = 4000000
n_test = n_train
n = n_train + n_test
n_chars = len(charset)

X_train = joblib.load(features_path)[:n_train].reshape((1, n_train, n_chars))
y_train = joblib.load(target_path)[:n_train].reshape((1, n_train, 1))

rnn_size = 64
epochs = 10
batch_size = 1

model = Sequential()
model.add(LSTM(rnn_size, return_sequences=True, input_shape=(n_train, n_chars)))
model.add(LSTM(rnn_size, return_sequences=True, input_shape=(n_train, n_chars), go_backwards=True))
model.add(Dropout(0.2))
model.add(LSTM(rnn_size, return_sequences=True, input_shape=(n_train, n_chars)))
model.add(LSTM(rnn_size, return_sequences=True, input_shape=(n_train, n_chars), go_backwards=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(units=1, activation='sigmoid')))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'binary_crossentropy'])

model.summary()
for i in range(20):
    model.fit(X_train, y_train, epochs=1)
    model.save(model_path  + '/epoch_%s.h5py' % (i + 1), overwrite=True)
