from keras.models import load_model
from glob import glob
import numpy as np
import joblib
from train import generate_batches


def get_batches_and_text(files_a, jump_size_a, files_b, jump_size_b, batch_size, sample_len, n):
    """first yields n batches, then yields a list of texts + all the labels"""
    gen = generate_batches(files_a, jump_size_a, files_b, jump_size_b, batch_size, sample_len, True)
    texts = []
    labels = []
    for i in range(n):
        X, y, txt = gen.next()
        texts.append(txt)
        labels.append(y.reshape((batch_size, sample_len)))
        yield (X, y)
    yield ["".join(parts) for parts in zip(*texts)], np.hstack(labels)


if __name__ == "__main__":
    model_path = '../models/model_1'
    output_path = '../output/output_1'
    steps = 10
    fa = glob('../data/sklearn_clean/*.py')
    juma = [100, 200]
    fb = glob('../data/austen_clean/*.txt')
    jumb = [100, 200]
    batch_size = 4
    seq_len = 100

    model = load_model(model_path)
    gen = get_batches_and_text(fa, juma, fb, jumb, batch_size, seq_len, steps)
    predictions = model.predict_generator(gen, steps=steps, max_q_size=1)
    texts, labels = gen.next()

    joblib.dump((texts, labels, predictions), output_path)
