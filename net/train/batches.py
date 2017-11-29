import numpy as np


def iterate_batches(X, y, batch_size):
    perm = np.arange(len(X))
    np.random.shuffle(perm)
    for i in range(0, len(X), batch_size):
        r = i + batch_size
        yield X[perm[i:r]], y[perm[i:r]]
