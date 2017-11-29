import numpy as np


def mix_text_vec2vec(sent):
    only_vec = (x for x in sent if not isinstance(x, str))
    matrix = np.vstack(tuple(only_vec))
    return np.concatenate((
        np.mean(matrix, axis=0),
        np.max(matrix, axis=0),
        np.min(matrix, axis=0)))
