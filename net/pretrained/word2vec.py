import gensim


def load_w2v(limit=2 * 10 ** 5, file_name='data/GoogleNews-vectors-negative300.bin'):
    return gensim.models.KeyedVectors.load_word2vec_format(
        file_name, binary=True, limit=limit)
