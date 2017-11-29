import pickle

from net.pretrained.word2vec import load_w2v
from other.helper import get_cur_time
from other.sents_process import sent2words

DATA_DIR = 'data'


def sents_save(sents, dir_name, w2v):
    MAX_BUCKET = 2000

    bucket, names = [], []
    buc_n = 0

    def save_bucket():
        matrices = ([w2v[word] for word in words if word in w2v]
                    for words in map(sent2words, bucket))
        data = list(zip(names, matrices))
        with open('{}/{}_{}.wtov'.format(dir_name, get_cur_time(), buc_n), 'wb') as file:
            pickle.dump(data, file)

    for id, text in sents:
        bucket.append(text)
        names.append(id)

        if len(bucket) >= MAX_BUCKET:
            save_bucket()
            bucket, names = [], []
            buc_n += 1
            print('Processed {}'.format(buc_n * MAX_BUCKET))
    if bucket:
        save_bucket()
    print('Finished')


if __name__ == '__main__':
    with open('data.w2v', 'rb') as fl:
        data = pickle.load(fl)
    print('data loaded')

    w2v = load_w2v(file_name=DATA_DIR + '/GoogleNews-vectors-negative300.bin', limit=5 * 10 ** 5)
    print('w2v loaded')

    sents_save(data, DATA_DIR, w2v)
