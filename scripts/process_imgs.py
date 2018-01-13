import os
import pickle
import sys

import imageio
import numpy as np

from net.pretrained.vgg16.file_worker import make_network
from other.preprocess_img import preprocess, resize_img


def imgs_process(imgs, net_fun):
    img_np = [img
              for img in map(lambda x:
                             preprocess(
                                 resize_img(
                                     np.array(x)))[0], imgs)]
    results = net_fun(img_np)
    return [{
        'prob': result[0],
        'fc8': result[1],
        'fc7': result[2],
        'fc6': result[3]}
        for result in zip(*results)]


def files_process(names, dir_name, net_fun):
    MAX_BUCKET = 30

    bucket, inds = [], []

    def bucket_process():
        return {id: v
                for id, v in zip(inds, imgs_process(bucket, net_fun))}

    for name in names:
        image = imageio.imread('{}/{}'.format(dir_name, name))
        bucket.append(image)
        inds.append(name)
        if len(bucket) >= MAX_BUCKET:
            result = bucket_process()
            bucket, inds = [], []
            yield result
    if bucket:
        yield bucket_process()


def main():
    argv = {k.split('=')[0]: k.split('=')[1] for k in sys.argv[1:]}

    if 'dir' in argv and 'start' in argv and 'end' in argv and 'to' in argv:
        to_dir = argv['to']
        data_dir = argv['dir']
        start = int(argv['start'])
        end = int(argv['end'])

        print(data_dir, start)
        net, prob_and_vec = make_network('net/vgg16.pkl')
        print('Network is created')

        names = list(sorted(os.listdir(data_dir)))[start:end]
        print(names[0], names[-1])

        count = 0
        for ind, data in enumerate(files_process(names, data_dir, prob_and_vec)):
            with open('{}/{}.imgs'.format(to_dir, str(start) + '_' + str(ind)), 'wb') as fl:
                pickle.dump(data, fl)
            count += len(data)
            print('Processed {}'.format(count))
    else:
        print('You must to write more flags.')


if __name__ == '__main__':
    main()
