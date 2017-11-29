import pickle
import sys
import imageio
from scripts.process_imgs import make_network, imgs_process

MAX_BUCKET = 30


def video_process(video, net_fun, start=0, freq=50):
    bucket, inds = [], []

    def bucket_process():
        return {id: v for id, v in zip(inds, imgs_process(bucket, net_fun))}

    print(len(video))
    for i in range(start, len(video) - 1, freq):
        image = video.get_data(i)
        bucket.append(image)
        inds.append(i)
        if len(bucket) >= MAX_BUCKET:
            result = bucket_process()
            bucket, inds = [], []
            yield result
    if bucket:
        yield bucket_process()


def main():
    argv = {k.split('=')[0]: k.split('=')[1] for k in sys.argv[1:]}

    if 'dir' in argv:
        data_dir = argv['dir']
    else:
        data_dir = '.'

    if 'vd' in argv:
        if 'start' in argv:
            start = int(argv['start'])
        else:
            start = 0

        freq = 10
        file_name = argv['vd']
        vd = imageio.get_reader(file_name)

        print('Loading is finished')
        net, prob_and_vec = make_network()

        print('Network is created')
        count = start // freq

        for ind, data in enumerate(video_process(vd, prob_and_vec, start=start, freq=freq)):
            with open('{}/{}.data'.format(data_dir, count), 'wb') as fl:
                pickle.dump(data, fl)
            count += len(data)
            print('Processed {}'.format(count + start // freq))
    else:
        print('You must to write vd. For example: "vd=test.mp4".')


if __name__ == '__main__':
    main()
