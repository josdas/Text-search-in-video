import imageio
import numpy as np
from scipy import spatial
from pandas import ewma
import net.zoo_new.model_net_2 as model_net
from backend.frames import frames_load
from net.pretrained.word2vec import load_w2v
from other.helper import pad_T
from other.sents_process import sent2words

BASE_ID = '8kBGjNI2bwA'


def link_from_time(time, id=BASE_ID):
    return "https://youtu.be/{}?t={}".format(id, time)


class QueryHandler:
    FRAMES_PER_SEC = 25

    def __init__(self, data_dir, net_file, video_file):
        frames = frames_load(data_dir)
        self.frames = {k: v['fc7'] for k, v in frames.items()}
        self.w2v = load_w2v()
        self.net = model_net.build_cnn(net_file)
        self.video = imageio.get_reader(video_file)

    def text2matrix(self, text):
        words = sent2words(text)
        vectors = [self.w2v[word] for word in words if word in self.w2v]

        bad_words = [word for word in words if word not in self.w2v]
        print("Unknown word:", *bad_words)

        if len(vectors) == 0:
            raise RuntimeError('Bad text')

        return pad_T(vectors, model_net.W)

    def get_obj_by_vec(self, vec):
        def sim(v1, v2):
            return spatial.distance.cosine(v1, v2)

        return [(id, sim(vec, point)) for id, point in self.frames.items()]

    def get_link(self, text):
        vector = self.net.predict_fun_det(
            [self.text2matrix(text)]
        )[0]
        frames = self.get_obj_by_vec(vector)

        x, y = np.transpose(frames)

        span = 2
        y = (ewma(y, span=span) + ewma(y[::-1], span=span)[::-1]) / 2
        y -= np.min(y)
        y /= np.max(y)

        frames = list(sorted(zip(x, y), key=lambda x: x[1]))

        for fr, di in frames[:4]:
            print(fr, di)
        print(np.argmin(y))

        frame_id = int(frames[0][0])
        return {'link': link_from_time(int(frame_id / QueryHandler.FRAMES_PER_SEC)),
                'img': self.video.get_data(frame_id),
                'id': frame_id}
