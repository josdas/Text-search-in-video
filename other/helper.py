import time
import numpy as np


def pad_T(vects, W):
    return np.pad(vects, [(0, W), (0, 0)], mode='constant', constant_values=0)[:W].T


def get_cur_time():
    return time.strftime('%d-%m-%Y_%H-%M-%S', time.localtime())
