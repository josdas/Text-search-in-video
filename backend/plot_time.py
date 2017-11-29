import datetime
import matplotlib.pyplot as plt
import numpy as np
from pandas import ewma
from backend.queryhandler import QueryHandler


def plot_time_similarity(query_handler, text, span=10):
    vector = query_handler.net.predict_fun_det(
        [query_handler.text2matrix(text)]
    )[0]
    frames = query_handler.get_obj_by_vec(vector)
    frames.sort(key=lambda frame: frame[0])

    x, y = np.transpose(frames)

    x = [int(s / QueryHandler.FRAMES_PER_SEC) for s in x]
    x = np.array([datetime.datetime(2013, 9, 28, 0, s // 60, s % 60)
                  for s in x])

    y = (ewma(y, span=span) + ewma(y[::-1], span=span)[::-1]) / 2
    y -= np.min(y)
    y /= np.max(y)

    y = 1 - y
    y = np.exp(y) / sum(np.exp(y))

    plt.figure(figsize=(10, 4))
    plt.plot(x, y)
    plt.show()
