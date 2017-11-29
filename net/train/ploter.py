import matplotlib.pyplot as plt
import numpy as np
from net.train.train import Training


def base_plot(losses_val, losses_train):
    plt.figure(figsize=(16, 8))

    plt.subplot(221)
    plt.plot(np.arange(len(losses_val)) * Training.MOD, losses_val, 'r', losses_train, 'b')
    plt.grid()
    plt.legend(['y = loss valid', 'y = loss train'], loc='upper right')

    plt.subplot(222)
    plt.plot(losses_train[-120:])
    plt.grid()
    plt.legend(['y = loss train on last 120'], loc='upper right')

    plt.subplot(223)
    plt.plot(losses_val)
    plt.grid()
    plt.legend(['y = loss valid'], loc='upper right')

    plt.subplot(224)
    plt.plot(losses_val[-60:])
    plt.grid()
    plt.legend(['y = loss valid on last 60'], loc='upper right')

    plt.show()
