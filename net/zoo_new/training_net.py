import os
import pickle
import time
import numpy as np
from net.file_net import save_net
from other.helper import get_cur_time
from net.train.batches import iterate_batches
from other.helper import pad_T


class Training:
    mod = 30
    num_epochs = 20000

    def __init__(self, net, batch_size, data_dir, danno):
        self.epoch = 0
        self.losses_val = []
        self.losses_train = []
        self.net = net
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.gdir = data_dir + '/imgs/'
        self.danno = danno

        # if there are some funs in net we can use thire
        interesting_attrs = ['train_fun', 'loss_fun', 'loss_fun_det']
        for attr in interesting_attrs:
            if hasattr(net, attr):
                setattr(self, attr, getattr(net, attr))

        self.X_val = []
        self.y_val = []
        names = os.listdir(self.gdir)
        for name in names:
            if name.split('_')[0] == '0':
                _x, _y = self.load_X(self.gdir + name)
                self.X_val += _x
                self.y_val += _y
                if len(self.X_val) > 5000:
                    break
        print('Loaded {} val Xy'.format(len(self.X_val)))

    def save_info(self, file):
        with open(file, 'wb') as fl:
            info = {
                'epoch': self.epoch,
                'losses_train': self.losses_train,
                'losses_val': self.losses_val}
            pickle.dump(info, fl)

    def load_X(self, name):
        X, y = [], []
        with open(name, 'rb') as fl:
            data = pickle.load(fl)
        for id, rv in data.items():
            id = str(int(id.split('.')[0].split('_')[2]))
            for ls in self.danno[id]:
                vects = pad_T([elem for elem in ls if not isinstance(elem, str)], W)
                X.append(vects)
                y.append(rv['fc7'])
        return X, y

    def reload_Xy(self):
        names = os.listdir(self.gdir)
        print(self.gdir)
        perm = np.arange(len(names))
        np.random.shuffle(perm)
        X, y = [], []
        for name in names[:200]:
            if name.split('_')[1] != '0':
                _x, _y = self.load_X(self.gdir + name)
                X += _x
                y += _y
        self.X_train = X
        self.y_train = y
        print('Loaded {} Xy'.format(len(X)))

    def training(I, format_file='epoch_{}_{}.net'):
        last_save_time = -1
        start_time_train = time.time()
        while I.epoch < Training.num_epochs:
            I.reload_Xy()

            train_err = 0
            train_batches = 0
            start_time = time.time()

            for batch in iterate_batches(I.X_train, I.y_train, I.batch_size):
                if time.time() - last_save_time > 60 * 10:
                    save_net(I.net['last'], ('{}/' + format_file)
                             .format(I.data_dir, I.net.version, get_cur_time()))
                    I.save_info(I.net.version + '.info')
                    last_save_time = time.time()

                inputs, targets = batch

                train_err_batch = I.train_fun(inputs, targets)
                I.losses_train.append(train_err_batch)

                train_err += train_err_batch
                train_batches += 1

                if train_batches % Training.mod == 1 or not val_err or train_batches < 5:
                    val_err = I.net.loss_fun_det(I.X_val, I.y_val)
                    I.losses_val.append(val_err)

                print('LossV={:2.3f} LossT={:3.3f} AvrTime={:2.3f} Num={} TotalTime={:5.1f}m'
                      .format(float(val_err),
                              float(train_err_batch),
                              float(time.time() - start_time) / train_batches,
                              train_batches,
                              float(time.time() - start_time_train) / 60))

            I.epoch += 1
            print('Epoch {} is finished'.format(I.epoch))
