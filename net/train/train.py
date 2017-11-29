import pickle
import time
from net.file_net import save_net
from other.helper import get_cur_time
from net.train.ploter import base_plot
from net.train.batches import iterate_batches


class Training:
    MOD = 10
    MOD_EPOCH = 5
    NUMBER_EPOCH = 2000

    def __init__(self, net, batch_size, version=''):
        self.epoch = 0
        self.version = version
        self.losses_val = []
        self.losses_train = []
        self.net = net
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.train_fun = None
        self.loss_fun = None
        self.loss_fun_det = None

        # if there are some funs in net we can use there
        interesting_attrs = ['train_fun', 'loss_fun', 'loss_fun_det']
        for attr in interesting_attrs:
            if hasattr(net, attr):
                setattr(self, attr, getattr(net, attr))

    def clean(self):
        self.epoch = 0
        self.losses_val = []
        self.losses_train = []

    def set_Xy(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        return self

    def set_fun(self, train_fun, loss_fun, loss_fun_det=None):
        self.train_fun = train_fun
        self.loss_fun = loss_fun
        self.loss_fun_det = loss_fun_det or loss_fun
        return self

    def save_info(self, file):
        with open(file, 'wb') as fl:
            info = {
                'epoch': self.epoch,
                'losses_train': self.losses_train,
                'losses_val': self.losses_val}
            pickle.dump(info, fl)

    def training(I, format_file='epoch_{}_{}.net', ploter=base_plot):
        start_time_train = time.time()
        while I.epoch < Training.NUMBER_EPOCH:
            if I.epoch % Training.MOD_EPOCH == 0:
                save_net(I.net['last'], format_file.format(I.version, get_cur_time()))

            train_err = 0
            train_batches = 0
            start_time = time.time()

            for i, batch in enumerate(iterate_batches(I.X_train, I.y_train, I.batch_size)):
                inputs, targets = batch

                train_err_batch = I.train_fun(inputs, targets)
                I.losses_train.append(train_err_batch)

                train_err += train_err_batch
                train_batches += 1

                if train_batches % Training.MOD == 1 or not val_err:
                    val_err = I.loss_fun_det(I.X_val, I.y_val)
                    I.losses_val.append(val_err)

                if ploter:
                    from IPython.display import clear_output
                    clear_output(True)
                    ploter(I.losses_val, I.losses_train)

                print('LV={:2.3f} LT={:3.3f} AvrTime={:2.3f} Num={} TotTime={:5.1f}m'
                      .format(float(val_err),
                              float(train_err_batch),
                              float(time.time() - start_time) / (i + 1),
                              i,
                              float(time.time() - start_time_train) / 60))

            if not ploter:
                val_batches = 0
                val_loss = 0

                for inputs, targets in iterate_batches(I.X_val, I.y_val, I.batch_size):
                    val_loss += I.loss_fun(inputs, targets)
                    val_batches += 1

                print("Epoch {} of {} took {:.3f}s".
                      format(I.epoch + 1, Training.NUMBER_EPOCH, time.time() - start_time))

                print("  training loss (in-iteration):\t\t{:.6f}".
                      format(train_err / train_batches))

                print("  valid loss (in-iteration):\t\t{:.6f}".
                      format(val_loss / val_batches))

            I.epoch += 1
