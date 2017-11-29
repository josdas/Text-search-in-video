import pickle
import lasagne


def save_net(net, file_name):
    with open(file_name, 'wb') as fl:
        net_info = {'params': lasagne.layers.get_all_param_values(net)}
        pickle.dump(net_info, fl, protocol=pickle.HIGHEST_PROTOCOL)


def load_net(net, file_name):
    with open(file_name, 'rb') as fl:
        net_info = pickle.load(fl)
        lasagne.layers.set_all_param_values(net, net_info['params'])
