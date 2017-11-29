import pickle

import lasagne
import theano
import theano.tensor

from net.pretrained.vgg16.vgg16 import build_model


def load_net(name):
    net = build_model()
    with open(name, 'rb') as f:
        weights = {k.decode("utf-8"): v
                   for k, v in pickle.load(f, encoding='bytes').items()}
    lasagne.layers.set_all_param_values(
        net['prob'],
        weights['param values'])
    return net


def make_network(name, deterministic=False):
    net = load_net(name)
    input_image = theano.tensor.tensor4('input')
    results = [lasagne.layers.get_output(net[layer_name], input_image, deterministic=deterministic)
               for layer_name in ['prob', 'fc8', 'fc7', 'fc6']]
    prob_and_vec = theano.function(
        [input_image],
        results)
    return net, prob_and_vec
