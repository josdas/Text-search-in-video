import lasagne
import lasagne.nonlinearities as nonlin
from lasagne import layers
import theano.tensor as T


def build_net():
    net = {}
    input_shape = [None, 900]

    net['input_X'] = T.matrix("input X", dtype='floatX')

    net['inp'] = layers.InputLayer(input_shape, input_var=net['input_X'])

    net['dens_0'] = layers.DenseLayer(net['inp'], num_units=2000, nonlinearity=nonlin.sigmoid)

    net['dens_1'] = layers.DenseLayer(net['dens_0'], num_units=3000, nonlinearity=nonlin.sigmoid)

    net['dens_2'] = layers.DenseLayer(net['dens_1'], num_units=5000, nonlinearity=nonlin.tanh)
    net['drop_2'] = layers.DropoutLayer(net['dens_2'], p=0.1)

    net['last'] = layers.DenseLayer(net['drop_2'], num_units=4096)

    return net
