import theano.tensor as T
from lasagne.layers import *
import lasagne.nonlinearities as nonlin
from net.builder import create_net

H = 300
W = 20


def build_cnn(file_name=None):
    net = {}
    input_shape = [None, H, W]

    net['input_X'] = T.tensor3("input X", dtype='floatX')

    net['inp'] = InputLayer(input_shape, input_var=net['input_X'])

    net['max'] = GlobalPoolLayer(net['inp'], pool_function=T.max)
    net['min'] = GlobalPoolLayer(net['inp'], pool_function=T.min)
    net['mean'] = GlobalPoolLayer(net['inp'], pool_function=T.mean)

    net['con_2'] = Conv1DLayer(net['inp'], num_filters=96, filter_size=2, nonlinearity=None)
    net['con_3'] = Conv1DLayer(net['inp'], num_filters=64, filter_size=3, nonlinearity=None)
    net['con_4'] = Conv1DLayer(net['inp'], num_filters=64, filter_size=4, nonlinearity=None)

    boltzmann_max = lambda a, axis: T.sum(a * T.exp(a), axis=-1) / T.exp(a).sum(-1)

    net['gmax_2b'] = GlobalPoolLayer(net['con_2'], pool_function=boltzmann_max)
    net['gmax_3b'] = GlobalPoolLayer(net['con_3'], pool_function=boltzmann_max)
    net['gmax_4b'] = GlobalPoolLayer(net['con_4'], pool_function=boltzmann_max)

    net['merge'] = ConcatLayer((net['max'], net['min'], net['mean'],
                                net['gmax_2b'], net['gmax_3b'], net['gmax_4b']))

    net['batch_0'] = batch_norm(net['merge'])

    net['dens_1'] = DenseLayer(net['batch_0'], num_units=400, nonlinearity=nonlin.elu)
    net['batch_1'] = batch_norm(net['dens_1'])
    net['drop_1'] = DropoutLayer(net['batch_1'], p=0.6)

    net['dens_2'] = DenseLayer(net['drop_1'], num_units=450, nonlinearity=nonlin.elu)
    net['batch_2'] = batch_norm(net['dens_2'])
    net['drop_2'] = DropoutLayer(net['batch_2'], p=0.6)

    net['dens_3'] = DenseLayer(net['drop_2'], num_units=500, nonlinearity=nonlin.elu)
    net['batch_3'] = batch_norm(net['dens_3'])
    net['drop_3'] = DropoutLayer(net['batch_3'], p=0.6)

    net['last'] = DenseLayer(net['drop_3'], num_units=4096)

    return create_net(net, file_name=file_name)
