import lasagne
from lasagne import layers, objectives
import theano
from theano import function, tensor
from net.neural_network import NNet
from net.file_net import load_net


def create_net(net, file_name=None):
    if 'input_X' not in net or 'last' not in net:
        raise NotImplemented('Net must have "input_X" and "last"')

    input_X = net['input_X']
    target_y = tensor.matrix("target Y", dtype='float32')

    y_predicted = layers.get_output(net['last'])
    y_predicted_det = layers.get_output(net['last'], deterministic=True)

    all_weights = layers.get_all_params(net['last'], trainable=True)

    learning_rate = theano.shared(lasagne.utils.floatX(0.002))

    loss = objectives.squared_error(target_y, y_predicted).mean()
    loss_det = objectives.squared_error(target_y, y_predicted_det).mean()

    updates = lasagne.updates.adam(loss, all_weights, learning_rate=learning_rate)

    train_fun = function([input_X, target_y],
                         loss, updates=updates, allow_input_downcast=True)

    loss_fun = function([input_X, target_y],
                        loss, allow_input_downcast=True)

    loss_fun_det = function([input_X, target_y],
                            loss_det, allow_input_downcast=True)

    predict_fun_det = function([input_X],
                               y_predicted_det, allow_input_downcast=True)

    if file_name:
        load_net(net['last'], file_name)

    return NNet(
        net=net,
        train_fun=train_fun,
        loss_fun=loss_fun,
        loss_fun_det=loss_fun_det,
        predict_fun_det=predict_fun_det,
        learning_rate=learning_rate)
