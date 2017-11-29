from net.zoo_new.training_net import Training
from other.sents_process import sents_load


def main(argv, data_dir, build_net, version):
    argv = {k.split('=')[0]: k.split('=')[1] for k in argv[1:]}
    print('Start')

    file_name = None

    if 'file' in argv:
        file_name = argv['file']

    net = build_net(file_name)
    net.version = version
    print('Building is finished')

    danno = sents_load(data_dir)

    print('Training')
    training_c = Training(net, 40, data_dir, danno)

    training_c.training()
