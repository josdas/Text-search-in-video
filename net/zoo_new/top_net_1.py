from net.zoo_new.main_net import main
from net.zoo_new.model_net_3 import build_cnn
import sys

data_dir = 'param'
version = 'cnn2_4'

if __name__ == '__main__':
    main(sys.argv, data_dir, build_cnn, version)
