from mtcnn_model import P_Net, R_Net, O_Net
from train import train
from config import config

base_dir = config.ROOT_DIR + '/data/train/imglists/'
prefix = config.ROOT_DIR + '/trained_models/'


def train_pnet():
    end_epoch = 30
    display = 10
    lr = 0.001

    train(P_Net, prefix, end_epoch, base_dir, display=display, base_lr=lr, net='PNet')


def train_rnet():
    end_epoch = 22
    display = 10
    lr = 0.001

    train(R_Net, prefix, end_epoch, base_dir, display=display, base_lr=lr, net='RNet')


def train_onet():
    end_epoch = 22
    display = 10
    lr = 0.001

    train(O_Net, prefix, end_epoch, base_dir, display=display, base_lr=lr, net='ONet')
