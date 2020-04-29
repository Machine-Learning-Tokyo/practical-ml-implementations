from mtcnn_model import P_Net
from train import train


def train_pnet():
    base_dir = '/ext/practical-ml-implementations/object_detection/face_detection/data/train/imglists/'
    model_path = '/ext/practical-ml-implementations/object_detection/face_detection/trained_models/'

    prefix = model_path
    end_epoch = 30
    display = 10
    lr = 0.001

    train(P_Net, prefix, end_epoch, base_dir, display=display, base_lr=lr, net='PNet')
