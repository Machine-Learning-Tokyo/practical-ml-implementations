#coding:utf-8
from train_models.mtcnn_model import R_Net
from train_models.train import train


base_dir = '/ext/practical-ml-implementations/object_detection/face_detection/data/train/imglists_noLM/RNet'

model_name = 'MTCNN'
model_path = '../data/%s_model/RNet_No_Landmark/RNet' % model_name
prefix = model_path
end_epoch = 22
display = 100
lr = 0.001

train(R_Net, prefix, end_epoch, base_dir, display=display, base_lr=lr)

