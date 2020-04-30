from gen_traininig_data import gen_labels

train_dir = '/ext/practical-ml-implementations/object_detection/face_detection/data/train/images_larger/'
train_labels = '/ext/practical-ml-implementations/object_detection/face_detection/data/train/opencv_labels.csv.larger'

gen_labels(train_dir, train_labels)