#coding:utf-8
import sys
#sys.path.append("../")
from prepare_data.utils import convert_to_square

sys.path.insert(0, '..')
import numpy as np
import os
import pickle as pickle
import cv2
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from Detection.MtcnnDetector import MtcnnDetector
from prepare_data.data_utils import *


def save_hard_example(net, data, save_path):
    im_idx_list = data['images']
    gt_boxes_list = data['bboxes']
    num_of_images = len(im_idx_list)

    print("processing %d images in total" % num_of_images)

    neg_label_file = "../../DATA/no_LM%d/neg_%d.txt" % (net, image_size)
    neg_file = open(neg_label_file, 'w')

    pos_label_file = "../../DATA/no_LM%d/pos_%d.txt" % (net, image_size)
    pos_file = open(pos_label_file, 'w')

    part_label_file = "../../DATA/no_LM%d/part_%d.txt" % (net, image_size)
    part_file = open(part_label_file, 'w')
    det_boxes = pickle.load(open(os.path.join(save_path, 'detections.pkl'), 'rb'))
    print(len(det_boxes))
    print(num_of_images)
    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3 and neg_num < 60:
                save_file = os.path.join(neg_dir, "%s.jpg" % n_idx)
                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = get_path(pos_dir, "%s.jpg" % p_idx)
                    pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_dir, "%s.jpg" % d_idx)
                    part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    neg_file.close()
    part_file.close()
    pos_file.close()


def t_net(prefix, epoch,
             batch_size, test_mode="PNet",
             thresh=[0.6, 0.6, 0.7], min_face_size=25,
             stride=2, slide_window=False, shuffle=False, vis=False):
    detectors = [None, None, None]
    print("Test model: ", test_mode)
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    print(model_path[0])
    if slide_window:
        PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
    else:
        PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet

    if test_mode in ["RNet", "ONet"]:
        print("==================================", test_mode)
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet

    if test_mode == "ONet":
        print("==================================", test_mode)
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet
        
    basedir = '../../DATA/'
    filename = './wider_face_train_bbx_gt.txt'
    data = read_annotation(basedir, filename)
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)
    print("==================================")

    print('load test data')
    test_data = TestLoader(data['images'])
    print('finish loading')
    print('start detecting....')
    detections, _ = mtcnn_detector.detect_face(test_data)
    print('finish detecting ')
    save_net = 'RNet'
    if test_mode == "PNet":
        save_net = "RNet"
    elif test_mode == "RNet":
        save_net = "ONet"
    save_path = os.path.join(data_dir, save_net)
    print('save_path is :')
    print(save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_file = os.path.join(save_path, "detections.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(detections, f,1)
    save_hard_example(image_size, data, save_path)


if __name__ == '__main__':
    net = 'ONet'

    if net == "RNet":
        image_size = 24
    if net == "ONet":
        image_size = 48

    data_dir = '/ext/practical-ml-implementations/object_detection/face_detection/data/train/{}'.format(image_size)
    
    neg_dir = os.path.join(data_dir, 'negative')
    pos_dir = os.path.join(data_dir, 'positive')
    part_dir = os.path.join(data_dir, 'part')

    for dir_path in [neg_dir, pos_dir, part_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    test_mode = 'RNet'
    prefix = ['../data/MTCNN_model/PNet_No_Landmark/PNet', '../data/MTCNN_model/RNet_No_Landmark/RNet', '../data/MTCNN_model/ONet_No_Landmark/ONet']
    epoch = [18, 14, 16]
    batch_size = [2048, 256, 16]
    thresh = [0.3, 0.1, 0.7]
    min_face = 20
    stride = 2
    gpu = 0
    shuffle = False
    vis = False
    slide_window = False
    t_net(prefix, epoch, batch_size, test_mode, thresh, min_face, stride, slide_window, shuffle, vis=False)
