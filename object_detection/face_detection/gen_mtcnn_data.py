# coding:utf-8
import os
import numpy as np
import numpy.random as npr
import pickle as pickle
import cv2
from mtcnn_model import P_Net, R_Net, O_Net
from loader import TestLoader
from detector import Detector
from fcn_detector import FcnDetector
from MtcnnDetector import MtcnnDetector
from data_utils import read_annotation, read_and_write_annotation, IoU
from utils.image_utils import IoU, convert_to_square


def gen_12net_data(anno_file, save_dir):
    pos_save_dir = os.path.join(save_dir, "positive")
    part_save_dir = os.path.join(save_dir, "part")
    neg_save_dir = os.path.join(save_dir, "negative")

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)

    pos_12_file = os.path.join(save_dir, 'pos_12.txt')
    part_12_file = os.path.join(save_dir, 'part_12.txt')
    neg_12_file = os.path.join(save_dir, 'neg_12.txt')
    if os.path.exists(pos_12_file) and os.path.exists(part_12_file) and os.path.exists(neg_12_file):
        return
    f1 = open(pos_12_file, 'w')
    f2 = open(neg_12_file, 'w')
    f3 = open(part_12_file, 'w')
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print("%d pics in total" % num)
    p_idx = 0  # positive
    n_idx = 0  # negative
    d_idx = 0  # don't care
    idx = 0
    box_idx = 0
    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_path = annotation[0]
        bbox = list(map(float, annotation[1:]))
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
        img = cv2.imread(im_path)
        idx += 1

        height, width, channel = img.shape

        neg_num = 0
        while neg_num < 5:
            size = npr.randint(12, min(width, height) / 2)
            nx = npr.randint(0, width - size)
            ny = npr.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])
            Iou = IoU(crop_box, boxes)

            cropped_im = img[ny: ny + size, nx: nx + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if len(list(Iou)) == 0:
                continue
            if np.max(Iou) < 0.3:
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(neg_save_dir + "/%s.jpg" % n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1

        # for every bounding boxes
        for box in boxes:
            x1, y1, w, h = box
            x2 = x1 + w
            y2 = y1 + h

            if max(w, h) < 20 or x1 < 0 or y1 < 0:
                continue

            for i in range(2):
                # size of the image to be cropped
                size = npr.randint(12, min(width, height) / 2)
                delta_x = npr.randint(max(-size, -x1), w)
                delta_y = npr.randint(max(-size, -y1), h)
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))
                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = IoU(crop_box, boxes)

                cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]

                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                if np.max(Iou) < 0.3:
                    # Iou with all gts must below 0.3
                    save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                    f2.write(neg_save_dir + "/%s.jpg" % n_idx + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1


            for i in range(100):
                size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                # delta here is the offset of box center
                if w < 5:
                    print(w)
                    continue

                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)

                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)
                # crop
                cropped_im = img[ny1: ny2, nx1: nx2, :]
                # resize
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                iou = IoU(crop_box, box_)
                if iou >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(pos_save_dir + "/%s.jpg" % p_idx + ' 1 %.2f %.2f %.2f %.2f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif iou >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(part_save_dir + "/%s.jpg" % d_idx + ' -1 %.2f %.2f %.2f %.2f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
            box_idx += 1
            if idx % 100 == 0:
                print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))
    f1.close()
    f2.close()
    f3.close()


def gen_imglist_pnet(data_dir):
    size = 12
    net = "PNet"

    with open(os.path.join(data_dir, '%s/pos_%s.txt' % (size, size)), 'r') as f:
        pos = f.readlines()

    with open(os.path.join(data_dir, '%s/neg_%s.txt' % (size, size)), 'r') as f:
        neg = f.readlines()

    with open(os.path.join(data_dir, '%s/part_%s.txt' % (size, size)), 'r') as f:
        part = f.readlines()

    dir_path = os.path.join(data_dir, 'imglists')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(os.path.join(dir_path)):
        os.makedirs(os.path.join(dir_path))
    with open(os.path.join(dir_path, "train_{}.txt".format(net)), "w") as f:
        nums = [len(neg), len(pos), len(part)]
        ratio = [3, 1, 1]
        base_num = min(nums)

        print(len(neg), len(pos), len(part), base_num)

        # shuffle the order of the initial data
        # if negative examples are more than 750k then only choose 750k
        if len(neg) > base_num * 3:
            neg_keep = npr.choice(len(neg), size=base_num * 3, replace=True)
        else:
            neg_keep = npr.choice(len(neg), size=len(neg), replace=True)
        pos_keep = npr.choice(len(pos), size=base_num, replace=True)
        part_keep = npr.choice(len(part), size=base_num, replace=True)
        print(len(neg_keep), len(pos_keep), len(part_keep))

        for i in pos_keep:
            f.write(pos[i])
        for i in neg_keep:
            f.write(neg[i])
        for i in part_keep:
            f.write(part[i])


def gen_24net_data():
    image_size = 24

    data_dir = '/ext/practical-ml-implementations/object_detection/face_detection/data/train/{}'.format(image_size)

    neg_dir = os.path.join(data_dir, 'negative')
    pos_dir = os.path.join(data_dir, 'positive')
    part_dir = os.path.join(data_dir, 'part')

    for dir_path in [neg_dir, pos_dir, part_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    test_mode = 'RNet'
    epoch = [18, 14, 16]
    batch_size = [2048, 256, 16]
    thresh = [0.3, 0.1, 0.7]
    min_face = 20
    stride = 2
    slide_window = False
    t_net(data_dir, image_size, epoch, batch_size, thresh, slide_window, test_mode, min_face, stride)


def gen_48net_data():
    image_size = 48
    data_dir = '/ext/practical-ml-implementations/object_detection/face_detection/data/train/{}'.format(image_size)

    neg_dir = os.path.join(data_dir, 'negative')
    pos_dir = os.path.join(data_dir, 'positive')
    part_dir = os.path.join(data_dir, 'part')

    for dir_path in [neg_dir, pos_dir, part_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    test_mode = 'RNet'
    epoch = [18, 14, 16]
    batch_size = [2048, 256, 16]
    thresh = [0.3, 0.1, 0.7]
    min_face = 20
    stride = 2
    slide_window = False
    t_net(data_dir, image_size, epoch, batch_size, thresh, slide_window, test_mode, min_face, stride)


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


def t_net(data_dir, image_size, epoch, batch_size, thresh, slide_window, test_mode="PNet", min_face_size=25, stride=2):

    prefix = ['../data/MTCNN_model/PNet_No_Landmark/PNet', '../data/MTCNN_model/RNet_No_Landmark/RNet',
              '../data/MTCNN_model/ONet_No_Landmark/ONet']

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
        pickle.dump(detections, f, 1)
    save_hard_example(image_size, data, save_path)



