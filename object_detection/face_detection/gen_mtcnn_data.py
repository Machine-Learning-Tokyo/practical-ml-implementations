# coding:utf-8
import os
import cv2
import numpy as np
import numpy.random as npr

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
        # img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
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
                # max here not really necessary
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))
                # if the right bottom point is out of image then skip
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

            # generate positive examples and part faces

            for i in range(100):
                # pos and part face size [minsize*0.8,maxsize*1.25]
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