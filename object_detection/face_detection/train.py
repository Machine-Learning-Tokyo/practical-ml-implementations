# coding:utf-8
import os
import random
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from config import config
from read_tfrecord import read_multi_tfrecords, read_single_tfrecord


def train_model(base_lr, loss, data_num):
    lr_factor = 0.1
    global_step = tf.Variable(0, trainable=False)
    boundaries = [int(epoch * data_num / config.BATCH_SIZE) for epoch in config.LR_EPOCH]
    lr_values = [base_lr * (lr_factor ** x) for x in range(0, len(config.LR_EPOCH) + 1)]
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
    train_op = optimizer.minimize(loss, global_step)
    return train_op, lr_op


def random_flip_images(image_batch, label_batch):
    if random.choice([0, 1]) > 0:
        fliplandmarkindexes = np.where(label_batch == -2)[0]
        flipposindexes = np.where(label_batch == 1)[0]
        flipindexes = np.concatenate((fliplandmarkindexes, flipposindexes))
        for i in flipindexes:
            cv2.flip(image_batch[i], 1, image_batch[i])

    return image_batch


def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    return inputs


def train(net_factory, prefix, end_epoch, base_dir,
          display=10, base_lr=0.01, net=None):
    if net is None:
        exit()
    label_file = os.path.join(base_dir, 'train_{}.txt'.format(net))
    print(label_file)
    f = open(label_file, 'r')
    num = len(f.readlines())
    print("Total size of the dataset is: ", num)
    print(prefix)

    if net == 'PNet':
        dataset_dir = os.path.join(base_dir, 'train_{}.tfrecord_shuffle'.format(net))
        print('dataset dir is:', dataset_dir)
        image_batch, label_batch, bbox_batch = read_single_tfrecord(dataset_dir, config.BATCH_SIZE, net)

    else:
        pos_dir = os.path.join(base_dir, 'pos.tfrecord_shuffle')
        part_dir = os.path.join(base_dir, 'part.tfrecord_shuffle')
        neg_dir = os.path.join(base_dir, 'neg.tfrecord_shuffle')
        landmark_dir = os.path.join('../../DATA/imglists/RNet', 'landmark_landmark.tfrecord_shuffle')
        dataset_dirs = [pos_dir, part_dir, neg_dir, landmark_dir]
        pos_radio = 1.0 / 6
        part_radio = 1.0 / 6
        landmark_radio = 1.0 / 6
        neg_radio = 3.0 / 6
        pos_batch_size = int(np.ceil(config.BATCH_SIZE * pos_radio))
        assert pos_batch_size != 0, "Batch Size Error "
        part_batch_size = int(np.ceil(config.BATCH_SIZE * part_radio))
        assert part_batch_size != 0, "Batch Size Error "
        neg_batch_size = int(np.ceil(config.BATCH_SIZE * neg_radio))
        assert neg_batch_size != 0, "Batch Size Error "
        landmark_batch_size = int(np.ceil(config.BATCH_SIZE * landmark_radio))
        assert landmark_batch_size != 0, "Batch Size Error "
        batch_sizes = [pos_batch_size, part_batch_size, neg_batch_size, landmark_batch_size]
        image_batch, label_batch, bbox_batch, landmark_batch = read_multi_tfrecords(dataset_dirs, batch_sizes, net)

    if net == 'PNet':
        image_size = 12
        radio_cls_loss = 1.0
        radio_bbox_loss = 0.5
        radio_landmark_loss = 0.5
    elif net == 'RNet':
        image_size = 24
        radio_cls_loss = 1.0
        radio_bbox_loss = 0.5
        radio_landmark_loss = 0.5
    else:
        radio_cls_loss = 1.0
        radio_bbox_loss = 0.5
        radio_landmark_loss = 1
        image_size = 48

    input_image = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 4], name='bbox_target')

    input_image = image_color_distort(input_image)
    cls_loss_op, bbox_loss_op, L2_loss_op, accuracy_op = net_factory(input_image, label, bbox_target, training=True)
    total_loss_op = radio_cls_loss * cls_loss_op + radio_bbox_loss * bbox_loss_op + L2_loss_op
    train_op, lr_op = train_model(base_lr, total_loss_op, num)

    init = tf.global_variables_initializer()
    sess = tf.Session()

    saver = tf.train.Saver(max_to_keep=0)
    sess.run(init)

    # visualize some variables
    tf.summary.scalar("cls_loss", cls_loss_op)  # cls_loss
    tf.summary.scalar("bbox_loss", bbox_loss_op)  # bbox_loss
    tf.summary.scalar("cls_accuracy", accuracy_op)  # cls_acc
    tf.summary.scalar("total_loss", total_loss_op)  # cls_loss, bbox loss, landmark loss and L2 loss add together
    summary_op = tf.summary.merge_all()
    logs_dir = "./logs/{}".format(net)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    writer = tf.summary.FileWriter(logs_dir, sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    MAX_STEP = int(num / config.BATCH_SIZE + 1) * end_epoch
    epoch = 0
    sess.graph.finalize()
    try:

        for step in range(MAX_STEP):
            i = i + 1
            if coord.should_stop():
                break
            image_batch_array, label_batch_array, bbox_batch_array = sess.run([image_batch, label_batch, bbox_batch])

            image_batch_array = random_flip_images(image_batch_array, label_batch_array)

            _, _, summary = sess.run([train_op, lr_op, summary_op], feed_dict={input_image: image_batch_array,
                                                                               label: label_batch_array,
                                                                               bbox_target: bbox_batch_array})

            if (step + 1) % display == 0:
                cls_loss, bbox_loss, L2_loss, lr, acc = sess.run(
                    [cls_loss_op, bbox_loss_op, L2_loss_op, lr_op, accuracy_op],
                    feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array})

                total_loss = radio_cls_loss * cls_loss + radio_bbox_loss * bbox_loss + L2_loss
                print(
                    "%s : Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,L2 loss: %4f, Total Loss: %4f ,lr:%f " % (
                        datetime.now(), step + 1, MAX_STEP, acc, cls_loss, bbox_loss, L2_loss,
                        total_loss, lr))
                # print("{}: Step: {}/{} --- Accuracy: {.}")

            if i * config.BATCH_SIZE > num * 2:
                epoch = epoch + 1
                i = 0
                path_prefix = saver.save(sess, os.path.join(prefix, net), global_step=epoch * 2)
                print('path prefix is :', path_prefix)
            writer.add_summary(summary, global_step=step)
    except tf.errors.OutOfRangeError:
        print("OutOfRangeError")
    finally:
        coord.request_stop()
        writer.close()
    coord.join(threads)
    sess.close()
