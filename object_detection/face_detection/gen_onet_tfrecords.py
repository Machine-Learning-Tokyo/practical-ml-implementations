# coding:utf-8
import os
import random
import sys

import tensorflow as tf

from utils.tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple


def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, net):
    return '%s/neg.tfrecord' % (output_dir)


def gen_onet_tfrecords(dataset_dir, output_dir, net='48', name='MTCNN', shuffling=True):
    tf_filename = _get_output_filename(output_dir, name, net)
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    dataset = get_dataset(dataset_dir, net=net)
    if shuffling:
        tf_filename = tf_filename + '_shuffle'
        random.shuffle(dataset)

    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):
            if (i % 100 == 0):
                sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(dataset)))
                sys.stdout.flush()
            filename = image_example['filename']
            _add_to_tfrecord(filename, image_example, tfrecord_writer)

    print('\nFinished converting the MTCNN dataset!')


def get_dataset(dir, net='PNet'):
    item = '%s/neg_%s.txt' % (net, net)
    dataset_dir = os.path.join(dir, item)
    imagelist = open(dataset_dir, 'r')

    dataset = []
    for line in imagelist.readlines():
        info = line.strip().split(' ')
        data_example = dict()
        bbox = dict()
        data_example['filename'] = info[0]
        data_example['label'] = int(info[1])
        bbox['xmin'] = float(info[2])
        bbox['ymin'] = float(info[3])
        bbox['xmax'] = float(info[4])
        bbox['ymax'] = float(info[5])

        bbox['xlefteye'] = 0
        bbox['ylefteye'] = 0
        bbox['xrighteye'] = 0
        bbox['yrighteye'] = 0
        bbox['xnose'] = 0
        bbox['ynose'] = 0
        bbox['xleftmouth'] = 0
        bbox['yleftmouth'] = 0
        bbox['xrightmouth'] = 0
        bbox['yrightmouth'] = 0

        data_example['bbox'] = bbox
        dataset.append(data_example)

    return dataset