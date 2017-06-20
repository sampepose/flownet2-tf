import argparse
import os
import sys
import numpy as np
from progressbar import ProgressBar, Percentage, Bar
from scipy.misc import imread
import tensorflow as tf

FLAGS = None

# Values defined here: https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs
TRAIN = 1
VAL = 2


# https://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
def open_flo_file(filename):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print 'Magic number incorrect. Invalid .flo file'
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            return np.resize(data, (w[0], h[0], 2))


# https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/how_tos/reading_data/convert_to_records.py
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_dataset(indices, name):
    # Open a TFRRecordWriter
    filename = os.path.join(FLAGS.out, name + '.tfrecords')
    writeOpts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(filename, options=writeOpts)

    # Load each data sample (image_a, image_b, flow) and write it to the TFRecord
    count = 0
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(indices)).start()
    for i in indices:
        image_a_path = os.path.join(FLAGS.data_dir, '%05d_img1.ppm' % (i + 1))
        image_b_path = os.path.join(FLAGS.data_dir, '%05d_img2.ppm' % (i + 1))
        flow_path = os.path.join(FLAGS.data_dir, '%05d_flow.flo' % (i + 1))

        image_a = imread(image_a_path)
        image_b = imread(image_b_path)

        # Convert from RGB -> BGR
        image_a = image_a[..., [2, 1, 0]]
        image_b = image_b[..., [2, 1, 0]]

        # Scale from [0, 255] -> [0.0, 1.0]
        image_a = image_a / 255.0
        image_b = image_b / 255.0

        image_a_raw = image_a.tostring()
        image_b_raw = image_b.tostring()
        flow_raw = open_flo_file(flow_path).tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_a': _bytes_feature(image_a_raw),
            'image_b': _bytes_feature(image_b_raw),
            'flow': _bytes_feature(flow_raw)}))
        writer.write(example.SerializeToString())
        pbar.update(count + 1)
        count += 1
    writer.close()


def main():
    # Load train/val split into arrays
    train_val_split = np.loadtxt(FLAGS.train_val_split)
    train_idxs = np.flatnonzero(train_val_split == TRAIN)
    val_idxs = np.flatnonzero(train_val_split == VAL)

    # Convert the train and val datasets into .tfrecords format
    convert_dataset(train_idxs, 'fc_train')
    convert_dataset(val_idxs, 'fc_val')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory that includes all .ppm and .flo files in the dataset'
    )
    parser.add_argument(
        '--train_val_split',
        type=str,
        required=True,
        help='Path to text file with train-validation split (1-train, 2-validation)'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Directory for output .tfrecords files'
    )
    FLAGS = parser.parse_args()

    # Verify arguments are valid
    if not os.path.isdir(FLAGS.data_dir):
        raise ValueError('data_dir must exist and be a directory')
    if not os.path.isdir(FLAGS.out):
        raise ValueError('out must exist and be a directory')
    if not os.path.exists(FLAGS.train_val_split):
        raise ValueError('train_val_split must exist')
    main()
