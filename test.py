import os
import tensorflow as tf
import numpy as np
from scipy.misc import imread
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

_preprocessing_ops = tf.load_op_library(
    tf.resource_loader.get_path_to_datafile("./src/ops/build/preprocessing.so"))


def read_flow(filename):
    """
    read optical flow from Middlebury .flo file
    :param filename: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print 'Magic number incorrect. Invalid .flo file'
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        print "Reading %d x %d flo file" % (h, w)
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
    f.close()
    return data2d


def display(img, c):
    plt.subplot(int('22' + str(c + 1)))
    plt.imshow(img[0, :, :, :])


def main():
    """
.Input("image_a: float32")
.Input("image_b: float32")
.Attr("crop: list(int) >= 2")
.Attr("params_a_name: list(string)")
.Attr("params_a_rand_type: list(string)")
.Attr("params_a_exp: list(bool)")
.Attr("params_a_mean: list(float32)")
.Attr("params_a_spread: list(float32)")
.Attr("params_a_prob: list(float32)")
.Attr("params_b_name: list(string)")
.Attr("params_b_rand_type: list(string)")
.Attr("params_b_exp: list(bool)")
.Attr("params_b_mean: list(float32)")
.Attr("params_b_spread: list(float32)")
.Attr("params_b_prob: list(float32)")
.Output("aug_image_a: float32")
.Output("aug_image_b: float32")
.Output("spatial_transform_a: float32")
.Output("inv_spatial_transform_b: float32")
    """

    crop = [364, 492]
    params_a_name = ['translate_x', 'translate_y']
    params_a_rand_type = ['uniform_bernoulli', 'uniform_bernoulli']
    params_a_exp = [False, False]
    params_a_mean = [0.0, 0.0]
    params_a_spread = [0.4, 0.4]
    params_a_prob = [1.0, 1.0]
    params_b_name = []
    params_b_rand_type = []
    params_b_exp = []
    params_b_mean = []
    params_b_spread = []
    params_b_prob = []

    with tf.Session() as sess:
        with tf.device('/gpu:0'):
            image_a = imread('./img0.ppm') / 255.0
            image_b = imread('./img1.ppm') / 255.0

            image_a_tf = tf.expand_dims(tf.to_float(tf.constant(image_a, dtype=tf.float64)), 0)
            image_b_tf = tf.expand_dims(tf.to_float(tf.constant(image_b, dtype=tf.float64)), 0)

            preprocess = _preprocessing_ops.data_augmentation(image_a_tf,
                                                              image_b_tf,
                                                              crop,
                                                              params_a_name,
                                                              params_a_rand_type,
                                                              params_a_exp,
                                                              params_a_mean,
                                                              params_a_spread,
                                                              params_a_prob,
                                                              params_b_name,
                                                              params_b_rand_type,
                                                              params_b_exp,
                                                              params_b_mean,
                                                              params_b_spread,
                                                              params_b_prob)

            out = sess.run(preprocess)

            plt.subplot(211)
            plt.imshow(image_a)
            plt.subplot(212)
            plt.imshow(out.aug_image_a[0, :, :, :])
            plt.show()

            # image_b_aug = sess.run(image_b_tf)
            #
            # display(np.expand_dims(image_a, 0), 0)
            # display(np.expand_dims(image_b, 0), 1)
            # display(image_a_aug, 2)
            # display(image_b_aug, 3)
            # plt.show()

            # o = _preprocessing_ops.flow_augmentation(flow, trans, inv_t, [4, 8])
            # print n[:, :, :]
            # print n[0, 0, 1], n[0, 0, 0]
            # print n[1, 0, 1], n[1, 0, 0]
            # print n[2, 0, 1], n[2, 0, 0]
            # print '---'
            # print sess.run(o)

            """# Goes along width first!!
            // Caffe, NKHW: ((n * K + k) * H + h) * W + w at point (n, k, h, w)
            // TF, NHWK: ((n * H + h) * W + w) * K + k at point (n, h, w, k)

            H=5, W=10, K=2
            n=0, h=1, w=5, k=0

            (2 * 10)                + c

            30      49                  n[0, 1, 5, 0]"""


print os.getpid()
raw_input("Press Enter to continue...")
main()

# Last index is channel!!

#   K

# value 13 should be at [0, 2, 7, 1] aka batch=0, height=1, width=0, channel=0. it is at index=20.
#
# items = {
#     'N': [0, 0],
#     'H': [5, 2],
#     'W': [10, 7],
#     'K': [2, 1],
# }
#
# for (i1, v1) in items.iteritems():
#     for (i2, v2) in items.iteritems():
#         for (i3, v3) in items.iteritems():
#             for (i4, v4) in items.iteritems():
#                 if ((v1[1] * v2[0] + v2[1]) * v3[0] + v3[1]) * v4[0] + v4[1] == 55:
#                     print 'found it: ', i1, i2, i3, i4
