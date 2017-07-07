"""
Please read README.md for usage instructions.

Give a path to a .npy file which contains a dictionary of model parameters.
Creates a TensorFlow Variable for each parameter and saves the session in a .ckpt file to restore later.
"""
import argparse
import numpy as np
import os
import tensorflow as tf
slim = tf.contrib.slim


def main():
    parameters = np.load(FLAGS.input)
    #  unpack the dictionary since serializing to .npy stored it in an array
    parameters = parameters[()]

    for (name, param) in parameters.iteritems():
        tf.Variable(param, name=name)
        print("Saving variable `" + name + "` of shape ", param.shape)

    global_step = slim.get_or_create_global_step()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        input_name = os.path.splitext(FLAGS.input)[0]
        save_path = saver.save(sess, input_name + '.ckpt', global_step=global_step)
        print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to a .npy file containing a dictionary of parameters'
    )
    FLAGS = parser.parse_args()
    main()
