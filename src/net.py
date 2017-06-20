import abc
from enum import Enum
import tensorflow as tf
from .flowlib import flow_to_image
import numpy as np
slim = tf.contrib.slim


class Mode(Enum):
    TRAIN = 1
    TEST = 2


class Net(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, mode=Mode.TRAIN, debug=False):
        self.global_step = slim.get_or_create_global_step()
        self.mode = mode
        self.debug = debug

    @abc.abstractmethod
    def model(self, inputs, training_schedule):
        """
        Defines the model and returns a tuple of Tensors needed for calculating the loss.
        """
        return

    @abc.abstractmethod
    def loss(self, **kwargs):
        """
        Accepts prediction Tensors from the output of `model`.
        Returns a single Tensor representing the total loss of the model.
        """
        return

    def train(self, log_dir, training_schedule, input_a, input_b, flow):
        tf.summary.image("image_a", tf.expand_dims(input_a[2, :, :, :],0), max_outputs=1)
        tf.summary.image("image_b", tf.expand_dims(input_b[2, :, :, :],0), max_outputs=1)

        self.learning_rate = tf.train.piecewise_constant(
            self.global_step,
            [tf.cast(v, tf.int64) for v in training_schedule['step_values']],
            training_schedule['learning_rates'])

        optimizer = tf.train.AdamOptimizer(
            self.learning_rate,
            training_schedule['momentum'],
            training_schedule['momentum2'])

        inputs = {
            'input_a': input_a,
            'input_b': input_b,
        }
        predictions = self.model(inputs, training_schedule)
        total_loss = self.loss(flow, predictions)
        tf.summary.scalar('loss', total_loss)

        # Show the generated flow in TensorBoard
        if 'flow' in predictions:
            pred_flow_0 = predictions['flow'][2, :, :, :]
            pred_flow_0 = tf.py_func(flow_to_image, [pred_flow_0], tf.uint8)
            pred_flow_0 = tf.expand_dims(pred_flow_0, 0)
            tf.summary.image('pred_flow', pred_flow_0, max_outputs=1)

        true_flow_0 = flow[2, :, :, :]
        true_flow_0 = tf.py_func(flow_to_image, [true_flow_0], tf.uint8)
        true_flow_0 = tf.expand_dims(true_flow_0, 0)
        tf.summary.image('true_flow', true_flow_0, max_outputs=1)

        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer)

        if self.debug:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                tf.train.start_queue_runners(sess)
                slim.learning.train_step(
                    sess,
                    train_op,
                    self.global_step,
                    {
                        'should_trace': tf.constant(1),
                        'should_log': tf.constant(1),
                        'logdir': log_dir + '/debug',
                    }
                )
        else:
            slim.learning.train(
                train_op,
                log_dir,
                # session_config=tf.ConfigProto(allow_soft_placement=True),
                global_step=self.global_step,
                save_summaries_secs=60,
                number_of_steps=training_schedule['max_iter']
            )
