'''
@author: Dmitry
'''
import numpy as np
import tensorflow as tf
from CapsTools import *
from CapsuleLayer import *

class LossConfig():
    def __init__(self):
        self.with_reconstruction = True
        self.margin_m_plus = 0.9
        self.margin_m_minus = 0.1
        self.margin_lambda = 0.5

        self.reconstruction_coef = 0.0001

class ConfigCapsNet():
    def __init__(self):
        self.conv1_size = 256
        self.conv1_kernel = (9, 9)
        self.conv1_stride = (1, 1)

        self.conv2_size = 256 # or should it be caps1_len * 32? 
        self.conv2_kernel = (9, 9)
        self.conv2_stride = (2, 2)

        self.caps1_len = 8
        self.caps2_len = 16

        self.num_outputs = 10
        self.reconstruction_1 = 512
        self.reconstruction_2 = 1024
        self.reconstruction_3 = 784 # should be w * h
        self.loss_config = LossConfig()



class SimpleCapsNet():
    def __init__(self, config = ConfigCapsNet()):
        self.config = config

    def run(self, input_):
        config = self.config
        self.input = input_

        self.conv1 = tf.layers.conv2d(input_, config.conv1_size, config.conv1_kernel, strides = config.conv1_stride, activation = tf.nn.relu, name = 'conv1')
        self.conv2 = tf.layers.conv2d(self.conv1, config.conv2_size, config.conv2_kernel, strides = config.conv2_stride, name = 'conv2')
        # transpose to NCHW then reshape and squash
        self.caps1 = squash(reshapeToCapsules(tf.transpose(self.conv2, [0, 3, 1, 2]), config.caps1_len, axis = 1))
        self.caps_layer = CapsLayer(self.caps1, config.num_outputs, config.caps2_len)
        self.caps2 = self.caps_layer.do_dynamic_routing()
        self.output_norms = norm(self.caps2)
        self.masked_output = tf.multiply(self.caps2, maskForMaxCapsule(self.output_norms))
        self.flattened = tf.contrib.layers.flatten(self.masked_output)
        self.rec1 = tf.layers.dense(self.flattened, config.reconstruction_1, activation = tf.nn.relu, name = 'rec1')
        self.rec2 = tf.layers.dense(self.rec1, config.reconstruction_2, activation = tf.nn.relu, name = 'rec2')
        rec3_size = np.prod(input_.shape.as_list()[1:])
        self.rec3 = tf.layers.dense(self.rec2, rec3_size, name = 'rec3')
        self.reconstructed = tf.reshape(tf.sigmoid(self.rec3), tf.shape(input_))

    def lossFunction(self, targets):
        loss_config = self.config.loss_config
        margin_loss = targets * tf.square(tf.maximum(0.0, loss_config.margin_m_plus - self.output_norms))
        margin_loss += (1 - targets) * tf.square(tf.maximum(0.0, self.output_norms - loss_config.margin_m_minus))
        margin_loss = tf.reduce_sum(margin_loss, axis = 1)

        reconstruction_loss = 0.0
        if (loss_config.with_reconstruction):
            input_flattened = tf.reshape(self.input, tf.shape(self.rec3))
            reconstruction_loss = loss_config.reconstruction_coef * tf.square(input_flattened - self.rec3) #tf.nn.sigmoid_cross_entropy_with_logits(logits=self.rec3, labels = input_flattened)
            reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis = -1)
        return tf.reduce_sum(margin_loss + reconstruction_loss)

    def num_classified(self, targets):
        return tf.reduce_sum(tf.cast(tf.equal(tf.argmax(targets, axis = -1), tf.argmax(self.output_norms, axis = -1)), tf.float32))
