'''
@author: Dmitry
'''

import tensorflow as tf
from common.elements.Elements import *


class LinearProgrammingDenseElement(RunElement):
    def __init__(self, num_outputs = -1, max_layers = 2, max_width = 10, threshold = 0.8, regularizer = None):
        self.max_layers = max_layers
        self.max_width = max_width
        self.num_layers = []

        self.threshold = threshold
        self.threshold_gap = 0.05 # point in or out along with some neighbourhood
        self.higher = (1.0 + self.threshold_gap) * self.threshold
        self.lower = (1.0 - self.threshold_gap) * self.threshold

        self.initialized = False
        self.num_outputs = num_outputs # usually deduced from output

    def step(self, X, y, training):
        y = tf.Print(y, ['Hello'])
        return y

    def init_out_layer(self, output):
        if self.initialized or (output is None):
            return

        self.num_outputs = output.shape[1] # onehot encoding assumed


    def build(self, inputs, dropout = 0.0, training = True, name = 'lp_dense', regularizer = None):
        X, y = inputs if isinstance(inputs, tuple) else (inputs, None)# need both input and output

        if y is None:
            training = False
            y = tf.zeros([tf.shape(X)[0], self.num_outputs], dtype=tf.float32)

        self.init_out_layer(y)

        result = tf.map_fn(lambda x: self.step(x[0], x[1], training), (X, y), dtype=tf.float32, name = 'lp', parallel_iterations = 1)
        #with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
        return result
