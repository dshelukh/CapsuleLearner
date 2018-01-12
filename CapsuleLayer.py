'''
@author: Dmitry
'''
import tensorflow as tf
from CapsTools import *

#TODO: should be convolution, not FC!
class CapsLayer():
    def __init__(self, input_, num_outputs, out_len, r = 3, name = 'capsweights'):
        # expected input shape: [<batch> x <number_of_capsules> x <input_capsule_length>]
        # shape of weight matrices: [<number_of_capsules> x<num_outputs>x<input_capsule_length>x<output_capsule_length>]
        self.input = input_
        self.name = name
        self.batch_size, self.input_caps_num, self.input_caps_len = input_.get_shape().as_list()
        self.batch_size = tf.shape(input_)[0]
        self.num_outputs = num_outputs
        self.capsule_len = out_len

        self.W = tf.Variable(tf.truncated_normal((self.input_caps_num, self.input_caps_len, num_outputs*out_len)), name = self.name) # we'll split later
        self.r = r

    def get_coupling_coef(self, b):
        return tf.nn.softmax(b)

    def do_dynamic_routing(self):
        # multiply input capsules and weights
        u_hats = tf.matmul(tf.transpose(self.input, [1, 0, 2]), self.W)
        # extract result for every output capsule
        u_hats_split = tf.reshape(u_hats, [self.input_caps_num, self.batch_size, self.capsule_len, self.num_outputs])

        # do the agreement
        b = tf.zeros([self.input_caps_num, self.batch_size, 1, self.num_outputs], tf.float32)
        for _ in range(self.r):
            c = self.get_coupling_coef(b) # softmax on last dim = P(input is important for output)
            # transpose and multiply with c then sum up matricies along input capsules dimension
            s = tf.reduce_sum(tf.multiply(u_hats_split, c), 0)

            # squash along output capsule length axis
            v = squash(s, 1)

            # update b
            # v has shape [batch_size x output_capsule_len x num_outputs] , u_hats_split: [input_caps_num x <shape of v>]
            # is there a better way to do dot product?
            b = tf.add(b, tf.reduce_sum(tf.multiply(u_hats_split, v), axis = 2, keep_dims=True))
        return tf.transpose(v, [0, 2, 1])
