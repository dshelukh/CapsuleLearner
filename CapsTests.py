'''
@author: Dmitry
'''
import unittest
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', reshape=False, one_hot=True, validation_size = 5000)

from SimpleCapsNet import *
from Trainer import *


class CapsTests(unittest.TestCase):
    def setUp(self):
        self.y = tf.Variable([1.0, 1.0], trainable = False)

    def addY(self, x):
        self.y = tf.subtract(self.y, tf.multiply(self.y, 0.01))
        return tf.add(x, self.y)

    @unittest.skip("skip basic")
    def test_basic_scenario(self):
        x = tf.Variable(tf.truncated_normal([2], 3.0, 1.0, dtype = tf.float32))
        
        z = self.addY(x)
        epochs = 250
        learning_rate = 0.1

        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        sqr_minimizer = optimizer.minimize(tf.square(z))

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for i in range(epochs):
                #self.addY(x)
                _, val, val2 = session.run((x, x, self.addY(x)))
                print(val, val2)
        np.testing.assert_allclose(val, [0.0, 0.0], rtol = 0.0, atol = 0.0001)

    def test_1(self):
        placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1])
        network = SimpleCapsNet()
        network.run(placeholder)
        saver = CustomSaver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.save_session(sess, True, (90))
        with tf.Session() as sess2:
            saver.restore_session(sess2, True)
            vals = sess2.run(network.caps_layer.W, {placeholder: mnist.train.images[0:2]})
        np.testing.assert_array_equal(list(vals.shape), [1152, 8, 160])

    def test_2(self):
        p = tf.placeholder(tf.float32, [None, 4])
        r = p[:, :3]
        with tf.Session() as sess:
            pr = sess.run(r, feed_dict = {p: [[1, 2, 3, 4], [5, 6, 7, 8]]})
        np.testing.assert_array_equal(pr, [[1, 2, 3], [5, 6, 7]])

    def test_3(self):
        x = tf.placeholder(tf.float32, [None, 2])
        y = tf.placeholder(tf.float32, [None, 2])
        z = tf.square(x)
        grads = tf.gradients(z, x, grad_ys = y)
        grads2 = tf.gradients(grads, x)

        with tf.Session() as sess:
            r = sess.run(grads2, feed_dict={x: [[2, 2], [4, 4]], y:[[3, 3], [5, 5]]})
        np.testing.assert_array_equal(r, [[[6, 6], [10, 10]]]) # 2 * y









