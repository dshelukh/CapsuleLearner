'''
@author: Dmitry
'''
import unittest
import numpy as np
import tensorflow as tf
from CapsuleLayer import *

class TestCapsLayer(unittest.TestCase):
    
    def test_basic_scenario(self):
        x = tf.placeholder(tf.float32, [2, 8, 3])
        caps_layer = CapsLayer(x, 7, 5)
        weights_shape = caps_layer.W.get_shape().as_list()

        np.testing.assert_array_equal(weights_shape, [8, 3, 35])