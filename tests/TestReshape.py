'''
@author: Dmitry
'''
import unittest
import numpy as np
import tensorflow as tf
from CapsTools import *

class TestReshape(unittest.TestCase):
    def run_reshape(self, _arr, _ndim, _axis):
        #input_shape = tf.placeholder(tf.int32, [None])
        input_arr = tf.placeholder(tf.int32, _arr.shape)
        ndim = _ndim
        axis = _axis
        result = reshapeToCapsules(input_arr, ndim, axis)

        with tf.Session() as s:
            output = s.run(result, feed_dict = {input_arr: _arr})
            s.close()
        return output

    def test_reshape_1(self):
        arr = np.array([[[1, 3, 5], [7, 9, 11]],
                        [[2, 4, 6], [8, 10, 12]]])
        result = self.run_reshape(arr, 2, 0) # expecting 6x2
        np.testing.assert_array_equal(result, [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

    def test_reshape_2(self):
        arr = np.array([[[1, 3, 5], [2, 4, 6]],
                        [[7, 9, 11], [8, 10, 12]]])
        result = self.run_reshape(arr, 2, 1) # expecting 2x3x2
        np.testing.assert_array_equal(result, [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])

    def test_reshape_3(self):
        arr = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]],
                        [[9, 10, 11, 12], [13, 14, 15, 16]]])
        result = self.run_reshape(arr, 2, 2) # expecting 2x2x2x2
        np.testing.assert_array_equal(result, [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])

    def test_reshape_4(self):
        # 2x3x2x2
        arr = np.array([
                            [[[1, 4], [7, 10]], [[2, 5], [8, 11]], [[3, 6], [9, 12]]],
                            [[[101, 104], [107, 110]], [[102, 105], [108, 111]], [[103, 106], [109, 112]]],
                       ])
        result = self.run_reshape(arr, 3, 1) # expecting 2x4x3
        np.testing.assert_array_equal(result, [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                                               [[101, 102, 103], [104, 105, 106], [107, 108, 109], [110, 111, 112]]])

    def test_reshape_5(self):
        # 1x6x2x2
        arr = np.array([
                            [[[1, 4], [7, 10]], [[2, 5], [8, 11]], [[3, 6], [9, 12]],
                            [[101, 104], [107, 110]], [[102, 105], [108, 111]], [[103, 106], [109, 112]]],
                       ])
        result = self.run_reshape(arr, 3, 1) # expecting 1x8x3
        np.testing.assert_array_equal(result, [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
                                               [101, 102, 103], [104, 105, 106], [107, 108, 109], [110, 111, 112]]])
        
        
        
        
        
        
        
        