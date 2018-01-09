'''
@author: Dmitry
'''
import unittest
import numpy as np
import tensorflow as tf
import math
from CapsTools import *

class TestSquash(unittest.TestCase):
    def run_tool(self, _arr, _axis = -1, tool = squash):
        #input_shape = tf.placeholder(tf.int32, [None])
        input_arr = tf.placeholder(tf.float32, _arr.shape)
        axis = _axis
        result = tool(input_arr, axis) if axis != -1 else tool(input_arr)

        with tf.Session() as s:
            output = s.run(result, feed_dict = {input_arr: _arr})
        return output

    def test_squash_1(self):
        arr = np.array([[1.0, 1.0], [1.0, 1.0]])
        result = self.run_tool(arr, [0, 1])
        np.testing.assert_allclose(result, [[0.4, 0.4], [0.4, 0.4]], atol = 0.000001)

    def test_squash_2(self):
        arr = np.array([[1.0, 1.0], [1.0, 0.0]])
        result = self.run_tool(arr)
        val = math.sqrt(2) / 3
        np.testing.assert_allclose(result, [[val, val], [0.5, 0.0]], atol = 0.000001)

    def test_squash_3(self):
        arr = np.array([[1.0, 1.0], [1.0, 0.0]])
        result = self.run_tool(arr, [0])
        val = math.sqrt(2) / 3
        np.testing.assert_allclose(result, [[val, 0.5], [val, 0.0]], atol = 0.000001)



    def test_norm_1(self):
        arr = np.array([1.0, 0.0])
        result = self.run_tool(arr, tool = norm)
        np.testing.assert_allclose(result, [1.0], atol = 0.000001)

    def test_norm_2(self):
        arr = np.array([[4.0, 4.0], [3.0, 3.0]])
        result = self.run_tool(arr, 0, tool = norm)
        np.testing.assert_allclose(result, [5.0, 5.0], atol = 0.000001)

    def test_l1norm_1(self):
        arr = np.array([1.0, -1.0])
        result = self.run_tool(arr, tool = l1norm)
        np.testing.assert_allclose(result, [2.0], atol = 0.000001)

    def test_l1norm_2(self):
        arr = np.array([[1.0, -2.0], [-3.0, 4.0]])
        result = self.run_tool(arr, 0, tool = l1norm)
        np.testing.assert_allclose(result, [4.0, 6.0], atol = 0.000001)

    def test_l2norm_1(self):
        arr = np.array([1.5, 0.5])
        result = self.run_tool(arr, tool = l2norm)
        np.testing.assert_allclose(result, [2.5], atol = 0.000001)

    def test_l2norm_2(self):
        arr = np.array([[-3.0, -1.0], [4.0, 0.0]])
        result = self.run_tool(arr, 0, tool = l2norm)
        np.testing.assert_allclose(result, [25.0, 1.0], atol = 0.000001)

    def test_mask_1(self):
        arr = np.array([[1, 2, 3], [3, 2, 1]])
        result = self.run_tool(arr, tool = maskForMaxCapsule)
        np.testing.assert_allclose(result, [[[0], [0], [1]], [[1], [0], [0]]], atol = 0.000001)

    def test_mask_2(self):
        arr = np.array([[[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]])
        arr_norm = self.run_tool(arr, tool = norm)
        result = arr * self.run_tool(arr_norm, tool = maskForMaxCapsule)
        np.testing.assert_allclose(result, [[[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]], atol = 0.000001)
