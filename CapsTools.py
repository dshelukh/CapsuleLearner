'''
@author: Dmitry
'''
import tensorflow as tf
import numpy as np

# TODO: for NHWC only one reshape is needed!
# Turn all dimensions starting from axis in a one-dimensional array with ndim-dimensional elements
# example: reshaping 100x4x8x8 array from axis = 1 with ndim = 2 should result in 100x128x2 array (128 = 8*8*4/ndim)
def reshapeToCapsules(arr, ndim, axis = 1):
    shape = tf.shape(arr)
    total = np.prod(arr.shape.as_list()[axis:]) // ndim
    shape_size = shape.get_shape().as_list()[0]

    # may be we don't need to stick to some order? In this case no need to add the axis and do this reshape
    axis = axis if axis != -1 else shape_size - 1
    tmp_shape = tf.concat([shape[:axis], [shape[axis] // ndim, ndim], shape[axis + 1:]], 0)
    arr2 = tf.reshape(arr, tmp_shape)

    reorder = np.array(range(shape_size + 1)) # increased on previous step
    reorder[axis + 1:-1], reorder[-1] = reorder[axis + 2:], reorder[axis + 1]

    result_shape = tf.concat([tmp_shape[:axis], [total, ndim]], 0)
    return tf.reshape(tf.transpose(arr2, reorder), result_shape)

# basic squash along some axis
def squash(arr, axis = -1):
    squared_norm = tf.reduce_sum(tf.square(arr), axis, keep_dims=True)
    norm = tf.sqrt(squared_norm)
    coef = tf.divide(norm, tf.add(squared_norm, 1.0))
    return tf.multiply(arr, coef)

def norm(arr, axis = -1):
    return tf.sqrt(tf.reduce_sum(tf.square(arr), axis))

# converts array with norm to a mask for element with maximum norm
def maskForMaxCapsule(arr):
    num_outputs = arr.shape[-1]
    max_capsules = tf.argmax(arr, axis = -1)
    return tf.expand_dims(tf.one_hot(max_capsules, num_outputs), -1)
