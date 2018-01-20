'''
@author: Dmitry
'''
import tensorflow as tf
from CapsTools import *
from CapsuleLayer import *

class RunElement():
    def run(self, *args, **kwargs):
        pass

class EmptyElement(RunElement):
    def run(self, config, data, *args, **kwargs):
        return data

# Extract labels information from random code
class CapsCodePrepare(RunElement):
    def run(self, config, code, *args, **kwargs):
        return squash(code), tf.one_hot(tf.argmax(l1norm(code), axis = -1), config.num_outputs)

class FlatCodePrepare(RunElement):
    def run(self, config, randoms, *args, **kwargs):
        return randoms, randoms[:, :config.num_outputs]

# Batch Normalization Elements

# Simple wrapper over batch normalization layer
class BatchNormElement(RunElement):
    def run(self, data, training, name, **kwargs):
        return tf.layers.batch_normalization(data, training = training, name = name)

class VbnElement(RunElement):
    def run(self, data, training, name, ref_size = None):
        x = data
        ref_size = 0 if ref_size is None else ref_size
        x, bn_data = tf.split(x, [tf.shape(x)[0] - ref_size, ref_size])

        if ref_size != 0:
            bn_data = tf.layers.batch_normalization(bn_data, training = training, name = name)
            training = False
    
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            x = tf.layers.batch_normalization(x, training = training, name = name, reuse = tf.AUTO_REUSE)
    
        x = tf.concat((x, bn_data), 0)
        return x

class BatchRenormElement(RunElement):
    def run(self, data, training, name):
        return tf.layers.batch_normalization(data, training = training, name = name, renorm = True)

# Convolution and deconvolution blocks with batch normalization, activation and dropout
class ConvBlockElement(RunElement):
    def __init__(self, activation = tf.nn.relu, batch_norm = BatchNormElement()):
        self.activation = activation
        self.batch_norm = batch_norm

    def get_processed_data(self, data, conv_config):
        return tf.layers.conv2d(data, *(conv_config.get_conv_data()), name = 'conv')

    def run(self, data, conv_config, name, training = True, dropout = 0.5, **kwargs):
        with tf.variable_scope(name):
            output = self.get_processed_data(data, conv_config)
            output = self.batch_norm.run(output, training, 'bn', **kwargs)
            if self.activation is not None:
                # TODO: add parameters
                output = self.activation(output)
            if (dropout > 0):
                output = tf.layers.dropout(output, dropout, training = training)
        return output

class DeconvBlockElement(ConvBlockElement):
    def get_processed_data(self, data, conv_config):
        return tf.layers.conv2d_transpose(data, *(conv_config.get_conv_data()), name = 'deconv')

# Base class for VB users
class VbUserElement(RunElement):
    def __init__(self):
        self.batch_norm = VbnElement()
        self.ref_batch = None

    def set_ref_batch(self, ref_batch):
        self.ref_batch = tf.convert_to_tensor(ref_batch, dtype = tf.float32)

    def get_virtual_batch_data(self, data):
        ref_size = None
        if (self.ref_batch is not None):
            ref_size = tf.shape(self.ref_batch)[0]
            data = tf.concat((data, self.ref_batch), 0)
        return ref_size, data

    def return_from_virtual_batch(self, data, ref_size):
        if (ref_size is not None):
            data = data[:-ref_size]
            #data = tf.Print(data, [ref_size, tf.shape(data)], 'return data:', summarize = 5)
        return data

# Generate images from code
class CapsGenerator(VbUserElement):
    def __init__(self, batch_norm = VbnElement()):
        super(CapsGenerator, self).__init__()
        self.batch_norm = batch_norm
        self.deconv_block = DeconvBlockElement(activation = leaky_relu, batch_norm = self.batch_norm)

    def run(self, config, code, reuse = False, training = True):
        ref_size, code = self.get_virtual_batch_data(code)

        with(tf.variable_scope('generator', reuse=reuse)):
            # only one active capsule expected, hence r = 1
            caps_layer0 = CapsLayer(code, config.num_outputs, config.caps2_len, name = 'caps_layer_g_0', r = 1)
            caps1 = caps_layer0.do_dynamic_routing()

            caps_layer = CapsLayer(caps1, np.prod(config.capsg_size), config.capsg_len, name = 'caps_layer_g_1')
            caps2 = caps_layer.do_dynamic_routing()
            reshaped = tf.reshape(caps2, [-1, config.capsg_size[0], config.capsg_size[1], config.capsg_size[2] * config.capsg_len])
            reshaped = tf.verify_tensor_all_finite(reshaped, 'Not all values are finite befire VBN1')
            reshaped = self.batch_norm.run(reshaped, training = training, name = 'ref_batch_norm1', ref_size = ref_size)
            reshaped = tf.verify_tensor_all_finite(reshaped, 'Not all values are finite after VBN1')

            deconv1_info = config.deconv1_info
            gconv1 = self.deconv_block.run(reshaped, deconv1_info, training = training, name = 'deconv1', dropout = 0.0, ref_size = ref_size)

            deconv2_info = config.deconv2_info
            gconv2 = tf.layers.conv2d_transpose(gconv1, *(deconv2_info.get_conv_data()))
            return self.return_from_virtual_batch(gconv2, ref_size)

# Convert images to code
class CapsEncoder(RunElement):
    def run(self, config, inputs, reuse = False, training = True):

        with(tf.variable_scope('discriminator', reuse=reuse)):
            conv1_info = config.conv1_info
            conv1 = ConvBlockElement(activation = leaky_relu).run(inputs, conv1_info, training = training, name = 'conv1')

            conv2_info = config.conv2_info
            conv2 = ConvBlockElement(activation = None).run(conv1, conv2_info, training = training, name = 'conv2')

            caps1 = squash(reshapeToCapsules(tf.transpose(conv2, [0, 3, 1, 2]), config.caps1_len, axis = 1))
            caps_layer = CapsLayer(caps1, config.num_outputs, config.caps2_len)
            caps2 = caps_layer.do_dynamic_routing()
            return caps2

# Element to add minibatch data
class Minibatcher(RunElement):
    def run(self, config, x, name = 'discriminator_minibatch', reuse = False):
        batch, l = tf.shape(x)[0], x.get_shape().as_list()[1]
        with(tf.variable_scope(name, reuse=reuse)):
            W = tf.get_variable('minibatch_weights', [l, config.minibatch_num * config.minibatch_len], initializer = tf.truncated_normal_initializer())
            minibatch_data = tf.matmul(x, W)
            #TODO: find better way
            diff = tf.expand_dims(minibatch_data, -1) - tf.transpose(minibatch_data, [1, 0])
            diff = tf.reshape(diff, [batch, config.minibatch_num, config.minibatch_len, batch])
            diff_norm = l1norm(diff, axis = -2) # euclidean norm here results in nan in gradients (of generator's convolution...)
            # diff_norm has zeros on main diagonal, hence subtract one.
            addition = tf.reduce_sum(tf.exp(-diff_norm), axis = -1) - 1
        return tf.concat((x, addition), axis = -1)

# Convert encoded data to features
class CapsFeatureExtractor(RunElement):
    def run(self, config, codes):
        masked = tf.multiply(codes, maskForMaxCapsule(l1norm(codes)))
        return tf.contrib.layers.flatten(masked)

class RemoveLabelsFeatureExtractor(RunElement):
    def run(self, config, codes):
        return codes[:, config.num_outputs:]

# Fully connected layer with one output, used in distinguishing between real and generated images
class DenseFakeDetector(RunElement):
    def run(self, config, data, name = 'discriminator_add', reuse = False):
        with (tf.variable_scope(name, reuse = reuse)):
            fake_detector = tf.layers.dense(data, 1)
        return fake_detector

# Get prediction data from code
# Prediction is a capsule length
class CapsPredict(RunElement):
    def run(self, config, codes):
        return l1norm(codes)

# Prediction is contained in first num_outputs values
class FlatPredict():
    def run(self, config, codes):
        return codes[:, :config.num_outputs]




