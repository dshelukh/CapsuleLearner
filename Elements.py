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
    def run(self, data, *args, **kwargs):
        return data

#TODO: one empty element is enough
class EmptyElementConfig(RunElement):
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
    def run(self, data, training, name, **kwargs):
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

class UpsamplingBlockElement(ConvBlockElement):
    def get_processed_data(self, data, conv_config):
        # data is supposed to be in NHWC format
        h, w = data.shape[1:3]
        strides = conv_config.stride

        if not isinstance(strides, tuple):
            strides = (strides, strides)
        
        resized = tf.image.resize_images(data, [h * strides[0], w * strides[1]], method = tf.image.ResizeMethod.BILINEAR)
        return tf.layers.conv2d(resized, conv_config.num_features, conv_config.kernel, padding = conv_config.padding, name = 'upsampl')

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
        self.deconv_block_class = DeconvBlockElement

    def set_deconv_block_class(self, block_class):
        self.deconv_block_class = block_class

    def run(self, config, code, reuse = False, training = True):
        ref_size, code = self.get_virtual_batch_data(code)

        with(tf.variable_scope('generator', reuse=reuse)):
            # only one active capsule expected, hence r = 1
            caps_layer0 = CapsLayer(code, config.num_outputs, config.caps2_len, name = 'caps_layer_g_0', r = 1)
            caps1 = caps_layer0.do_dynamic_routing()

            caps_layer = CapsLayer(caps1, np.prod(config.capsg_size), config.capsg_len, name = 'caps_layer_g_1')
            caps2 = caps_layer.do_dynamic_routing()
            reshaped = tf.reshape(caps2, [-1, config.capsg_size[0], config.capsg_size[1], config.capsg_size[2] * config.capsg_len])
            reshaped = self.batch_norm.run(reshaped, training = training, name = 'ref_batch_norm1', ref_size = ref_size)

            cur = reshaped
            for i, conv_info in enumerate(config.deconv_g):
                use_batch_norm = self.batch_norm if i < len(config.deconv_g) - 1 else EmptyElement() # no batch norm on last layer
                deconv = self.deconv_block_class(activation = conv_info.activation, batch_norm = use_batch_norm)
                cur = deconv.run(cur, conv_info, training = training, dropout = 0.0, name = 'deconv%d' % i, ref_size = ref_size)

            return self.return_from_virtual_batch(cur, ref_size)

class LSTMGenerator(VbUserElement):
    #(code from Udacity course)
    def run(self, config, lstm_output, training = True):
        # Reshape output so it's a bunch of rows, one row for each step for each sequence.
        x = tf.reshape(lstm_output, [-1, config.lstm2_size])

        # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
        # of rows of logit outputs, one for each step and sequence
        logits = tf.layers.dense(x, config.out_size)

        # Back to shape with batch size
        logits = tf.reshape(logits, [-1, tf.shape(lstm_output)[1], config.out_size])

        return logits

# Convert images to code
class CapsEncoder(RunElement):
    def run(self, config, inputs, reuse = False, training = True):

        with(tf.variable_scope('discriminator', reuse=reuse)):
            cur = inputs
            for i, conv_info in enumerate(config.conv_d):
                cur = ConvBlockElement(activation = conv_info.activation).run(cur, conv_info, training = training, name = 'conv%d' % i)

            caps1 = squash(reshapeToCapsules(tf.transpose(cur, [0, 3, 1, 2]), config.caps1_len, axis = 1))
            caps_layer = CapsLayer(caps1, config.num_outputs, config.caps2_len)
            caps2 = caps_layer.do_dynamic_routing()
            return caps2

class LSTMEncoder(RunElement):
    ### Build the LSTM Cell (code from Udacity course)
    def build_cell(self, lstm_size, keep_prob):
        # Use a basic LSTM cell
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    
        # Add dropout to the cell outputs
        lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return lstm

    def build_lstm(self, lstm_size, num_layers, batch_size, keep_prob):

        # Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell([self.build_cell(lstm_size, keep_prob) for i in range(num_layers)])
        initial_state = cell.zero_state(batch_size, tf.float32)

        return cell, initial_state

    def run(self, config, inputs, training = True):
        # Build the LSTM cell
        keep_prob = tf.cond(training, lambda: 1.0 - config.dropout, lambda: 1.0)

        cell1, self.initial_state1 = self.build_lstm(config.lstm1_size, config.lstm1_layers, tf.shape(inputs)[0], keep_prob)
        cell2, self.initial_state2 = self.build_lstm(config.lstm2_size, config.lstm2_layers, tf.shape(inputs)[0], keep_prob)
        ### Run the data through the RNN layers

        cur = tf.expand_dims(inputs, axis = -1)
        for conv in config.convs:
            cur = tf.layers.conv2d(cur, *conv.get_conv_data(), activation = conv.activation)
        cur = tf.reshape(inputs, [-1, cur.shape[1], cur.shape[1] * cur.shape[2]])

        # Run each sequence step through the RNN with tf.nn.dynamic_rnn 
        outputs, state = tf.nn.dynamic_rnn(cell1, inputs, initial_state=self.initial_state1, scope='lstm1')
        added_features = tf.concat((tf.layers.dense(outputs, config.dense_size, activation = tf.tanh), outputs), axis = 2)

        outputs2, state = tf.nn.dynamic_rnn(cell2, added_features, initial_state=self.initial_state2, scope='lstm2')
        self.final_state = state
        return outputs2


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
        return tf.multiply(codes, maskForMaxCapsule(l1norm(codes)))

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




