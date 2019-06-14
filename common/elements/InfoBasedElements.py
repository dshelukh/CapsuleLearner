'''
@author: Dmitry
'''

import tensorflow as tf
from common.elements.Elements import *


def get_data_and_world_info(inputs, world_size = 16, activation = tf.tanh, regularizer = None):
    if not isinstance(inputs, tuple):
        inputs = (inputs, None)

    if not (inputs[1] is None):
        return inputs
    else:
        return (inputs[0], tf.layers.dense(tf.layers.flatten(inputs[0]), world_size, activation = activation, kernel_regularizer = regularizer))

class InfoBasedDenseElement(RunElement):
    def __init__(self, regularizer = None):
        self.regularizer = regularizer
    def run(self, inputs, num_outputs, activation = None, world_size = 16, training = False, name = 'weird_dense'):
        with tf.variable_scope(name):
            data, world_info = get_data_and_world_info(inputs, world_size, tf.tanh, self.regularizer)#tf.layers.dense(flattened, world_size, activation = tf.tanh)
            world_info = BatchNormElement().run(world_info, training, 'world_norm')
            #world_info = tf.layers.dropout(world_info, 0.1, training = training, name = 'world_dropout')
            flattened = tf.layers.flatten(data)
            input_size = flattened.get_shape()[-1]
            dense_weights = tf.layers.dense(world_info, input_size * num_outputs, kernel_regularizer = self.regularizer)
            dense_weights = tf.reshape(dense_weights, [-1, input_size, num_outputs])
            dense_bias = tf.layers.dense(world_info, num_outputs, kernel_regularizer = self.regularizer)
            input_data = tf.expand_dims(flattened, axis = 1)
            print('InfoBased dense', input_data.get_shape(), dense_weights.get_shape())
            output = tf.squeeze(tf.matmul(input_data, dense_weights), [1]) + dense_bias
            print('InfoBased dense', output.get_shape())
            if activation:
                output = activation(output)
        return output

class InfoBasedConvBlockElement_(ConvBlockElement):
    def __init__(self, activation = tf.nn.relu, batch_norm = BatchNormElement(), world_size = 4, regularizer = None):
        self.world_size = world_size
        super(InfoBasedConvBlockElement_, self).__init__(activation, batch_norm, regularizer)

    def get_processed_data(self, inputs, conv_config, training = True):
        #internal = tf.layers.dense(tf.layers.flatten(data), self.world_size, activation = tf.tanh)
        data, world_info = get_data_and_world_info(inputs, self.world_size, None, self.regularizer)#tf.layers.dense(tf.layers.flatten(data), self.world_size, activation = tf.tanh)
        #world_info = InfoBasedDenseElement().run(data, self.world_size, self.world_size)
        world_info = self.batch_norm.run(world_info, training, 'world_norm')
        world_info = tf.sigmoid(world_info)
        world_info = tf.layers.dropout(world_info, 0.33, training = training, name = 'world_dropout')
        world_info = tf.Print(world_info, [world_info[0], world_info[1]], summarize = 4)
        n_features, kernel, stride, padding = conv_config.get_conv_data()
        kw, kh = kernel if isinstance(kernel, tuple) else (kernel, kernel)
        sw, sh = stride if isinstance(stride, tuple) else (stride, stride)
        num_channels = data.get_shape().as_list()[-1]

        activation = None#tf.tanh
        conv_kernels = tf.layers.dense(world_info, n_features * kw * kh * num_channels, activation = activation, kernel_regularizer = self.regularizer)
        conv_kernels = tf.reshape(conv_kernels, [-1, kw, kh, num_channels, n_features])
        conv_biases = tf.layers.dense(world_info, n_features, activation = activation, kernel_regularizer = self.regularizer)

        #data = tf.Print(data, [conv_kernels[0][:kw][:kh][0][0], conv_kernels[1][:kw][:kh][0][0], conv_biases[0], conv_biases[1]], summarize = 9)
        convolution = tf.map_fn(lambda x: tf.nn.convolution([x[0]], x[1], padding.upper(), [sw, sh]) + x[2], (data, conv_kernels, conv_biases), dtype=tf.float32, name = 'conv', parallel_iterations = 1000)
        #convolution = tf.Print(convolution, [tf.shape(convolution)], summarize = 10)
        convolution = tf.squeeze(convolution, [1])
        print(convolution.get_shape())
        #convolution = tf.Print(convolution, [convolution])
        return convolution

class InfoBasedConvBlockElement(ConvBlockElement):
    def __init__(self, activation = tf.nn.relu, batch_norm = BatchNormElement(), world_size = 4, regularizer = None):
        self.world_size = world_size
        super(InfoBasedConvBlockElement, self).__init__(activation, batch_norm, regularizer)

    def get_processed_data(self, inputs, conv_config, training = True):
        #internal = tf.layers.dense(tf.layers.flatten(data), self.world_size, activation = tf.tanh)
        data, world_info = get_data_and_world_info(inputs, self.world_size, None, self.regularizer)#tf.layers.dense(tf.layers.flatten(data), self.world_size, activation = tf.tanh)
        #world_info = InfoBasedDenseElement().run(data, self.world_size, self.world_size)
        world_info = self.batch_norm.run(world_info, training, 'world_norm')
        world_info = tf.sigmoid(world_info)
        #world_info = tf.layers.dropout(world_info, 0.2, training = training, name = 'world_dropout')
        n_features, kernel, stride, padding = conv_config.get_conv_data()
        kw, kh = kernel if isinstance(kernel, tuple) else (kernel, kernel)
        sw, sh = stride if isinstance(stride, tuple) else (stride, stride)
        num_channels = data.get_shape().as_list()[-1]

        activation = None#tf.tanh
        conv_kernels = tf.layers.dense(world_info, n_features * kw * kh * num_channels, activation = activation, kernel_regularizer = self.regularizer)
        conv_kernels = tf.reshape(conv_kernels, [-1, kw, kh, num_channels, n_features])
        conv_biases = tf.layers.dense(world_info, n_features, activation = activation, kernel_regularizer = self.regularizer)

        convolution = tf.map_fn(lambda x: tf.nn.convolution([x[0]], x[1], padding.upper(), [sw, sh]) + x[2], (data, conv_kernels, conv_biases), dtype=tf.float32, name = 'conv', parallel_iterations = 1000)
        #convolution = tf.Print(convolution, [tf.shape(convolution)], summarize = 10)
        convolution = tf.squeeze(convolution, [1])
        print(convolution.get_shape())
        #convolution = tf.Print(convolution, [convolution])
        return convolution

class InfoBasedConvBlockElement2(ConvBlockElement):
    def __init__(self, activation = tf.nn.relu, batch_norm = BatchNormElement(), world_size = 4):
        self.world_size = world_size
        super(InfoBasedConvBlockElement2, self).__init__(activation, batch_norm)

    def get_processed_data(self, inputs, conv_config, training = True):
        n_features, kernel, stride, padding = conv_config.get_conv_data()
        data, world_info = get_data_and_world_info(inputs, n_features * self.world_size, tf.tanh)
        world_info = tf.reshape(world_info, [-1, n_features, self.world_size])

        world_info = self.batch_norm.run(world_info, training, 'world_norm')
        #world_info = tf.layers.dropout(world_info, 0.33, training = training, name = 'world_dropout')

        kw, kh = kernel if isinstance(kernel, tuple) else (kernel, kernel)
        sw, sh = stride if isinstance(stride, tuple) else (stride, stride)
        num_channels = data.get_shape().as_list()[-1]

        activation = None #tf.tanh
        conv_kernels = tf.layers.dense(world_info, kw * kh * num_channels, activation = activation, kernel_regularizer = lambda x: tf.reduce_sum(tf.sqrt(tf.abs(x))))
        conv_kernels = tf.transpose(conv_kernels, [0, 2, 1])
        conv_kernels = tf.reshape(conv_kernels, [-1, kw, kh, num_channels, n_features])
        conv_biases = tf.layers.dense(world_info, 1, activation = activation)
        conv_biases = tf.squeeze(conv_biases, 2)

        #data = tf.Print(data, [tf.shape(data), tf.shape(conv_kernels), tf.shape(conv_biases)], summarize = 10)
        convolution = tf.map_fn(lambda x: tf.nn.convolution([x[0]], x[1], padding.upper(), [sw, sh]) + x[2], (data, conv_kernels, conv_biases), dtype=tf.float32, name = 'conv', parallel_iterations = 1000)
        #convolution = tf.Print(convolution, [tf.shape(convolution)], summarize = 10)
        convolution = tf.squeeze(convolution, [1])
        print(convolution.get_shape())
        #convolution = tf.Print(convolution, [convolution])
        return convolution

class InfoBasedConvBlockElement3(ConvBlockElement):
    def __init__(self, activation = tf.nn.relu, batch_norm = BatchNormElement(), world_size = 16, param_size = 4):
        self.world_size = world_size
        self.param_size = param_size
        super(InfoBasedConvBlockElement3, self).__init__(activation, batch_norm)

    def get_processed_data(self, inputs, conv_config, training = True):
        n_features, kernel, stride, padding = conv_config.get_conv_data()
        data, world_info = get_data_and_world_info(inputs, self.world_size, tf.sigmoid)
        world_info = tf.layers.dense(world_info, n_features * self.param_size, tf.sigmoid, kernel_regularizer = lambda x: tf.reduce_sum(tf.sqrt(tf.abs(x))))
        world_info = tf.reshape(world_info, [-1, n_features, self.param_size])

        world_info = self.batch_norm.run(world_info, training, 'world_norm')
        #world_info = tf.layers.dropout(world_info, 0.33, training = training, name = 'world_dropout')

        kw, kh = kernel if isinstance(kernel, tuple) else (kernel, kernel)
        sw, sh = stride if isinstance(stride, tuple) else (stride, stride)
        num_channels = data.get_shape().as_list()[-1]

        activation = None #tf.tanh
        conv_kernels = tf.layers.dense(world_info, kw * kh * num_channels, activation = activation)
        conv_kernels = tf.transpose(conv_kernels, [0, 2, 1])
        conv_kernels = tf.reshape(conv_kernels, [-1, kw, kh, num_channels, n_features])
        conv_biases = tf.layers.dense(world_info, 1, activation = activation)
        conv_biases = tf.squeeze(conv_biases, 2)

        #data = tf.Print(data, [tf.shape(data), tf.shape(conv_kernels), tf.shape(conv_biases)], summarize = 10)
        convolution = tf.map_fn(lambda x: tf.nn.convolution([x[0]], x[1], padding.upper(), [sw, sh]) + x[2], (data, conv_kernels, conv_biases), dtype=tf.float32, name = 'conv', parallel_iterations = 1000)
        #convolution = tf.Print(convolution, [tf.shape(convolution)], summarize = 10)
        convolution = tf.squeeze(convolution, [1])
        print(convolution.get_shape())
        #convolution = tf.Print(convolution, [convolution])
        return convolution

class InfoBasedConvBlockElement4(ConvBlockElement):
    def __init__(self, activation = tf.nn.relu, batch_norm = BatchNormElement(), world_size = 16, dense_param_size = 4, conv_param_size = 8):
        self.world_size = world_size
        self.dense_param_size = dense_param_size
        self.conv_param_size = conv_param_size
        super(InfoBasedConvBlockElement4, self).__init__(activation, batch_norm)

    def get_processed_data(self, inputs, conv_config, training = True):
        n_features, kernel, stride, padding = conv_config.get_conv_data()
        data, world_info = get_data_and_world_info(inputs, self.world_size, tf.sigmoid)
        choice_regularizer = lambda x: tf.reduce_sum(tf.sqrt(tf.abs(x)))

        world_info = tf.layers.dense(world_info, n_features * self.dense_param_size, kernel_regularizer = choice_regularizer)
        world_info = tf.reshape(world_info, [-1, n_features, self.dense_param_size])

        world_info = self.batch_norm.run(world_info, training, 'world_norm')

        kw, kh = kernel if isinstance(kernel, tuple) else (kernel, kernel)
        sw, sh = stride if isinstance(stride, tuple) else (stride, stride)
        h, w, num_channels = tuple(data.get_shape().as_list()[1:])

        preconv = tf.layers.conv2d(data, self.conv_param_size * n_features, 1, name = 'preconv', kernel_regularizer = choice_regularizer)
        #extract features for each conv
        preconv = tf.reshape(preconv, [-1, h, w, self.conv_param_size, n_features])
        #move kernelwise features up
        preconv = tf.transpose(preconv, [0, 4, 1, 2, 3])
        #combine batch and kernelwise features
        preconv = tf.reshape(preconv, [-1, h, w, self.conv_param_size])

        activation = None
        conv_kernels = tf.layers.dense(world_info, kw * kh * self.conv_param_size, activation = activation)
        conv_kernels = tf.reshape(conv_kernels, [-1, kw, kh, self.conv_param_size, 1])
        conv_biases = tf.layers.dense(world_info, 1, activation = activation)
        conv_biases = tf.reshape(conv_biases, [-1])

        convolution = tf.map_fn(lambda x: tf.nn.convolution([x[0]], x[1], padding.upper(), [sw, sh]) + x[2], (preconv, conv_kernels, conv_biases), dtype=tf.float32, name = 'conv', parallel_iterations = 10000)
        #convolution = tf.Print(convolution, [tf.shape(convolution)], summarize = 10)
        h1, w1 = tuple(convolution.get_shape().as_list()[2:4])
        convolution = tf.reshape(convolution, [-1, n_features, h1, w1])
        convolution = tf.transpose(convolution, [0, 2, 3, 1])
        print(convolution.get_shape())
        #convolution = tf.Print(convolution, [convolution])
        return convolution