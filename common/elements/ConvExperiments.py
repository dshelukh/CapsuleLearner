'''
@author: Dmitry
'''

import tensorflow as tf
from common.elements.Elements import *
from common.elements.InfoBasedElements import get_data_and_world_info


def ortho_regularizer(x):
    w, h, pred, next = x.shape
    kernel_base = tf.reshape(x, [w * h * pred, next])
    kernel_square = tf.expand_dims(tf.transpose(kernel_base, [1, 0]), -1) * kernel_base
    result = tf.reduce_sum(tf.abs(tf.reduce_sum(kernel_square, axis = -2)))
    #print('Ortho input', w * h * pred, next, 'square', kernel_square.shape)
    return result

class OrthoConvBlockElement(ConvBlockElement):
    def __init__(self, activation = tf.nn.relu, batch_norm = BatchNormElement(), regularizer = None):
        super(OrthoConvBlockElement, self).__init__(activation, batch_norm, ortho_regularizer)

def extract_patches_func(k, s, padding):
    return lambda x: tf.extract_image_patches([x], (1, k[0], k[1], 1), (1, s[0], s[1], 1), (1, 1, 1, 1), padding = padding.upper())

def ideal_conv(image, ideal, bias, extractor):
    patches = extractor(image)
    patches = tf.expand_dims(patches, axis = -1)
    return l2_squared_norm(patches - ideal, axis = -2) + bias

class InfoBasedIdealConvBlockElement(ConvBlockElement):
    def __init__(self, activation = tf.nn.relu, batch_norm = BatchNormElement(), world_size = 16):
        self.world_size = world_size
        super().__init__(activation, batch_norm)

    def get_processed_data(self, inputs, conv_config, training = True):
        #internal = tf.layers.dense(tf.layers.flatten(data), self.world_size, activation = tf.tanh)
        data, world_info = get_data_and_world_info(inputs, self.world_size, tf.tanh)#tf.layers.dense(tf.layers.flatten(data), self.world_size, activation = tf.tanh)
        #world_info = InfoBasedDenseElement().run(data, self.world_size, self.world_size)
        world_info = self.batch_norm.run(world_info, training, 'world_norm')
        world_info = tf.layers.dropout(world_info, 0.33, training = training, name = 'world_dropout')
        #world_info = BatchRenormElement().run(world_info, training, 'renorm')
        n_features, kernel, stride, padding = conv_config.get_conv_data()
        kw, kh = kernel if isinstance(kernel, tuple) else (kernel, kernel)
        sw, sh = stride if isinstance(stride, tuple) else (stride, stride)
        num_channels = data.get_shape().as_list()[-1]

        activation = None #tf.tanh
        conv_ideals = tf.layers.dense(world_info, n_features * kw * kh * num_channels, activation = activation)
        conv_ideals = tf.reshape(conv_ideals, [-1, 1, 1, kw * kh * num_channels, n_features])
        conv_biases = tf.layers.dense(world_info, n_features, activation = activation)
        conv_biases = tf.reshape(conv_biases, [-1, 1, 1, n_features])

        #patches = tf.extract_image_patches(inputs, (1, kw, kh, 1), (1, sw, sh, 1), (1, 1, 1, 1), padding = padding.upper())
        #patches = tf.expand_dims(patches, axis = -1)
        extractor = extract_patches_func((kw, kh), (sw, sh), padding)
        #Compared to straight calculation map_fn helps reduce space requirements (about 1.5-2 times)
        result = tf.map_fn(lambda x: ideal_conv(*x, extractor), (data, conv_ideals, conv_biases), dtype=tf.float32, name = 'conv', parallel_iterations = 1000)
        #convolution = tf.Print(convolution, [tf.shape(convolution)], summarize = 10)
        result = tf.squeeze(result, [1])
        #result = l2_squared_norm(patches - conv_ideals, axis = -2) + conv_biases
        return result

class IdealConvBlockElement(ConvBlockElement):
    def __init__(self, activation = tf.nn.relu, batch_norm = BatchNormElement(), world_size = 16):
        self.world_size = world_size
        super().__init__(activation, batch_norm)

    def get_processed_data(self, inputs, conv_config, training = True):
        n_features, kernel, stride, padding = conv_config.get_conv_data()
        kw, kh = kernel if isinstance(kernel, tuple) else (kernel, kernel)
        sw, sh = stride if isinstance(stride, tuple) else (stride, stride)
        num_channels = inputs.get_shape().as_list()[-1]

        conv_ideals = tf.get_variable('ideals', [1, 1, kw * kh * num_channels, n_features], initializer = tf.truncated_normal_initializer())
        conv_biases = tf.get_variable('ideals',[1, 1, n_features], initializer = tf.truncated_normal_initializer())

        patches = tf.extract_image_patches(inputs, (1, kw, kh, 1), (1, sw, sh, 1), (1, 1, 1, 1), padding = padding.upper())
        patches = tf.expand_dims(patches, axis = -1)
        result = l2_squared_norm(patches - conv_ideals, axis = -2) + conv_biases
        return result

class GroupConvBlockElement(ConvBlockElement):
    def __init__(self, activation = tf.nn.relu, batch_norm = BatchNormElement(), regularizer = None, ngroups = 16, element = OrthoConvBlockElement):
        super().__init__(activation, batch_norm, regularizer)
        self.ngroups = ngroups
        self.element = element

    def get_processed_data(self, inputs, conv_config, training = True):
        convs = []
        config = conv_config
        config.num_features = int(config.num_features / self.ngroups)
        config.element = self.element
        for i in range(self.ngroups):
            convs.append(config)
        return ConvLayout(convs, isParallel = True).build(inputs, 0.0, training, 'group_conv', self.regularizer)

class LongConvBlockElement(ConvBlockElement):
    def __init__(self, activation = tf.nn.relu, batch_norm = BatchNormElement(), regularizer = None):
        super().__init__(activation, batch_norm, regularizer)

    def get_processed_data(self, inputs, conv_config, training = True):
        config = conv_config
        n = config.num_features
        config.num_features = 1
        cur = inputs
        for i in range(n):
            tmp = ConvBlockElement(self.activation, self.batch_norm, self.regularizer).run(cur, config, training = training, name = 'longconv' + str(i))
            cur = tf.concat([cur, tmp], axis = -1)
        return cur
