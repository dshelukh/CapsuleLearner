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

class SimpleCodePrepare(RunElement):
    def run(self, config, randoms, *args, **kwargs):
        return randoms, 0
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
    def __init__(self, activation = tf.nn.relu, batch_norm = BatchNormElement(), regularizer = None):
        self.activation = activation
        self.batch_norm = batch_norm
        self.regularizer = regularizer

    def get_processed_data(self, data, conv_config, training = True):
        return tf.layers.conv2d(data, *(conv_config.get_conv_data()), name = 'conv', kernel_initializer = tf.variance_scaling_initializer(scale = 2.0, mode = 'fan_in'), kernel_regularizer = self.regularizer)

    def run(self, data, conv_config, name, training = True, dropout = 0.5, **kwargs):
        with tf.variable_scope(name):
            output = self.get_processed_data(data, conv_config, training)
            output = self.batch_norm.run(output, training, 'bn', **kwargs)
            if self.activation is not None:
                # TODO: add parameters
                output = self.activation(output)
            if (dropout > 0):
                output = tf.layers.dropout(output, dropout, training = training)
        return output

class DeconvBlockElement(ConvBlockElement):
    def get_processed_data(self, data, conv_config, training = True):
        return tf.layers.conv2d_transpose(data, *(conv_config.get_conv_data()), name = 'deconv', kernel_regularizer = self.regularizer)

class UpsamplingBlockElement(ConvBlockElement):
    def get_processed_data(self, data, conv_config, training = True):
        # data is supposed to be in NHWC format
        h, w = data.shape[1:3]
        strides = conv_config.stride

        if not isinstance(strides, tuple):
            strides = (strides, strides)
        
        resized = tf.image.resize_images(data, [h * strides[0], w * strides[1]], method = tf.image.ResizeMethod.BILINEAR)
        return tf.layers.conv2d(resized, conv_config.num_features, conv_config.kernel, padding = conv_config.padding, name = 'upsampl', kernel_regularizer = self.regularizer)

def get_data_and_world_info(inputs, world_size = 16, activation = tf.tanh, regularizer = None):
    if not isinstance(inputs, tuple):
        inputs = (inputs, None)

    if not (inputs[1] is None):
        return inputs
    else:
        return (inputs[0], tf.layers.dense(tf.layers.flatten(inputs[0]), world_size, activation = activation, kernel_regularizer = regularizer))

class DenseElement(RunElement):
    def __init__(self, regularizer = None):
        self.regularizer = regularizer
    def run(self, inputs, num_outputs, activation = None, name = 'dense', *args, **kwargs):
        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            result = tf.layers.dense(inputs, num_outputs, activation = activation, kernel_regularizer = self.regularizer)
        return result

class WeirdDenseElement(RunElement):
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
            print('Weird dense', input_data.get_shape(), dense_weights.get_shape())
            output = tf.squeeze(tf.matmul(input_data, dense_weights), [1]) + dense_bias
            print('Weird dense', output.get_shape())
            if activation:
                output = activation(output)
        return output

class WeirdConvBlockElement_(ConvBlockElement):
    def __init__(self, activation = tf.nn.relu, batch_norm = BatchNormElement(), world_size = 4, regularizer = None):
        self.world_size = world_size
        super(WeirdConvBlockElement_, self).__init__(activation, batch_norm, regularizer)

    def get_processed_data(self, inputs, conv_config, training = True):
        #internal = tf.layers.dense(tf.layers.flatten(data), self.world_size, activation = tf.tanh)
        data, world_info = get_data_and_world_info(inputs, self.world_size, None, self.regularizer)#tf.layers.dense(tf.layers.flatten(data), self.world_size, activation = tf.tanh)
        #world_info = WeirdDenseElement().run(data, self.world_size, self.world_size)
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

class WeirdConvBlockElement(ConvBlockElement):
    def __init__(self, activation = tf.nn.relu, batch_norm = BatchNormElement(), world_size = 4, regularizer = None):
        self.world_size = world_size
        super(WeirdConvBlockElement, self).__init__(activation, batch_norm, regularizer)

    def get_processed_data(self, inputs, conv_config, training = True):
        #internal = tf.layers.dense(tf.layers.flatten(data), self.world_size, activation = tf.tanh)
        data, world_info = get_data_and_world_info(inputs, self.world_size, None, self.regularizer)#tf.layers.dense(tf.layers.flatten(data), self.world_size, activation = tf.tanh)
        #world_info = WeirdDenseElement().run(data, self.world_size, self.world_size)
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

class WeirdConvBlockElement2(ConvBlockElement):
    def __init__(self, activation = tf.nn.relu, batch_norm = BatchNormElement(), world_size = 4):
        self.world_size = world_size
        super(WeirdConvBlockElement2, self).__init__(activation, batch_norm)

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

class WeirdConvBlockElement3(ConvBlockElement):
    def __init__(self, activation = tf.nn.relu, batch_norm = BatchNormElement(), world_size = 16, param_size = 4):
        self.world_size = world_size
        self.param_size = param_size
        super(WeirdConvBlockElement3, self).__init__(activation, batch_norm)

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

class WeirdConvBlockElement4(ConvBlockElement):
    def __init__(self, activation = tf.nn.relu, batch_norm = BatchNormElement(), world_size = 16, dense_param_size = 4, conv_param_size = 8):
        self.world_size = world_size
        self.dense_param_size = dense_param_size
        self.conv_param_size = conv_param_size
        super(WeirdConvBlockElement4, self).__init__(activation, batch_norm)

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

class WeirdIdealConvBlockElement(ConvBlockElement):
    def __init__(self, activation = tf.nn.relu, batch_norm = BatchNormElement(), world_size = 16):
        self.world_size = world_size
        super().__init__(activation, batch_norm)

    def get_processed_data(self, inputs, conv_config, training = True):
        #internal = tf.layers.dense(tf.layers.flatten(data), self.world_size, activation = tf.tanh)
        data, world_info = get_data_and_world_info(inputs, self.world_size, tf.tanh)#tf.layers.dense(tf.layers.flatten(data), self.world_size, activation = tf.tanh)
        #world_info = WeirdDenseElement().run(data, self.world_size, self.world_size)
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

class CleverDenseElement(RunElement):
    def __init__(self, num_outputs, dropin = 0.05, threshold = 0.8, decay = 0.97, regularizer = None):
        self.regularizer = regularizer
        self.num_outputs = num_outputs
        self.threshold = threshold
        self.dropin = dropin
        self.decay = decay

    def training_step(self, inputs, outputs):
        #in training do dropin first
        randoms = tf.random_uniform(tf.shape(outputs))
        dropins_mask = tf.less(randoms, self.dropin)
        #dropins_mask = tf.Print(dropins_mask, [tf.reduce_sum(tf.cast(dropins_mask, tf.float32))], message = 'Dropin cells:')
        outputs = outputs + self.threshold * tf.cast(dropins_mask, tf.float32)

        # actual training
        out_transposed = tf.transpose(outputs)
        activated = tf.greater(out_transposed, 0.0)
        not_activated = tf.logical_not(activated)
        activation_matrix = tf.cast(activated, tf.float32)
        # combined way
        #activation_matrix = tf.cast(activated, tf.float32) - tf.cast(not_activated, tf.float32)
        #correlations = tf.transpose(tf.matmul(activation_matrix, inputs)) / tf.cast(tf.shape(inputs)[0], tf.float32)
        inputs_mask = tf.greater(inputs, 0.0)
        inputs_mask_neg = tf.logical_not(inputs_mask)
        inputs_activation = tf.cast(inputs_mask, tf.float32) - tf.cast(inputs_mask_neg, tf.float32)
        correlations = tf.transpose(tf.matmul(activation_matrix, inputs_activation)) / tf.reduce_sum(activation_matrix, axis = 1)

        # update weights
        update = tf.assign(self.weights, self.weights * self.decay + (1.0 - self.decay) * correlations)
        update = tf.Print(update, [correlations[0]], message = 'First input correlations:')
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update)
        return outputs

    def build(self, input, dropout = 0.0, training = True, name = 'clever_dense', regularizer = None):
        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            inputs = tf.layers.flatten(input)
            print('Clever dense inputs shape flattened:', inputs.shape)
            self.weights = tf.get_variable('weights', [inputs.shape[1], self.num_outputs], trainable = False)
            # outputs as in normal dense without bias
            outputs = tf.matmul(inputs, self.weights)
            # remove small activations
            masked = tf.greater(outputs, self.threshold)
            print('Clever dense mask shape:', masked.shape)
            masked = tf.Print(masked, [tf.shape(outputs)[1], tf.shape(outputs)[0]], message = 'Total cells and number of images:')
            masked = tf.Print(masked, [tf.reduce_sum(tf.cast(masked, tf.float32))], message = 'Activated cells:')
            outputs = outputs * tf.cast(masked, tf.float32)
            print('Clever dense outputs shape after mask:', outputs.shape)
            result = tf.cond(training, lambda: self.training_step(inputs, outputs), lambda: outputs)
            print('Clever dense result shape after cond:', result.shape)
            result = tf.maximum(tf.minimum(result, 1.0), 0.0)
            result = tf.Print(result, [self.weights[0]], message = 'First input cell weights:')
        return result

class ConvData():
    def __init__(self, num_features, kernel, stride, padding = 'same', activation = None, element = ConvBlockElement):
        self.num_features, self.kernel, self.stride = num_features, kernel, stride
        self.padding = padding
        self.activation = activation
        self.element = element

    def get_conv_data(self):
        return self.num_features, self.kernel, self.stride, self.padding

class ConvLayout():
    def __init__(self, convs, isParallel = False, batch_norm = EmptyElement()):
        self.convs = convs
        self.isParallel = isParallel
        self.batch_norm = batch_norm

    def build(self, input, dropout = 0.0, training = True, name = 'conv', regularizer = None):
        cur = input
        retVal = []
        pre_vals = []
        pre_bools = []
        out_bools = tf.range(tf.shape(input)[0])

        for i, c in enumerate(self.convs):
            conv_name = name + ('_%d' % i)#name + ('_%d' % i) #name + str(i)
            if isinstance(c, ConvData):
                output = c.element(activation = c.activation, batch_norm = self.batch_norm, regularizer = regularizer).run(cur, c, dropout = dropout, training = training, name = conv_name)
            else:
                output = c.build(cur, dropout, training, conv_name, regularizer)
                
                if isinstance(c, PreliminaryResultElement):
                    if self.isParallel:
                        raise ValueError('Can\'t take preliminary results in forked execution')
                    resolved = output[0]
                    pre_vals.append(output[0])
                    pre_bools.append(tf.boolean_mask(out_bools, output[1]))
                    out_bools = tf.boolean_mask(out_bools, output[2])
                    output = output[3]
                    #output = tf.Print(output, [resolved, out_bools], message='preliminary', summarize = 10)
                
            if self.isParallel:
                retVal.append(output)
            else:
                cur = output

        if (len(pre_vals) > 0):
            print('pre_vals not none!')
            pre_vals.append(cur)
            pre_bools.append(out_bools)
            cur = tf.gather(tf.concat(pre_vals, axis = 0), tf.stop_gradient(tf.contrib.framework.argsort(tf.concat(pre_bools, axis = 0))))

        #cur = tf.Print(cur, [cur], message='result!', summarize = 20)

        return tf.concat(retVal, axis = -1) if self.isParallel else cur

class ResidualBlock():
    def __init__(self, convs, input_conv = None):
        self.convs = convs
        self.input_conv = input_conv

    def build(self, input, dropout = 0.0, training = True, name = 'resconv', regularizer = None):
        res = self.convs.build(input, dropout, training, name, regularizer)
        orig = input
        if (self.input_conv is not None):
            conv_for_input = self.input_conv.element(activation = self.input_conv.activation, batch_norm = EmptyElement(), regularizer = regularizer)
            orig = conv_for_input.run(input, self.input_conv, dropout = dropout, training = training, name = 'input_' + name)
        return res + orig

class ReshapeBlock():
    def __init__(self, size):
        self.size = size

    def build(self, input, dropout = 0.0, training = True, name = 'resconv', regularizer = None):
        return tf.reshape(input, [-1, *self.size])

class DropoutBlock():
    def __init__(self, dropout):
        self.dropout = dropout

    def build(self, input, dropout = 0.0, training = True, name = 'dropout', regularizer = None):
        return tf.layers.dropout(input, self.dropout, training = training, name = name)

class FinalizingElement():
    def __init__(self, num_outputs, use_GAP = True, dense_element = DenseElement):
        self.num_outputs = num_outputs
        self.use_GAP = use_GAP
        self.dense_element = dense_element

    def build(self, data, dropout = 0.0, training = True, name = 'finish', regularizer = None):
        if (self.use_GAP):
            data = tf.reduce_mean(data, axis = [1, 2])
        else:
            data = tf.layers.flatten(data)

        return self.dense_element(regularizer).run(data, self.num_outputs, name = name)

def get_top_k_mask(data, k):
    topk, indicies = tf.nn.top_k(data, k)
    mask = tf.reduce_sum(tf.one_hot(indicies, tf.shape(data)[0]), axis = -2)
    #mask = tf.Print(mask, [data, indicies, mask], summarize = 128)
    return tf.cast(mask, tf.bool)

class PreliminaryResultElement():
    def __init__(self, num_outputs, percent_ready, use_GAP = True, dense_element = DenseElement, train_bounds = (0.05, 0.6)):
        self.finalizer = FinalizingElement(num_outputs, use_GAP, dense_element)
        self.percent_ready = percent_ready
        self.min, self.max = train_bounds

    def training_prepare(self, bools, pre_max):
        passed = tf.reduce_sum(tf.cast(bools, tf.int32))
        total = tf.cast(tf.shape(bools)[0], tf.float32)
        expected = tf.cast(total * tf.constant(self.percent_ready), tf.int32) 
        min_num = tf.cast(total * tf.constant(self.min), tf.int32)
        max_num = tf.cast(total * tf.constant(self.max), tf.int32)
        new_bools = tf.cond(tf.less(passed, min_num), lambda: get_top_k_mask(pre_max, min_num), lambda: tf.cond(tf.greater(passed, max_num), lambda: get_top_k_mask(pre_max, max_num), lambda: bools))
        decay = 0.95
        expected_val = tf.nn.top_k(pre_max, expected)[0][-1]
        new_threshold = tf.assign(self.ready_threshold, self.ready_threshold * decay + (1.0 - decay) * expected_val)
        with tf.control_dependencies([new_threshold]):
            #new_bools = tf.Print(new_bools, [self.ready_threshold, expected_val])
            return new_bools

    def build(self, data, dropout = 0.0, training = True, name = 'preliminary', regularizer = None):
        pre_logits = self.finalizer.build(data, dropout, training, name, regularizer)
        #pre_logits = tf.check_numerics(pre_logits, 'Logits numerics check failed in' + name)
        pre_percent = tf.nn.softmax(pre_logits)
        #pre_percent = tf.check_numerics(pre_percent, 'Softmax numerics check failed in' + name)
        pre_max = tf.reduce_max(pre_percent, axis = -1)

        self.ready_threshold = tf.get_variable(name + '/threshold', initializer = tf.ones([1]), trainable = False)

        bools = tf.greater(pre_max, self.ready_threshold)
        bools = tf.cond(training, lambda: self.training_prepare(bools, pre_max), lambda: bools)
        bools = tf.stop_gradient(bools)
        not_bools = tf.logical_not(bools)
        ready = tf.boolean_mask(pre_logits, bools)
        need_continue = tf.boolean_mask(data, not_bools)
        return ready, bools, not_bools, need_continue

class BatchNormBlock():
    def __init__(self, element = BatchNormElement):
        self.element = element

    def build(self, input, dropout = 0.0, training = True, name = 'bn', regularizer = None):
        return self.element().run(input, training = training, name = name)

class ActivationBlock():
    def __init__(self, fn):
        self.fn = fn

    def build(self, input, dropout = 0.0, training = True, name = 'activation', regularizer = None):
        return self.fn(input)

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
        x = tf.reshape(lstm_output, [-1, lstm_output.shape[2]])

        # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
        # of rows of logit outputs, one for each step and sequence
        logits = tf.layers.dense(x, config.out_size) #config.out_size)

        # Back to shape with batch size
        logits = tf.reshape(logits, [-1, tf.shape(lstm_output)[1], config.out_size])

        return logits

# Convert images to code
class CapsEncoder(RunElement):
    def __init__(self, convs, batch_norm = BatchNormElement()):
        self.batch_norm = batch_norm
        self.convs = convs

    def run(self, config, inputs, reuse = False, training = True):

        with(tf.variable_scope('discriminator', reuse=reuse)):
            #cur = inputs
            cur = self.convs.build(inputs, 0.0, training)
            #for i, conv_info in enumerate(config.conv_d):
            #    cur = WeirdConvBlockElement(activation = conv_info.activation, batch_norm = self.batch_norm).run(cur, conv_info, training = training, name = 'conv%d' % i)

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

    def run(self, config, inputs, reuse = False, training = True):
        # Build the LSTM cell
        keep_prob = tf.cond(training, lambda: 1.0 - config.dropout, lambda: 1.0)

        cell1, self.initial_state1 = self.build_lstm(config.lstm1_size, config.lstm1_layers, tf.shape(inputs)[0], keep_prob)
        cell2, self.initial_state2 = self.build_lstm(config.lstm2_size, config.lstm2_layers, tf.shape(inputs)[0], keep_prob)
        ### Run the data through the RNN layers

        cur = tf.expand_dims(inputs, axis = -1)
        #for conv in config.convs:
        #    cur = tf.layers.conv2d(cur, *conv.get_conv_data(), activation = conv.activation)
        #cur = tf.reshape(cur, [-1, cur.shape[1], cur.shape[2] * cur.shape[3]])

        # Run each sequence step through the RNN with tf.nn.dynamic_rnn 
        outputs, state = tf.nn.dynamic_rnn(cell1, cur, initial_state=self.initial_state1, scope='lstm1')
        new_features = tf.layers.dense(outputs, config.dense_size, activation = tf.tanh) #WeirdDenseElement().run(outputs, config.dense_size, activation = tf.tanh, training = training, name = 'denseIn') #
        added_features = tf.concat((new_features, outputs), axis = 2)

        outputs2, state = tf.nn.dynamic_rnn(cell2, added_features, initial_state=self.initial_state2, scope='lstm2')
        self.final_state = state

        #cur = tf.expand_dims(outputs2, axis = -2)
        #for i, conv_info in enumerate(config.convs):
        #    deconv = UpsamplingBlockElement(activation = conv_info.activation, batch_norm = EmptyElement())
        #    cur = deconv.run(cur, conv_info, training = training, dropout = 0.0, name = 'deconv%d' % i)
        #cur = tf.reshape(cur, [-1, cur.shape[1], cur.shape[2] * cur.shape[3]])
        cur = outputs2[:, -1, :]
        print('LSTM output', cur.shape)
        return cur

class ConvEncoder(RunElement):
    def __init__(self, convs, dense_element = DenseElement):
        self.convs = convs
        self.dense_element = dense_element

    def run(self, config, inputs, reuse = False, training = True):

        with(tf.variable_scope('encoder', reuse=reuse)):
            regularizer = tf.contrib.layers.l2_regularizer(scale = config.weight_decay)
            cur = self.convs.build(inputs, 0.0, training, regularizer = regularizer)

            #cur = tf.layers.flatten(cur)#tf.reduce_mean(cur, axis = [1, 2])#tf.layers.flatten(cur)
            #cur = self.dense_element().run(cur, config.num_outputs)
            return [cur]
            '''
            output = None
            world_size = 16
            world_info = None
            coef = 1.0
            num_outputs = config.num_outputs
            retVal = []
            for num, i in enumerate(config.weirds):
                size, convs = i

                with(tf.variable_scope('convLayout%d' % num)):
                    resized = tf.image.resize_images(inputs, [size, size], method = tf.image.ResizeMethod.BILINEAR)
                    result = convs.build(resized, 0.5 * size / 32.0, training)
                    result = tf.layers.flatten(result)
                    result = WeirdDenseElement().run(result, world_size + num_outputs, world_size = world_size)#tf.layers.dense(cur, config.num_outputs)
                #conv = ConvData(8, 3, 2, activation = tf.tanh)
                #result =  WeirdConvBlockElement(world_size = world_size).run((resized, None), conv, training = training, dropout = 0.5 * i / 32.0, name = 'weird%d' % num)
                #result =  WeirdDenseElement().run((resized, world_info), world_size + num_outputs, world_size = world_size)
                #result = BatchNormElement().run(result, training = training, name = 'bn%d' % i)
                #result = tf.tanh(result)
                #result = tf.layers.dropout(result, 0.5, training = training)
                #result =  WeirdDenseElement().run((result, None), world_size + num_outputs, world_size = world_size, activation = tf.tanh)
                if output is None:
                    output = result[:, :num_outputs]
                else:
                    output += coef * result[:, :num_outputs]
                    coef = coef * 1.0
                world_info = result[:, num_outputs:]
                retVal.append(output)
            print('Output size:', output.shape)
            return retVal
            '''

class DenseEncoder(RunElement):
    def __init__(self, dense, base_name = 'encoder'):
        self.dense = dense
        self.base_name = base_name

    def run(self, config, inputs, reuse = False, training = True):
        element = config.dense_element
        cur = inputs
        with(tf.variable_scope(self.base_name, reuse=reuse)):
            regularizer = tf.contrib.layers.l2_regularizer(scale = config.weight_decay)
            for i, dense in enumerate(self.dense):
                #if i != 0:
                #    cur = BatchNormElement().run(cur, training, 'bn%d' % i)
                #    cur = DropoutBlock(0.3).build(cur, 0.5, training, 'dropout%d' % i)
                cur = element(regularizer).run(cur, *dense, training = training, name = 'dense%d' % i)
        return cur



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

class DensePredict():
    def run(self, config, codes):
        return tf.nn.softmax(codes)

class DenseCalcAndPredict():
    def run(self, config, codes):
        return tf.layers.dense(codes, config.num_outputs, name = 'discriminator_result', activation = None)



