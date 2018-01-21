'''
@author: Dmitry
'''

import tensorflow as tf

from Trainer import *
from Runners import *

class ConvData():
    def __init__(self, num_features, kernel, stride, padding = 'same'):
        self.num_features, self.kernel, self.stride = num_features, kernel, stride
        self.padding = padding

    def get_conv_data(self):
        return self.num_features, self.kernel, self.stride, self.padding

class SemiSupervisedConfig():
    def __init__(self, num_outputs = 10, num_d_conv = 3, num_g_conv = 3, alpha = 0.01, start_features = 16, conv_kernel = 3, conv_stride = 2,
                 code_size = 80, first_decode_shape = [4, 4, 16], orig_channels = 3, mu = 0.1):
        self.alpha = alpha
        self.d_convs = []
        self.g_convs = []

        n_features = start_features
        for i in range(num_d_conv):
            self.d_convs.insert(0, ConvData(n_features, conv_kernel, conv_stride))
            n_features *= 2

        n_features = start_features
        for i in range(num_g_conv):
            self.g_convs.append(ConvData(n_features, conv_kernel, conv_stride))
            n_features *= 2

        self.code_size = code_size
        self.num_outputs = num_outputs
        self.first_decode_shape = first_decode_shape
        self.orig_channels = orig_channels # derive from input?
        self.mu = mu

class SemiSupervisedNetwork():
    def __init__(self, config = SemiSupervisedConfig()):
        self.config = config

    def get_code(self, images, reuse = False, training = True):
        config = self.config

        cur = images
        with(tf.variable_scope('discriminator', reuse=reuse)):
            for i, conv in enumerate(config.d_convs):
                cur = tf.layers.conv2d(cur, conv.num_features, conv.kernel, conv.stride, padding = 'same', name = 'dconv%d' % i)
                cur = tf.layers.batch_normalization(cur, training = training, name = 'dbatch%d' % i)
                cur = leaky_relu(cur, config.alpha)
            cur = tf.layers.flatten(cur)
            labels, code = tf.layers.dense(cur, config.num_outputs), tf.layers.dense(cur, config.code_size, activation = tf.tanh)
        return labels, code

    def get_decoded(self, encoded, reuse = False, training = True):
        config = self.config

        with(tf.variable_scope('generator', reuse=reuse)):
            dense_size = np.prod(config.first_decode_shape)
            cur = tf.layers.dense(encoded, dense_size, activation = tf.tanh)
            cur = tf.reshape(cur, [-1, *config.first_decode_shape])
            w, h = config.first_decode_shape[1], config.first_decode_shape[0]

            for conv in config.g_convs:
                cur = tf.layers.conv2d_transpose(cur, conv.num_features, conv.kernel, conv.stride, padding = 'same')
                cur = tf.layers.batch_normalization(cur, training = training)
                cur = leaky_relu(cur, config.alpha)
                w, h = 2 * w, 2 * h
                #cur = tf.image.resize_images(cur, (w, h), method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            cur = tf.layers.conv2d(cur, config.orig_channels, 1, padding = 'same')
        return cur

    def run_autoencoder(self, inputs, training = True):
        self.ae_inputs = inputs
        self.labels, self.code = self.get_code(inputs, training = training)
        encoded = tf.concat((tf.nn.softmax(self.labels), tf.tanh(self.code)), axis = 1)
        self.decoded = self.get_decoded(encoded, training = training)
        self.img = tf.tanh(self.decoded)

    def ae_loss(self, targets, images):
        ae_loss = tf.reduce_sum(tanh_cross_entropy(self.decoded, images))
        labels_mask = tf.reduce_sum(targets, axis = -1)
        classification_loss = tf.reduce_sum(labels_mask * tf.nn.softmax_cross_entropy_with_logits(labels = targets, logits = self.labels))
        return classification_loss + self.config.mu * ae_loss

    def get_num_classified(self, targets, labels):
        labels_mask = tf.reduce_sum(targets, axis = -1)
        classified = tf.equal(tf.argmax(labels, axis = -1), tf.argmax(targets, axis = -1))
        return tf.reduce_sum(tf.cast(classified, tf.float32) * labels_mask)

    def num_classified_ae(self, targets, images):
        return self.get_num_classified(targets, self.labels)

    def get_minimizers_ae(self, optimizer, loss):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            minimizers = [optimizer.minimize(l) for l in loss]
        return minimizers

    def is_fake(self, code, reuse = False):
        with (tf.variable_scope('discriminator_add', reuse = reuse)):
            fake_detector = tf.layers.dense(code, 1)
        return fake_detector

    def run_semi_supervised(self, images, randoms, training = True):
        self.generated = tf.tanh(self.get_decoded(randoms, reuse = tf.AUTO_REUSE, training = training))
        self.gen_labels = randoms[:, :self.config.num_outputs]

        classification, codes = self.get_code(tf.concat((images, self.generated), axis = 0), reuse = tf.AUTO_REUSE, training = training)
        sizes = [tf.shape(images)[0], tf.shape(randoms)[0]]
        self.classification_orig, self.classification_fake = tf.split(classification, sizes)
        #code_orig, code_fake = tf.split(codes, sizes)
        self.orig, self.fakes = tf.split(self.is_fake(tf.tanh(codes)), sizes)
        #self.fakes = self.is_fake(tf.tanh(code_fake), reuse = True)

    def semi_loss(self, targets, images):
        labels_mask = tf.reduce_sum(targets, axis = -1)

        classification_loss_d = tf.reduce_sum(labels_mask * tf.nn.softmax_cross_entropy_with_logits(labels = targets, logits = self.classification_orig))
        classification_loss_g = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = self.gen_labels, logits = self.classification_fake))
        d_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.orig), logits = self.orig))
        d_loss += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(self.fakes), logits = self.fakes))
        d_loss += classification_loss_d # + 0.1 * classification_loss_g

        g_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.fakes), logits = self.fakes)) # + 0.1 * classification_loss_g
        #g_loss = classification_loss_g
        return d_loss, g_loss

    def num_classified_semi(self, targets, images):
        return self.get_num_classified(targets, self.classification_orig)

    def get_minimizers_semi(self, optimizer, loss):
        all_vars = tf.trainable_variables()
        d_vars = [v for v in all_vars if v.name.startswith('discriminator')]
        g_vars = [v for v in all_vars if v.name.startswith('generator')]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            minimizer_d = optimizer.minimize(loss[0], var_list = d_vars)
            minimizer_g = optimizer.minimize(loss[1], var_list = g_vars)
            #minimizer = optimizer.optimize(loss[0], loss[1], d_vars, g_vars)
        return [minimizer_d, minimizer_g]

    def get_ae_functions_for_trainer(self):
        return 'ae_loss', 'run_autoencoder', 'num_classified_ae', 'get_minimizers_ae'

    def get_semi_functions_for_trainer(self):
        return 'semi_loss', 'run_semi_supervised', 'num_classified_semi', 'get_minimizers_semi'












class BasicNet():
    def __init__(self, modes_dict, mode, elements_dict = None, config = None):
        self.modes = modes_dict
        self.config = config
        self.runner = None

        self.mode = None
        self.set_mode(mode)
        self.run_elements = None
        self.set_elements(elements_dict)

    def set_mode(self, mode):
        self.mode = mode
        self.runner = self.modes[self.mode]

    def set_elements(self, elements_dict):
        self.run_elements = elements_dict

    def run(self, *args, **kwargs):
        print('Run:', self.mode)
        return self.runner.run(self.config, self.run_elements, *args, **kwargs)

    def loss_function(self, *args, **kwargs):
        print('Loss:', self.mode)
        return self.runner.loss_function(self.config, *args, **kwargs)

    def num_classified(self, *args, **kwargs):
        return self.runner.num_classified(self.config, *args, **kwargs)

    def get_minimizers(self, *args, **kwargs):
        return self.runner.get_minimizers(*args, **kwargs)

    def get_functions_for_trainer(self):
        return 'loss_function', 'run', 'num_classified', 'get_minimizers'

    def get_runner(self, mode = None):
        return self.runner if mode is None else self.modes[mode]

class SemiCapsCodeGenerator():
    def __init__(self, num_outputs, caps_len):
        self.num_outputs = num_outputs
        self.caps_len = caps_len
        self.code_size = num_outputs * caps_len

    def get_code(self, batch_size):
        random_targets = np.eye(self.num_outputs)[np.random.randint(0, self.num_outputs, size = [batch_size])]
        random_code = 2 * np.random.rand(batch_size, self.code_size) - 1 # shoud be like after tanh
        randoms = np.concatenate((random_targets, random_code), axis = 1)
        return self.preprocess_code(randoms)

    def preprocess_code(self, code):
        labels, data = code[:, :self.num_outputs], code[:, self.num_outputs:]
        data = np.reshape(data, [-1, self.num_outputs, self.caps_len])
        labels = np.expand_dims(labels, -1)
        return labels * data # masking out all except capsule for label

#
# With Capsules
#
class SemiSupLossConfig():
    def __init__(self, label_smoothing = 0.9):
        self.label_smoothing = label_smoothing
        self.margin_m_plus = label_smoothing
        self.margin_m_minus = 0.0
        self.margin_lambda = 0.9

        self.reconst_coef_g = 0.8
        self.reconst_coef_d = 0.3
        self.gen_classify_coef_g = 0.5
        self.gen_classify_coef_d = 0.2

        self.prediction_coef = 0.5

        self.use_feature_matching = False

class SemiSupCapsConfig():
    def __init__(self, num_outputs = 10, caps1_len = 8, caps2_len = 16, capsg_len = 8, capsg_size = [8, 8, 32], loss_config = SemiSupLossConfig()):
        # discriminator
        self.conv1_info = ConvData(128, 5, 2)
        self.conv2_info = ConvData(128, 5, 2)
        self.caps1_len = caps1_len

        self.caps2_len = caps2_len
        self.num_outputs = num_outputs
        self.code_size = caps2_len * num_outputs
        self.leaky_alpha = 0.01

        self.minibatch_num = 32
        self.minibatch_len = 16

        # generator
        self.capsg_len = capsg_len
        self.capsg_size = capsg_size
        self.deconv1_info = ConvData(128, 5, 2)
        self.deconv2_info = ConvData(3, 5, 2)

        self.use_minibatch = True
        self.with_reconstruction = False
        self.loss_config = loss_config
'''
def batch_norm_with_ref(data, ref_size, training, name):
    x = data
    ref_size = 0 if ref_size is None else ref_size
    x, bn_data = tf.split(x, [tf.shape(x)[0] - ref_size, ref_size])

    if ref_size != 0:
        bn_data = tf.layers.batch_normalization(bn_data, training = training, name = name, renorm = True)
        training = False

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        x = tf.layers.batch_normalization(x, training = training, name = name, reuse = tf.AUTO_REUSE)

    x = tf.concat((x, bn_data), 0)
    return x

class SemiSupCapsNet():
    def __init__(self, config = SemiSupCapsConfig()):
        self.config = config
        self.ref_batch = None
        self.code_gen = SemiCapsCodeGenerator(config.num_outputs, config.caps2_len)

    def set_reference_batch(self, ref_batch):
        # first num_outputs is a label
        ref_batch = np.reshape(ref_batch, [-1, self.config.num_outputs, self.config.caps2_len])
        self.ref_batch = tf.convert_to_tensor(ref_batch, dtype = tf.float32)

    def run_encoder(self, inputs, reuse = False, training = False):
        config = self.config

        with(tf.variable_scope('discriminator', reuse=reuse)):
            conv1_info = config.conv1_info
            conv1 = tf.layers.conv2d(inputs, conv1_info.num_features, conv1_info.kernel, conv1_info.stride, padding = 'same')
            conv1 = tf.layers.batch_normalization(conv1, training = training)
            conv1 = leaky_relu(conv1, config.leaky_alpha)
            conv1 = tf.layers.dropout(conv1, 0.5, training = training)

            conv2_info = config.conv2_info
            conv2 = tf.layers.conv2d(conv1, conv2_info.num_features, conv2_info.kernel, conv2_info.stride, padding = 'same')
            conv2 = tf.layers.batch_normalization(conv2, training = training) # do we need it?
            conv2 = tf.layers.dropout(conv2, 0.5, training = training)

            caps1 = squash(reshapeToCapsules(tf.transpose(conv2, [0, 3, 1, 2]), config.caps1_len, axis = 1))
            caps_layer = CapsLayer(caps1, config.num_outputs, config.caps2_len)
            caps2 = caps_layer.do_dynamic_routing()
            #caps2 = tf.layers.dropout(caps2, 0.5, training = training)
            return caps2

    def get_masked_code(self, code):
        masked = tf.multiply(code, maskForMaxCapsule(l1norm(code)))
        return masked

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

    def generate_from_code(self, code, reuse = False, training = True):
        config = self.config

        ref_size, code = self.get_virtual_batch_data(code)

        with(tf.variable_scope('generator', reuse=reuse)):
            caps0 = squash(code)
            # only one active capsule expected, hence r = 1
            caps_layer0 = CapsLayer(caps0, config.num_outputs, config.caps2_len, name = 'caps_layer_g_0', r = 1)
            caps1 = caps_layer0.do_dynamic_routing()

            caps_layer = CapsLayer(caps1, np.prod(config.capsg_size), config.capsg_len, name = 'caps_layer_g_1')
            caps2 = caps_layer.do_dynamic_routing()
            reshaped = tf.reshape(caps2, [-1, config.capsg_size[0], config.capsg_size[1], config.capsg_size[2] * config.capsg_len])
            reshaped = tf.verify_tensor_all_finite(reshaped, 'Not all values are finite befire VBN1')
            reshaped = batch_norm_with_ref(reshaped, ref_size, training = training, name = 'ref_batch_norm1')
            reshaped = tf.verify_tensor_all_finite(reshaped, 'Not all values are finite after VBN1')

            deconv1_info = config.deconv1_info
            gconv1 = tf.layers.conv2d_transpose(reshaped, *(deconv1_info.get_conv_data()))
            gconv1 = tf.verify_tensor_all_finite(gconv1, 'Not all values are finite before VBN2')
            gconv1 = batch_norm_with_ref(gconv1, ref_size, training = training, name = 'ref_batch_norm2')
            gconv1 = tf.verify_tensor_all_finite(gconv1, 'Not all values are finite after VBN2')
            gconv1 = leaky_relu(gconv1, config.leaky_alpha)
            gconv1 = tf.layers.dropout(gconv1, 0.5, training = training)

            deconv2_info = config.deconv2_info
            gconv2 = tf.layers.conv2d_transpose(gconv1, *(deconv2_info.get_conv_data()))
            return self.return_from_virtual_batch(gconv2, ref_size)

    def add_minibatch_data(self, x, reuse = False):
        config = self.config

        batch, l = tf.shape(x)[0], x.get_shape().as_list()[1]
        with(tf.variable_scope('discriminator_minibatch', reuse=reuse)):
            W = tf.get_variable('minibatch_weights', [l, config.minibatch_num * config.minibatch_len], initializer = tf.truncated_normal_initializer())
            minibatch_data = tf.matmul(x, W)
            #TODO: find better way
            diff = tf.expand_dims(minibatch_data, -1) - tf.transpose(minibatch_data, [1, 0])
            diff = tf.reshape(diff, [batch, config.minibatch_num, config.minibatch_len, batch])
            diff_norm = l1norm(diff, axis = -2) # euclidean norm here results in nan in gradients (of generator's convolution...)
            # diff_norm has zeros on main diagonal, hence subtract one.
            addition = tf.reduce_sum(tf.exp(-diff_norm), axis = -1) - 1
        return tf.concat((x, addition), axis = -1)

    def is_fake(self, code, reuse = False):
        with (tf.variable_scope('discriminator_add', reuse = reuse)):
            fake_detector = tf.layers.dense(code, 1)
        return fake_detector

    def run(self, inputs, code, training = True):
        config = self.config
        # code should have shape [batch_size x num_labels x caps_len]
        self.code_labels = tf.one_hot(tf.argmax(l1norm(code), axis = -1), config.num_outputs)

        # generate images from code
        self.generated = self.generate_from_code(code, False, training)

        # combine generated images with real ones and run classificator
        images = tf.concat((inputs, tf.tanh(self.generated)), axis = 0)
        sizes = [tf.shape(inputs)[0], tf.shape(self.generated)[0]]
        self.codes = self.run_encoder(images, False, training)
        self.inputs_code, self.gen_code = tf.split(self.codes, sizes)

        # mask out all capsules but the one with largest norm and add minibatch features (if needed)
        features = tf.contrib.layers.flatten(self.get_masked_code(self.codes))
        self.real_features, self.fake_features = tf.split(features, sizes)
        if config.use_minibatch:
            self.real_features = self.add_minibatch_data(self.real_features)
            self.fake_features = self.add_minibatch_data(self.fake_features, reuse = True)
        output_features = tf.concat((self.real_features, self.fake_features), 0)

        output_features = tf.verify_tensor_all_finite(output_features, 'Not all values are finite within minibatch')

        if config.with_reconstruction:
            self.reconstructed = self.generate_from_code(self.inputs_code, True, training)
        # discriminate fake and original images
        self.orig, self.fake = tf.split(self.is_fake(output_features), sizes)

    def get_feature_matching_loss(self):
        num_features = tf.cast(tf.shape(self.real_features)[1], tf.float32)
        return l1norm(tf.reduce_mean(self.real_features, 0) - tf.reduce_mean(self.fake_features, 0))# / num_features

    def get_reconstruction_loss(self, images):
        condition = tf.constant(self.config.with_reconstruction)
        calc_reconstruction_loss = lambda: tf.reduce_sum(tf.reduce_mean(tanh_cross_entropy(logits = self.reconstructed, labels = images), axis = [-1, -2, -3]))
        return tf.cond(condition, calc_reconstruction_loss, lambda: 0.0)

    def get_loss_on_targets(self, targets, outputs, targets_mask):
        loss_config = self.config.loss_config
        m_plus = loss_config.margin_m_plus
        m_minus = loss_config.margin_m_minus

        margin_loss = targets * tf.square(m_plus - outputs)
        margin_loss += loss_config.margin_lambda * (1.0 - targets) * tf.square(outputs - m_minus)
        loss = targets_mask * tf.reduce_sum(margin_loss, axis = 1)

        return (tf.to_float(tf.shape(targets)[0]) * tf.reduce_sum(loss)) / (tf.reduce_sum(targets_mask) + 1e-6)

    def get_fake_detection_loss(self, labels, logits):
        smooth = self.config.loss_config.label_smoothing
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = smooth * labels, logits = logits)
        return tf.reduce_sum(loss)

    def loss_function(self, targets, images):
        lconfig = self.config.loss_config

        loss_on_targets = self.get_loss_on_targets(targets, l1norm(self.inputs_code), tf.reduce_sum(targets, axis = -1))
        loss_on_gen = self.get_loss_on_targets(self.code_labels, l1norm(self.gen_code), tf.ones([tf.shape(self.code_labels)[0]], dtype = tf.float32))
        reconstruction_loss = self.get_reconstruction_loss(images)
        orig_detection_loss = self.get_fake_detection_loss(labels = tf.ones_like(self.orig), logits = self.orig)
        fake_detection_loss = self.get_fake_detection_loss(labels = tf.zeros_like(self.fake), logits = self.fake)
        camouflage_loss = self.get_fake_detection_loss(labels = tf.ones_like(self.fake), logits = self.fake)
        feature_matching_loss = self.get_feature_matching_loss()

        d_loss = loss_on_targets + lconfig.reconst_coef_d * reconstruction_loss
        d_loss += orig_detection_loss + fake_detection_loss
        d_loss += lconfig.gen_classify_coef_d * loss_on_gen

        #g_loss = tf.reduce_sum(camouflage_loss)
        g_loss = feature_matching_loss + lconfig.reconst_coef_g * reconstruction_loss
        g_loss += lconfig.gen_classify_coef_g * loss_on_gen
        return d_loss, g_loss, loss_on_targets, orig_detection_loss, fake_detection_loss, loss_on_gen, reconstruction_loss

    def get_num_classified(self, targets, labels):
        labels_mask = tf.reduce_sum(targets, axis = -1)
        classified = tf.equal(tf.argmax(labels, axis = -1), tf.argmax(targets, axis = -1))
        return tf.reduce_sum(tf.cast(classified, tf.float32) * labels_mask)

    def num_classified(self, targets, images):
        return self.get_num_classified(targets, norm(self.inputs_code))

    def get_minimizers(self, optimizer, loss):
        all_vars = tf.trainable_variables()
        d_vars = [v for v in all_vars if v.name.startswith('discriminator')]
        g_vars = [v for v in all_vars if v.name.startswith('generator')]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            minimizer_d = optimizer.minimize(loss[0], var_list = d_vars)
            minimizer_g = optimizer.minimize(loss[1], var_list = g_vars)
            #minimizer = optimizer.optimize(loss[0], loss[1], d_vars, g_vars)
        return [minimizer_d, minimizer_g]

    def get_functions_for_trainer(self):
        return 'loss_function', 'run', 'num_classified', 'get_minimizers'

    def get_code_generator(self):
        return self.code_gen
'''
class SemiCapsNet(BasicNet):
    def __init__(self, config = SemiSupCapsConfig()):
        self.ref_batch = None
        self.code_gen = SemiCapsCodeGenerator(config.num_outputs, config.caps2_len)

        modes_dict = {
            'semi-supervised': GanRunner(),
            'gan' : GanRunner(semi_supervised = False),
            'ae' : AERunner()
            }
        element_dict = {
            'generator': CapsGenerator(),
            'encoder': CapsEncoder(),
            'predictor': CapsPredict(),
            'minibatcher': Minibatcher(),
            'labeler': CapsCodePrepare(),
            'extractor': CapsFeatureExtractor(),
            'fake_detector': DenseFakeDetector(),
            }
        super(SemiCapsNet, self).__init__(modes_dict, 'semi-supervised', element_dict, config = config)

    def set_reference_batch(self, ref_batch):
        self.run_elements['generator'].set_ref_batch(ref_batch)

    def get_code_generator(self):
        return self.code_gen


